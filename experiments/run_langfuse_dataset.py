from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langfuse import Langfuse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluation import (
    DEFAULT_EVALUATION_CHANNEL,
    build_experiment_output,
    build_langfuse_evaluations,
    extract_dataset_case,
    normalize_evaluation_session_id,
)


DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_CONCURRENCY = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Langfuse dataset against the Gala /evaluate endpoint.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("LANGFUSE_DATASET_NAME", "").strip(),
        help="Langfuse dataset name.",
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("LANGFUSE_EXPERIMENT_NAME", "").strip(),
        help="Experiment name and dataset run name in Langfuse.",
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("EVALUATION_BACKEND_URL", "").strip(),
        help="Backend base URL, for example http://localhost:5000.",
    )
    parser.add_argument(
        "--endpoint-token",
        default=os.getenv("EVALUATION_ENDPOINT_TOKEN", "").strip(),
        help="Optional X-Eval-Token header value.",
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("EVALUATION_CHANNEL", DEFAULT_EVALUATION_CHANNEL).strip()
        or DEFAULT_EVALUATION_CHANNEL,
        help="Channel value sent to /evaluate.",
    )
    parser.add_argument(
        "--description",
        default=os.getenv("LANGFUSE_EXPERIMENT_DESCRIPTION", "").strip() or None,
        help="Optional description shown in Langfuse.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_int_env("EVALUATION_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        help="HTTP timeout per /evaluate request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=_int_env("LANGFUSE_EXPERIMENT_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY),
        help="Maximum concurrent dataset items.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    _validate_args(args)

    langfuse = _build_langfuse_client()
    dataset = langfuse.get_dataset(args.dataset)
    if not dataset.items:
        raise RuntimeError(f"Dataset '{args.dataset}' has no items.")

    evaluate_url = _normalize_evaluate_url(args.backend_url)
    git_commit = _git_commit()
    experiment_metadata = {
        "dataset_name": args.dataset,
        "backend_url": args.backend_url,
        "evaluate_url": evaluate_url,
        "git_commit": git_commit,
        "channel": args.channel,
    }

    task = _build_task(
        dataset_name=args.dataset,
        experiment_name=args.experiment,
        backend_url=args.backend_url,
        evaluate_url=evaluate_url,
        endpoint_token=args.endpoint_token,
        channel=args.channel,
        timeout_seconds=max(1, args.timeout_seconds),
        git_commit=git_commit,
    )

    result = dataset.run_experiment(
        name=args.experiment,
        run_name=args.experiment,
        description=args.description,
        task=task,
        evaluators=[_item_evaluator],
        max_concurrency=max(1, args.max_concurrency),
        metadata=experiment_metadata,
    )
    langfuse.flush()

    print(result.format())
    if result.dataset_run_url:
        print(f"\nDataset run URL: {result.dataset_run_url}")

    return 0


def _build_task(
    *,
    dataset_name: str,
    experiment_name: str,
    backend_url: str,
    evaluate_url: str,
    endpoint_token: str,
    channel: str,
    timeout_seconds: int,
    git_commit: str | None,
):
    def _task(*, item: Any, **_kwargs: dict[str, Any]) -> dict[str, Any]:
        item_input = _item_field(item, "input")
        item_expected_output = _item_field(item, "expected_output")
        item_metadata = _item_field(item, "metadata")
        item_id = _item_field(item, "id")
        case = extract_dataset_case(
            input_value=item_input,
            expected_output=item_expected_output,
            metadata=item_metadata if isinstance(item_metadata, dict) else {},
            item_id=str(item_id) if item_id else None,
        )
        question = case.get("question") or ""
        session_id = normalize_evaluation_session_id(
            case.get("provided_session_id"),
            fallback_seed=str(case.get("item_id") or question or dataset_name),
        )

        if not str(question).strip():
            output = _blank_backend_output(
                session_id=session_id,
                error_message="dataset_item_missing_question",
            )
            return build_experiment_output(
                output,
                case={
                    **case,
                    "dataset_name": dataset_name,
                },
                experiment_name=experiment_name,
                backend_url=backend_url,
                git_commit=git_commit,
            )

        request_metadata = {
            **(item_metadata if isinstance(item_metadata, dict) else {}),
            "dataset_name": dataset_name,
            "dataset_item_id": case.get("item_id"),
            "experiment_name": experiment_name,
            "case_type": case.get("case_type"),
            "expected_route": case.get("expected_route"),
            "expected_topic": case.get("expected_topic"),
            "expected_behavior": case.get("expected_behavior"),
        }
        payload = {
            "question": question,
            "session_id": session_id,
            "channel": channel,
            "metadata": {
                key: value
                for key, value in request_metadata.items()
                if value not in (None, "", [], {})
            },
        }
        headers = {
            "Content-Type": "application/json",
        }
        if endpoint_token:
            headers["X-Eval-Token"] = endpoint_token

        backend_response = _call_evaluate_endpoint(
            evaluate_url=evaluate_url,
            payload=payload,
            headers=headers,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
        )
        experiment_output = build_experiment_output(
            backend_response,
            case={
                **case,
                "dataset_name": dataset_name,
            },
            experiment_name=experiment_name,
            backend_url=backend_url,
            git_commit=git_commit,
        )
        experiment_output["evaluation_metadata"].update(
            {
                "http_status": backend_response.get("http_status"),
                "request_payload": payload,
                "expected_output": item_expected_output,
            }
        )
        return experiment_output

    return _task


def _call_evaluate_endpoint(
    *,
    evaluate_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    session_id: str,
) -> dict[str, Any]:
    try:
        response = requests.post(
            evaluate_url,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        output = _blank_backend_output(
            session_id=session_id,
            error_message=f"{type(exc).__name__}: {str(exc)}",
        )
        output["http_status"] = None
        return output

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = {
            "error": "invalid_json_response",
            "raw_text": response.text[:500],
        }

    if response.ok and isinstance(response_payload, dict):
        output = _blank_backend_output(session_id=session_id)
        output.update(response_payload)
        output["http_status"] = response.status_code
        return output

    error_message = _extract_error_message(response_payload)
    output = _blank_backend_output(
        session_id=session_id,
        error_message=error_message or f"http_{response.status_code}",
    )
    output["http_status"] = response.status_code
    output["backend_error_body"] = response_payload
    return output


def _item_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
    **_kwargs: dict[str, Any],
) -> list[Any]:
    item_id = None
    if isinstance(output, dict):
        evaluation_metadata = output.get("evaluation_metadata")
        if isinstance(evaluation_metadata, dict):
            item_id = evaluation_metadata.get("dataset_item_id")

    return build_langfuse_evaluations(
        item_input=input,
        output=output if isinstance(output, dict) else {},
        expected_output=expected_output,
        metadata=metadata,
        item_id=str(item_id) if item_id else None,
    )


def _build_langfuse_client() -> Langfuse:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

    if not public_key or not secret_key:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required to run experiments."
        )

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    if not client.auth_check():
        raise RuntimeError("Langfuse auth_check failed. Verify your credentials.")
    return client


def _validate_args(args: argparse.Namespace) -> None:
    if not args.dataset:
        raise RuntimeError("Missing dataset name. Use --dataset or LANGFUSE_DATASET_NAME.")
    if not args.experiment:
        raise RuntimeError(
            "Missing experiment name. Use --experiment or LANGFUSE_EXPERIMENT_NAME."
        )
    if not args.backend_url:
        raise RuntimeError(
            "Missing backend URL. Use --backend-url or EVALUATION_BACKEND_URL."
        )


def _normalize_evaluate_url(backend_url: str) -> str:
    cleaned = backend_url.rstrip("/")
    if cleaned.endswith("/evaluate"):
        return cleaned
    return f"{cleaned}/evaluate"


def _blank_backend_output(
    *,
    session_id: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "answer": None,
        "route": None,
        "topic": None,
        "used_rag": None,
        "used_tool": None,
        "fallback": None,
        "needs_clarification": None,
        "guardrail_blocked": None,
        "documents_count": None,
        "needs_human_review": None,
        "dataset_candidate": None,
        "session_id": session_id,
        "trace_id": None,
        "latency_ms": None,
        "app_version": None,
        "graph_version": None,
        "evaluation_error": error_message,
    }


def _extract_error_message(response_payload: Any) -> str | None:
    if isinstance(response_payload, dict):
        for key in ("message", "error", "detail"):
            value = response_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(response_payload, ensure_ascii=True)[:300]

    if isinstance(response_payload, str) and response_payload.strip():
        return response_payload.strip()[:300]

    return None


def _item_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = completed.stdout.strip()
    return commit or None


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return max(1, int(raw_value or default))
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
