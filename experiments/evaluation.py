from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

from observability.evaluators import evaluate_whatsapp_format
from observability.metrics import build_final_trace_metadata


FALSE_ENV_VALUES = {"0", "false", "no", "off"}
DEV_ENVIRONMENTS = {"local", "dev", "development", "test"}
DEFAULT_EVALUATION_CHANNEL = "langfuse_experiment"
DEFAULT_EVALUATION_ENTRYPOINT = "evaluate_endpoint"
DEFAULT_EVALUATION_USER_ID = "eval-user"
DEFAULT_EVALUATION_TAGS = [
    "gala",
    "langgraph",
    "evaluation",
    "dataset",
    "langfuse_experiment",
    "dataset_run",
]
ROUTE_ALIASES = {
    "rag": "loans_rag",
    "loans_rag": "loans_rag",
    "bcra_agent": "bcra_agent",
    "bcra_credit_status": "bcra_agent",
    "branch_locator": "branch_locator",
    "benefits": "benefits",
    "credit_card_statement": "credit_card_statement",
    "chitchat": "chitchat",
    "fallback": "fallback",
    "sensitive": "sensitive",
}
TOPIC_ALIASES = {
    "prestamos": "prestamos",
    "situacion_crediticia": "situacion_crediticia",
    "situacion_crediticia_bcra": "situacion_crediticia",
    "sucursales": "sucursales",
    "sucursales_cercanas": "sucursales",
    "beneficios": "beneficios",
    "resumen_tarjeta": "resumen_tarjeta",
    "saludo": "saludo",
    "conversacion": "saludo",
}
TOOL_EXPECTED_ROUTES = {
    "benefits",
    "bcra_agent",
    "branch_locator",
    "credit_card_statement",
}


def is_local_dev_environment(app_env: str | None = None) -> bool:
    return _slugify(app_env or os.getenv("APP_ENV", "dev")) in DEV_ENVIRONMENTS


def is_evaluation_endpoint_enabled() -> bool:
    raw_value = os.getenv("EVALUATION_ENDPOINT_ENABLED")
    if raw_value is None:
        return is_local_dev_environment()

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_evaluation_request_authorized(provided_token: str | None) -> bool:
    required_token = (os.getenv("EVALUATION_ENDPOINT_TOKEN", "") or "").strip()
    if required_token:
        return bool(provided_token) and provided_token == required_token

    return is_local_dev_environment()


def normalize_evaluation_session_id(
    session_id: str | None,
    *,
    fallback_seed: str | None = None,
) -> str:
    candidate = _safe_text(session_id)
    if candidate:
        if candidate.lower().startswith("eval-"):
            return candidate

        normalized = _slugify(candidate)
        if normalized:
            return f"eval-{normalized}"

    seed = _slugify(fallback_seed) or str(int(time.time() * 1000))
    return f"eval-{seed}"


def build_evaluation_langfuse_user_id(metadata: dict[str, Any] | None = None) -> str:
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    explicit_user_id = _safe_text(safe_metadata.get("langfuse_user_id"))
    if explicit_user_id:
        return normalize_evaluation_session_id(explicit_user_id)

    dataset_name = _safe_text(safe_metadata.get("dataset_name"))
    if dataset_name:
        return normalize_evaluation_session_id(dataset_name)

    return DEFAULT_EVALUATION_USER_ID


def build_evaluation_langfuse_tags(
    *,
    channel: str | None = None,
    extra_tags: list[str] | None = None,
) -> list[str]:
    tags: list[str] = []
    for tag in [*DEFAULT_EVALUATION_TAGS, *(extra_tags or [])]:
        cleaned = _safe_text(tag)
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    cleaned_channel = _safe_text(channel)
    if cleaned_channel and cleaned_channel not in tags:
        tags.append(cleaned_channel)

    return tags


def build_evaluation_trace_metadata(
    *,
    session_id: str,
    channel: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    trace_metadata = dict(safe_metadata)
    trace_metadata.update(
        {
            "evaluation_mode": True,
            "evaluation_channel": _safe_text(channel) or DEFAULT_EVALUATION_CHANNEL,
            "evaluation_session_id": session_id,
            "evaluation_tags": ["evaluation", "langfuse_experiment", "dataset_run"],
        }
    )
    return trace_metadata


def build_evaluation_response(
    result: dict[str, Any] | None,
    *,
    session_id: str,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_result = result if isinstance(result, dict) else {}
    safe_trace_context = trace_context if isinstance(trace_context, dict) else {}
    initial_trace_metadata = _as_dict(safe_trace_context.get("initial_trace_metadata"))
    final_trace_metadata = _as_dict(safe_trace_context.get("final_trace_metadata"))

    if not final_trace_metadata:
        final_trace_metadata = build_final_trace_metadata(
            safe_result,
            total_latency_ms=_int_or_none(safe_trace_context.get("total_latency_ms")),
        )

    quality_payload = _as_dict(safe_trace_context.get("quality_payload"))
    quality_metadata = _as_dict(quality_payload.get("metadata"))

    response = {
        "answer": _optional_text(
            safe_result.get("final_answer") or safe_result.get("answer")
        ),
        "route": _optional_text(final_trace_metadata.get("final_route")),
        "topic": _optional_text(final_trace_metadata.get("final_topic")),
        "used_rag": _bool_or_none(final_trace_metadata.get("used_rag")),
        "used_tool": _bool_or_none(final_trace_metadata.get("used_tool")),
        "fallback": _bool_or_none(final_trace_metadata.get("fallback")),
        "needs_clarification": _bool_or_none(
            final_trace_metadata.get("needs_clarification")
        ),
        "guardrail_blocked": _bool_or_none(final_trace_metadata.get("guardrail_blocked")),
        "documents_count": _int_or_none(final_trace_metadata.get("documents_count")),
        "needs_human_review": _quality_flag(
            safe_result,
            quality_metadata,
            "needs_human_review",
        ),
        "dataset_candidate": _quality_flag(
            safe_result,
            quality_metadata,
            "dataset_candidate",
        ),
        "session_id": session_id,
        "trace_id": _optional_text(safe_trace_context.get("trace_id")),
        "latency_ms": _int_or_none(
            safe_trace_context.get("total_latency_ms")
            or final_trace_metadata.get("total_latency_ms")
        ),
        "app_version": _optional_text(initial_trace_metadata.get("app_version")),
        "graph_version": _optional_text(initial_trace_metadata.get("graph_version")),
    }
    return response


def extract_dataset_case(
    *,
    input_value: Any,
    expected_output: Any = None,
    metadata: dict[str, Any] | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    input_dict = _as_dict(input_value)
    expected_dict = _as_dict(expected_output)
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    merged = {
        **metadata_dict,
        **input_dict,
        **expected_dict,
    }

    return {
        "item_id": _optional_text(item_id) or _optional_text(merged.get("id")),
        "dataset_name": _optional_text(merged.get("dataset_name")),
        "question": _extract_question(input_value, merged),
        "expected_route": _optional_text(
            merged.get("expected_route") or merged.get("route")
        ),
        "expected_topic": _optional_text(
            merged.get("expected_topic") or merged.get("topic")
        ),
        "expected_behavior": _optional_text(merged.get("expected_behavior")),
        "expected_used_rag": _bool_or_none(merged.get("expected_used_rag")),
        "expected_used_tool": _bool_or_none(merged.get("expected_used_tool")),
        "case_type": _optional_text(merged.get("case_type")),
        "must_include": _string_list(merged.get("must_include")),
        "must_not_include": _string_list(
            merged.get("must_not_include") or merged.get("forbidden_terms")
        ),
        "provided_session_id": _optional_text(merged.get("session_id")),
        "expected_output_text": (
            _optional_text(expected_output)
            if isinstance(expected_output, str)
            else _optional_text(expected_dict.get("expected_output"))
        ),
        "raw_input": input_value,
        "raw_expected_output": expected_output,
        "raw_metadata": metadata_dict,
    }


def evaluate_scores(
    *,
    item_input: Any,
    output: dict[str, Any] | None,
    expected_output: Any = None,
    metadata: dict[str, Any] | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    case = extract_dataset_case(
        input_value=item_input,
        expected_output=expected_output,
        metadata=metadata,
        item_id=item_id,
    )
    safe_output = output if isinstance(output, dict) else {}
    answer = _safe_text(safe_output.get("answer"))
    actual_route = _optional_text(safe_output.get("route"))
    actual_topic = _optional_text(safe_output.get("topic"))
    used_rag = _bool_or_none(safe_output.get("used_rag"))
    used_tool = _bool_or_none(safe_output.get("used_tool"))
    fallback = _bool_or_none(safe_output.get("fallback"))

    must_include = case["must_include"]
    must_not_include = case["must_not_include"]
    normalized_answer = _normalize_text(answer)

    route_match = None
    if case["expected_route"]:
        route_match = (
            _normalize_route(case["expected_route"]) == _normalize_route(actual_route)
        )

    topic_match = None
    if case["expected_topic"]:
        topic_match = (
            _normalize_topic(case["expected_topic"]) == _normalize_topic(actual_topic)
        )

    must_include_coverage = None
    if must_include:
        matched_terms = sum(
            1 for term in must_include if _normalize_text(term) in normalized_answer
        )
        must_include_coverage = round(matched_terms / len(must_include), 4)

    forbidden_terms_avoided = None
    if must_not_include:
        forbidden_terms_avoided = not any(
            _normalize_text(term) in normalized_answer for term in must_not_include
        )

    used_rag_expected = None
    expected_used_rag = case["expected_used_rag"]
    if expected_used_rag is not None:
        used_rag_expected = used_rag == expected_used_rag
    elif _expects_rag(case):
        used_rag_expected = used_rag is True

    used_tool_expected = None
    expected_used_tool = case["expected_used_tool"]
    if expected_used_tool is not None:
        used_tool_expected = used_tool == expected_used_tool
    elif _expects_tool(case):
        used_tool_expected = used_tool is True

    whatsapp_eval = evaluate_whatsapp_format(answer)

    return {
        "route_match": route_match,
        "topic_match": topic_match,
        "fallback_avoided": None if fallback is None else not fallback,
        "used_rag_expected": used_rag_expected,
        "used_tool_expected": used_tool_expected,
        "must_include_coverage": must_include_coverage,
        "forbidden_terms_avoided": forbidden_terms_avoided,
        "needs_human_review_after_run": _bool_or_none(
            safe_output.get("needs_human_review")
        ),
        "dataset_candidate_after_run": _bool_or_none(
            safe_output.get("dataset_candidate")
        ),
        "answer_non_empty": bool(answer.strip()),
        "whatsapp_format_basic_ok": bool(whatsapp_eval.get("whatsapp_format_ok", 0)),
    }


def build_langfuse_evaluations(
    *,
    item_input: Any,
    output: dict[str, Any] | None,
    expected_output: Any = None,
    metadata: dict[str, Any] | None = None,
    item_id: str | None = None,
) -> list[Any]:
    from langfuse.experiment import Evaluation

    case = extract_dataset_case(
        input_value=item_input,
        expected_output=expected_output,
        metadata=metadata,
        item_id=item_id,
    )
    scores = evaluate_scores(
        item_input=item_input,
        output=output,
        expected_output=expected_output,
        metadata=metadata,
        item_id=item_id,
    )
    evaluations: list[Evaluation] = []

    for name, value in scores.items():
        if value is None:
            continue

        comment = _score_comment(name, value, case=case, output=output)
        evaluation_metadata = {
            "case_type": case.get("case_type"),
            "dataset_item_id": case.get("item_id"),
            "expected_route": case.get("expected_route"),
            "expected_topic": case.get("expected_topic"),
        }
        evaluations.append(
            Evaluation(
                name=name,
                value=value,
                comment=comment,
                metadata={key: value for key, value in evaluation_metadata.items() if value},
                data_type=_langfuse_data_type(value),
            )
        )

    return evaluations


def build_experiment_output(
    backend_response: dict[str, Any] | None,
    *,
    case: dict[str, Any],
    experiment_name: str,
    backend_url: str,
    git_commit: str | None = None,
    evaluation_timestamp: str | None = None,
) -> dict[str, Any]:
    safe_response = dict(backend_response or {})
    safe_response["evaluation_metadata"] = {
        "dataset_item_id": case.get("item_id"),
        "dataset_name": case.get("dataset_name"),
        "experiment_name": experiment_name,
        "backend_url": backend_url,
        "app_version": safe_response.get("app_version"),
        "graph_version": safe_response.get("graph_version"),
        "git_commit": git_commit,
        "route": safe_response.get("route"),
        "topic": safe_response.get("topic"),
        "case_type": case.get("case_type"),
        "expected_behavior": case.get("expected_behavior"),
        "evaluation_timestamp": evaluation_timestamp
        or datetime.now(timezone.utc).isoformat(),
    }
    return safe_response


def _extract_question(input_value: Any, merged: dict[str, Any]) -> str:
    if isinstance(input_value, str):
        return input_value.strip()

    for key in ("question", "prompt", "query", "input", "user_question"):
        value = merged.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested_value = _extract_question(value, value)
            if nested_value:
                return nested_value

    return _safe_text(input_value)


def _expects_rag(case: dict[str, Any]) -> bool:
    expected_route = _normalize_route(case.get("expected_route"))
    expected_behavior = _slugify(case.get("expected_behavior"))
    return expected_route == "loans_rag" or "rag" in expected_behavior


def _expects_tool(case: dict[str, Any]) -> bool:
    expected_route = _normalize_route(case.get("expected_route"))
    expected_behavior = _slugify(case.get("expected_behavior"))
    return expected_route in TOOL_EXPECTED_ROUTES or "tool" in expected_behavior


def _quality_flag(
    result: dict[str, Any],
    quality_metadata: dict[str, Any],
    key: str,
) -> bool | None:
    direct_value = _bool_or_none(result.get(key))
    if direct_value is not None:
        return direct_value

    if quality_metadata.get("quality_eval_enabled"):
        return _bool_or_none(quality_metadata.get(key))

    return None


def _score_comment(
    name: str,
    value: Any,
    *,
    case: dict[str, Any],
    output: dict[str, Any] | None,
) -> str | None:
    safe_output = output if isinstance(output, dict) else {}
    if name == "route_match":
        return (
            f"expected={case.get('expected_route')} actual={safe_output.get('route')}"
        )
    if name == "topic_match":
        return (
            f"expected={case.get('expected_topic')} actual={safe_output.get('topic')}"
        )
    if name == "must_include_coverage":
        return f"coverage={value} over {len(case.get('must_include') or [])} terms"
    if name == "forbidden_terms_avoided":
        return "no forbidden terms detected" if value else "forbidden terms detected"
    if name == "used_rag_expected":
        return f"used_rag={safe_output.get('used_rag')}"
    if name == "used_tool_expected":
        return f"used_tool={safe_output.get('used_tool')}"
    if name == "fallback_avoided":
        return f"fallback={safe_output.get('fallback')}"
    if name == "answer_non_empty":
        return "answer returned" if value else "empty answer"
    if name == "whatsapp_format_basic_ok":
        return "basic WhatsApp formatting looks valid" if value else "format issues detected"
    return None


def _langfuse_data_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, str):
        return "CATEGORICAL"
    return None


def _normalize_route(value: Any) -> str:
    normalized = _slugify(value)
    return ROUTE_ALIASES.get(normalized, normalized)


def _normalize_topic(value: Any) -> str:
    normalized = _slugify(value)
    return TOPIC_ALIASES.get(normalized, normalized)


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", _safe_text(value))
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip().lower()


def _slugify(value: Any) -> str:
    normalized = _normalize_text(value)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = _safe_text(value)
    return text or None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False

    return bool(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            cleaned = _safe_text(item)
            if cleaned:
                result.append(cleaned)
        return result

    cleaned = _safe_text(value)
    return [cleaned] if cleaned else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
