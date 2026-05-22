import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

from observability.langfuse_config import safe_score, safe_update_observation
from observability.metrics import (
    build_node_end_metadata,
    build_node_scores,
    build_node_start_metadata,
    duration_ms,
    is_langfuse_node_observability_enabled,
    is_langfuse_node_score_enabled,
    now_ms,
)


_NODE_OBSERVABILITY_ACTIVE: ContextVar[bool] = ContextVar(
    "node_observability_active",
    default=False,
)
_NODE_OBSERVABILITY_CLIENT: ContextVar[Any] = ContextVar(
    "node_observability_client",
    default=None,
)
_UNSET = object()
NODE_OBSERVATION_TYPES = {
    "router": "agent",
    "contextualizer": "chain",
    "query_rewriter": "chain",
    "retriever": "retriever",
    "answer": "chain",
    "chitchat_answer": "chain",
    "bcra_agent": "agent",
    "benefits": "tool",
    "branch_locator": "tool",
    "credit_card_statement": "tool",
    "guardrail": "guardrail",
    "save_memory": "chain",
}


@contextmanager
def node_observability_context(langfuse_client: Any):
    is_active = bool(langfuse_client) and is_langfuse_node_observability_enabled()
    active_token = _NODE_OBSERVABILITY_ACTIVE.set(is_active)
    client_token = _NODE_OBSERVABILITY_CLIENT.set(langfuse_client if is_active else None)

    try:
        yield
    finally:
        _NODE_OBSERVABILITY_CLIENT.reset(client_token)
        _NODE_OBSERVABILITY_ACTIVE.reset(active_token)


def observe_node(node_name: str, node_fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapped_node(state: Any):
        if not _NODE_OBSERVABILITY_ACTIVE.get():
            return node_fn(state)

        langfuse_client = _NODE_OBSERVABILITY_CLIENT.get()
        if langfuse_client is None:
            return node_fn(state)

        start_ms = now_ms()
        app_version = os.getenv("APP_VERSION", "unknown").strip() or None
        start_metadata = build_node_start_metadata(node_name, state=state)
        result: Any = _UNSET
        node_error: Exception | None = None

        try:
            with langfuse_client.start_as_current_observation(
                name=node_name,
                as_type=NODE_OBSERVATION_TYPES.get(node_name, "span"),
                metadata=start_metadata,
                version=app_version,
            ) as observation:
                try:
                    result = node_fn(state)
                except Exception as exc:
                    node_error = exc
                    latency_ms = duration_ms(start_ms)
                    effective_state = _merge_states(state, None)
                    end_metadata = build_node_end_metadata(
                        node_name,
                        state=effective_state,
                        latency_ms=latency_ms,
                        error=exc,
                    )
                    safe_update_observation(
                        observation,
                        metadata={**start_metadata, **end_metadata},
                        level="ERROR",
                        status_message=type(exc).__name__,
                        version=app_version,
                    )
                    _send_node_scores(
                        observation,
                        node_name,
                        effective_state,
                        latency_ms=latency_ms,
                        error=exc,
                    )
                    _log_node_observability(node_name, latency_ms, "error", end_metadata)
                    raise

                latency_ms = duration_ms(start_ms)
                effective_state = _merge_states(state, result)
                node_result_error = _resulting_node_error(state, effective_state)
                end_metadata = build_node_end_metadata(
                    node_name,
                    state=effective_state,
                    latency_ms=latency_ms,
                    error=node_result_error,
                )
                safe_update_observation(
                    observation,
                    metadata={**start_metadata, **end_metadata},
                    version=app_version,
                )
                _send_node_scores(
                    observation,
                    node_name,
                    effective_state,
                    latency_ms=latency_ms,
                    error=node_result_error,
                )
                _log_node_observability(
                    node_name,
                    latency_ms,
                    str(end_metadata.get("node_status") or "success"),
                    end_metadata,
                )

            if result is not _UNSET:
                return result
        except Exception:
            if node_error is not None:
                raise

            if result is not _UNSET:
                return result

            return node_fn(state)

    return wrapped_node


def _merge_states(original_state: Any, node_result: Any) -> dict[str, Any]:
    base_state = dict(original_state) if isinstance(original_state, dict) else {}

    if isinstance(node_result, dict):
        merged_state = dict(base_state)
        merged_state.update(node_result)
        return merged_state

    return base_state


def _resulting_node_error(
    original_state: Any,
    effective_state: dict[str, Any],
) -> str | None:
    original_error = _normalized_error(original_state)
    current_error = _normalized_error(effective_state)

    if current_error and current_error != original_error:
        return current_error

    return None


def _normalized_error(state: Any) -> str:
    if not isinstance(state, dict):
        return ""

    return str(state.get("error") or "").strip()


def _send_node_scores(
    observation: Any,
    node_name: str,
    state: dict[str, Any],
    *,
    latency_ms: int,
    error: Any = None,
) -> None:
    if not is_langfuse_node_score_enabled():
        return

    scores = build_node_scores(
        node_name,
        state=state,
        latency_ms=latency_ms,
        error=error,
    )

    for score_name, score_value in scores.items():
        score_type = _infer_score_type(score_name)
        safe_score(
            observation,
            getattr(observation, "trace_id", None),
            score_name,
            score_value,
            data_type=score_type,
            comment=_build_score_comment(node_name, score_name),
            scope="observation",
        )


def _infer_score_type(score_name: str) -> str:
    if score_name.endswith("_latency_ms"):
        return "NUMERIC"

    numeric_suffixes = (
        "_docs_count",
        "_results_count",
        "_context_chars",
        "_length",
    )
    if score_name.endswith(numeric_suffixes):
        return "NUMERIC"

    return "BOOLEAN"


def _build_score_comment(node_name: str, score_name: str) -> str:
    if score_name.endswith("_latency_ms"):
        return f"Latencia del nodo {node_name} en milisegundos."
    if score_name.endswith("_success"):
        return f"Indica si el nodo {node_name} termino correctamente."
    if score_name.endswith("_error"):
        return f"Indica si el nodo {node_name} termino con error."
    if score_name == "retriever_docs_count":
        return "Cantidad de documentos relevantes recuperados por el nodo retriever."
    if score_name == "retriever_has_context":
        return "Indica si el retriever construyo contexto."
    if score_name == "retriever_empty_retrieval":
        return "Indica si el retriever termino sin documentos relevantes."
    if score_name == "retriever_context_chars":
        return "Cantidad de caracteres del contexto recuperado."
    if score_name == "answer_length":
        return "Cantidad de caracteres de la respuesta generada por el nodo answer."
    if score_name == "answer_fallback_used":
        return "Indica si el nodo answer devolvio fallback."
    if score_name == "answer_used_rag":
        return "Indica si el nodo answer uso contexto RAG."
    if score_name == "benefits_used_tool":
        return "Indica si el nodo benefits uso tool o API."
    if score_name == "benefits_results_count":
        return "Cantidad de resultados encontrados por el nodo benefits."
    if score_name == "benefits_needs_clarification":
        return "Indica si el nodo benefits pidio aclaracion."
    if score_name == "guardrail_blocked":
        return "Indica si el nodo guardrail bloqueo o modifico la salida."
    if score_name == "guardrail_fallback_used":
        return "Indica si guardrail termino en fallback."
    if score_name == "router_fallback_used":
        return "Indica si el router eligio la ruta fallback."
    return f"Score del nodo {node_name}: {score_name}."


def _log_node_observability(
    node_name: str,
    latency_ms: int,
    status: str,
    metadata: dict[str, Any],
) -> None:
    details = [
        f"node={node_name}",
        f"latency_ms={latency_ms}",
        f"status={status}",
    ]

    route = str(metadata.get("route") or "").strip()
    if route:
        details.append(f"route={route}")

    documents_count = metadata.get("documents_count")
    if isinstance(documents_count, int) and documents_count > 0:
        details.append(f"docs={documents_count}")

    results_count = metadata.get("results_count")
    if isinstance(results_count, int) and results_count > 0:
        details.append(f"results={results_count}")

    print(f"[observability] {' '.join(details)}")
