from __future__ import annotations

import threading
from collections import Counter
from copy import deepcopy
from typing import Any

from observability.metrics import build_categorical_scores, normalize_category_value, safe_len


SESSION_NUMERIC_SCORE_DEFINITIONS = {
    "session_turn_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos observados para la sesion.",
    },
    "session_total_latency_ms": {
        "data_type": "NUMERIC",
        "comment": "Latencia total acumulada de la sesion en milisegundos.",
    },
    "session_avg_latency_ms": {
        "data_type": "NUMERIC",
        "comment": "Latencia promedio por turno de la sesion en milisegundos.",
    },
    "session_max_latency_ms": {
        "data_type": "NUMERIC",
        "comment": "Mayor latencia observada en un turno de la sesion.",
    },
    "session_fallback_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos que terminaron en fallback.",
    },
    "session_rag_turns_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos con uso de RAG.",
    },
    "session_tool_turns_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos con uso de tools o integraciones.",
    },
    "session_needs_clarification_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos que pidieron aclaracion o datos faltantes.",
    },
    "session_guardrail_blocked_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos bloqueados o modificados por guardrail.",
    },
    "session_human_review_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos marcados para revision humana.",
    },
    "session_dataset_candidate_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos candidatos a dataset o regresion.",
    },
    "session_low_value_answer_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de turnos con senal de baja calidad o bajo valor.",
    },
    "session_total_answer_chars": {
        "data_type": "NUMERIC",
        "comment": "Cantidad total acumulada de caracteres respondidos por la sesion.",
    },
    "session_total_retrieval_docs": {
        "data_type": "NUMERIC",
        "comment": "Cantidad total acumulada de documentos recuperados en la sesion.",
    },
    "session_distinct_routes_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad de rutas distintas observadas en la sesion.",
    },
    "session_distinct_topics_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad de temas distintos observados en la sesion.",
    },
    "session_total_tokens": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de tokens totales observados en la sesion, si esta disponible.",
    },
    "session_input_tokens": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de input tokens observados en la sesion, si esta disponible.",
    },
    "session_output_tokens": {
        "data_type": "NUMERIC",
        "comment": "Cantidad acumulada de output tokens observados en la sesion, si esta disponible.",
    },
    "session_total_cost": {
        "data_type": "NUMERIC",
        "comment": "Costo total acumulado observado en la sesion, si esta disponible.",
    },
}
SESSION_CATEGORICAL_SCORE_DEFINITIONS = {
    "session_status": {
        "data_type": "CATEGORICAL",
        "comment": "Estado agregado de la sesion.",
    },
    "session_primary_route": {
        "data_type": "CATEGORICAL",
        "comment": "Ruta predominante observada en la sesion.",
    },
    "session_primary_topic": {
        "data_type": "CATEGORICAL",
        "comment": "Tema predominante observado en la sesion.",
    },
    "session_has_rag": {
        "data_type": "CATEGORICAL",
        "comment": "Indica si la sesion ya uso RAG en al menos un turno.",
    },
    "session_has_tool": {
        "data_type": "CATEGORICAL",
        "comment": "Indica si la sesion ya uso tools o integraciones en al menos un turno.",
    },
    "session_has_fallback": {
        "data_type": "CATEGORICAL",
        "comment": "Indica si la sesion ya tuvo fallback en al menos un turno.",
    },
    "session_complexity": {
        "data_type": "CATEGORICAL",
        "comment": "Complejidad agregada de la sesion segun cantidad de turnos y mezcla de rutas.",
    },
}


_SESSION_METRICS_STORE: dict[str, dict[str, Any]] = {}
_SESSION_METRICS_LOCK = threading.RLock()


def update_session_metrics(
    session_id: str,
    state: dict[str, Any],
    scores: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return {}

    safe_state = state if isinstance(state, dict) else {}
    safe_scores = scores if isinstance(scores, dict) else {}
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    combined_sources = {
        **safe_metadata,
        **safe_scores,
    }
    categorical_scores = build_categorical_scores(
        safe_state,
        final_metadata=combined_sources,
    )
    route = categorical_scores.get("conversation_route", "unknown")
    topic = categorical_scores.get("conversation_topic", "unknown")
    tool_name = _resolve_tool_name(safe_state, combined_sources, route)

    with _SESSION_METRICS_LOCK:
        session_metrics = _SESSION_METRICS_STORE.setdefault(
            normalized_session_id,
            _build_empty_session_metrics(normalized_session_id),
        )

        session_metrics["session_turn_count"] += 1

        latency_ms = _resolve_optional_number(
            combined_sources,
            "total_latency_ms",
        )
        if latency_ms is not None:
            session_metrics["session_total_latency_ms"] += latency_ms
            session_metrics["session_max_latency_ms"] = max(
                session_metrics["session_max_latency_ms"],
                latency_ms,
            )
            session_metrics["session_avg_latency_ms"] = round(
                session_metrics["session_total_latency_ms"] / session_metrics["session_turn_count"],
                2,
            )

        if _resolve_bool_flag(combined_sources, "fallback_used", "fallback"):
            session_metrics["session_fallback_count"] += 1

        if _resolve_bool_flag(combined_sources, "used_rag"):
            session_metrics["session_rag_turns_count"] += 1

        if _resolve_bool_flag(combined_sources, "used_tool"):
            session_metrics["session_tool_turns_count"] += 1

        if _resolve_bool_flag(combined_sources, "needs_clarification"):
            session_metrics["session_needs_clarification_count"] += 1

        if _resolve_bool_flag(combined_sources, "guardrail_blocked"):
            session_metrics["session_guardrail_blocked_count"] += 1

        if _resolve_bool_flag(combined_sources, "needs_human_review"):
            session_metrics["session_human_review_count"] += 1

        if _resolve_bool_flag(combined_sources, "dataset_candidate"):
            session_metrics["session_dataset_candidate_count"] += 1

        if _resolve_bool_flag(combined_sources, "likely_low_value_answer"):
            session_metrics["session_low_value_answer_count"] += 1

        answer_chars = _resolve_optional_number(
            combined_sources,
            "answer_chars",
            "answer_length",
        )
        if answer_chars is None:
            answer_chars = safe_len(
                str(safe_state.get("final_answer") or safe_state.get("answer") or "")
            )
        session_metrics["session_total_answer_chars"] += answer_chars

        retrieval_docs = _resolve_optional_number(
            combined_sources,
            "documents_count",
            "retrieval_docs_count",
        )
        if retrieval_docs is None:
            retrieval_docs = safe_len(safe_state.get("documents") or [])
        session_metrics["session_total_retrieval_docs"] += retrieval_docs

        if route != "unknown":
            session_metrics["routes_seen"].add(route)
            session_metrics["route_counts"][route] += 1

        if topic != "unknown":
            session_metrics["topics_seen"].add(topic)
            session_metrics["topic_counts"][topic] += 1

        if tool_name and tool_name != "unknown":
            session_metrics["tools_seen"].add(tool_name)
            session_metrics["tool_counts"][tool_name] += 1

        session_metrics["session_distinct_routes_count"] = len(session_metrics["routes_seen"])
        session_metrics["session_distinct_topics_count"] = len(session_metrics["topics_seen"])

        _accumulate_optional_number(
            session_metrics,
            combined_sources,
            target_key="session_total_tokens",
            source_keys=("session_total_tokens", "total_tokens", "usage_total_tokens"),
        )
        _accumulate_optional_number(
            session_metrics,
            combined_sources,
            target_key="session_input_tokens",
            source_keys=("session_input_tokens", "input_tokens", "prompt_tokens"),
        )
        _accumulate_optional_number(
            session_metrics,
            combined_sources,
            target_key="session_output_tokens",
            source_keys=("session_output_tokens", "output_tokens", "completion_tokens"),
        )
        _accumulate_optional_number(
            session_metrics,
            combined_sources,
            target_key="session_total_cost",
            source_keys=("session_total_cost", "total_cost", "cost", "cost_usd"),
        )

        return _snapshot_session_metrics(session_metrics)


def build_session_numeric_scores(session_metrics: dict[str, Any]) -> dict[str, int | float]:
    safe_metrics = _snapshot_session_metrics(session_metrics) if session_metrics else {}
    numeric_scores: dict[str, int | float] = {}

    for score_name in SESSION_NUMERIC_SCORE_DEFINITIONS:
        value = safe_metrics.get(score_name)
        if value is None:
            continue
        numeric_scores[score_name] = value

    return numeric_scores


def build_session_categorical_scores(session_metrics: dict[str, Any]) -> dict[str, str]:
    safe_metrics = _snapshot_session_metrics(session_metrics) if session_metrics else {}
    return {
        "session_status": build_session_status(safe_metrics),
        "session_primary_route": _primary_category(
            safe_metrics.get("route_counts") or {},
            distinct_count=int(safe_metrics.get("session_distinct_routes_count") or 0),
        ),
        "session_primary_topic": _primary_category(
            safe_metrics.get("topic_counts") or {},
            distinct_count=int(safe_metrics.get("session_distinct_topics_count") or 0),
        ),
        "session_has_rag": "true" if int(safe_metrics.get("session_rag_turns_count") or 0) > 0 else "false",
        "session_has_tool": "true" if int(safe_metrics.get("session_tool_turns_count") or 0) > 0 else "false",
        "session_has_fallback": "true" if int(safe_metrics.get("session_fallback_count") or 0) > 0 else "false",
        "session_complexity": _session_complexity(safe_metrics),
    }


def build_session_status(session_metrics: dict[str, Any]) -> str:
    safe_metrics = _snapshot_session_metrics(session_metrics) if session_metrics else {}
    turn_count = int(safe_metrics.get("session_turn_count") or 0)
    if turn_count <= 0:
        return "unknown"

    if int(safe_metrics.get("session_guardrail_blocked_count") or 0) > 0:
        return "risky"

    if (
        int(safe_metrics.get("session_human_review_count") or 0) > 0
        or int(safe_metrics.get("session_dataset_candidate_count") or 0) > 0
        or int(safe_metrics.get("session_low_value_answer_count") or 0) > 0
    ):
        return "needs_review"

    if int(safe_metrics.get("session_fallback_count") or 0) >= 2:
        return "fallback_heavy"

    if int(safe_metrics.get("session_distinct_routes_count") or 0) >= 3:
        return "mixed"

    return "healthy"


def reset_session_metrics(session_id: str) -> None:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return

    with _SESSION_METRICS_LOCK:
        _SESSION_METRICS_STORE.pop(normalized_session_id, None)


def get_session_metrics(session_id: str) -> dict[str, Any]:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return {}

    with _SESSION_METRICS_LOCK:
        session_metrics = _SESSION_METRICS_STORE.get(normalized_session_id)
        if session_metrics is None:
            return {}
        return _snapshot_session_metrics(session_metrics)


def build_session_debug_payload(session_metrics: dict[str, Any]) -> dict[str, Any]:
    safe_metrics = _snapshot_session_metrics(session_metrics) if session_metrics else {}
    categorical_scores = build_session_categorical_scores(safe_metrics)
    return {
        "turn_count": int(safe_metrics.get("session_turn_count") or 0),
        "routes_seen": list(safe_metrics.get("routes_seen") or []),
        "topics_seen": list(safe_metrics.get("topics_seen") or []),
        "fallback_count": int(safe_metrics.get("session_fallback_count") or 0),
        "rag_count": int(safe_metrics.get("session_rag_turns_count") or 0),
        "tool_count": int(safe_metrics.get("session_tool_turns_count") or 0),
        "status": categorical_scores.get("session_status", "unknown"),
    }


def _build_empty_session_metrics(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "session_turn_count": 0,
        "session_total_latency_ms": 0.0,
        "session_avg_latency_ms": 0.0,
        "session_max_latency_ms": 0.0,
        "session_fallback_count": 0,
        "session_rag_turns_count": 0,
        "session_tool_turns_count": 0,
        "session_needs_clarification_count": 0,
        "session_guardrail_blocked_count": 0,
        "session_human_review_count": 0,
        "session_dataset_candidate_count": 0,
        "session_low_value_answer_count": 0,
        "session_total_answer_chars": 0,
        "session_total_retrieval_docs": 0,
        "session_distinct_routes_count": 0,
        "session_distinct_topics_count": 0,
        "session_total_tokens": None,
        "session_input_tokens": None,
        "session_output_tokens": None,
        "session_total_cost": None,
        "routes_seen": set(),
        "topics_seen": set(),
        "tools_seen": set(),
        "route_counts": Counter(),
        "topic_counts": Counter(),
        "tool_counts": Counter(),
    }


def _snapshot_session_metrics(session_metrics: dict[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(session_metrics)
    snapshot["routes_seen"] = sorted(snapshot.get("routes_seen") or [])
    snapshot["topics_seen"] = sorted(snapshot.get("topics_seen") or [])
    snapshot["tools_seen"] = sorted(snapshot.get("tools_seen") or [])
    snapshot["route_counts"] = dict(snapshot.get("route_counts") or {})
    snapshot["topic_counts"] = dict(snapshot.get("topic_counts") or {})
    snapshot["tool_counts"] = dict(snapshot.get("tool_counts") or {})
    return snapshot


def _resolve_tool_name(
    state: dict[str, Any],
    combined_sources: dict[str, Any],
    route: str,
) -> str:
    if not _resolve_bool_flag(combined_sources, "used_tool"):
        return ""

    raw_tool_name = str(
        combined_sources.get("tool_name")
        or state.get("tool_name")
        or route
        or ""
    ).strip()
    return normalize_category_value(raw_tool_name)


def _resolve_bool_flag(combined_sources: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in combined_sources:
            continue

        value = combined_sources.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False

    return False


def _resolve_optional_number(
    combined_sources: dict[str, Any],
    *keys: str,
) -> int | float | None:
    for key in keys:
        if key not in combined_sources:
            continue
        value = _safe_number(combined_sources.get(key))
        if value is not None:
            return value
    return None


def _accumulate_optional_number(
    session_metrics: dict[str, Any],
    combined_sources: dict[str, Any],
    *,
    target_key: str,
    source_keys: tuple[str, ...],
) -> None:
    value = _resolve_optional_number(combined_sources, *source_keys)
    if value is None:
        return

    if session_metrics.get(target_key) is None:
        session_metrics[target_key] = 0.0

    session_metrics[target_key] += value


def _safe_number(value: Any) -> int | float | None:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
        bounded_numeric = max(0.0, numeric)
        if bounded_numeric.is_integer():
            return int(bounded_numeric)
        return bounded_numeric
    except (TypeError, ValueError):
        return None


def _primary_category(counts: dict[str, int], *, distinct_count: int) -> str:
    if not counts:
        return "unknown"

    most_common = Counter(counts).most_common()
    if not most_common:
        return "unknown"

    if distinct_count >= 3:
        return "mixed"

    top_value = most_common[0][1]
    tied_categories = [category for category, count in most_common if count == top_value]
    if len(tied_categories) > 1:
        return "mixed"

    return normalize_category_value(most_common[0][0])


def _session_complexity(session_metrics: dict[str, Any]) -> str:
    turn_count = int(session_metrics.get("session_turn_count") or 0)
    distinct_routes_count = int(session_metrics.get("session_distinct_routes_count") or 0)

    if turn_count <= 0:
        return "unknown"
    if distinct_routes_count >= 3:
        return "mixed"
    if turn_count == 1:
        return "single_turn"
    if 2 <= turn_count <= 3:
        return "short"
    if 4 <= turn_count <= 7:
        return "medium"
    return "long"
