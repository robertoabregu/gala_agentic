import os
import re
import time
import unicodedata
from typing import Any

from core.constants import FALLBACK_ANSWER


RAG_ROUTES = {"rag", "loans_rag"}
TOOL_ROUTES = {
    "bcra_credit_status",
    "benefits",
    "branch_locator",
    "credit_card_statement",
}
TOPIC_BY_ROUTE = {
    "loans_rag": "prestamos",
    "rag": "prestamos",
    "bcra_credit_status": "situacion_crediticia_bcra",
    "branch_locator": "sucursales_cercanas",
    "benefits": "beneficios",
    "credit_card_statement": "resumen_tarjeta",
    "chitchat": "conversacion",
    "fallback": "fallback",
}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
ALLOWED_ROUTE_CATEGORIES = {
    "chitchat",
    "benefits",
    "rag",
    "answer",
    "bcra_agent",
    "branch_locator",
    "credit_card_statement",
    "fallback",
    "unknown",
}
ROUTE_CATEGORY_ALIASES = {
    "loans_rag": "rag",
    "rag": "rag",
    "bcra_credit_status": "bcra_agent",
    "bcra_agent": "bcra_agent",
    "branch_locator": "branch_locator",
    "benefits": "benefits",
    "credit_card_statement": "credit_card_statement",
    "chitchat": "chitchat",
    "answer": "answer",
    "fallback": "fallback",
    "sensitive": "fallback",
}
ALLOWED_TOPIC_CATEGORIES = {
    "saludo",
    "beneficios",
    "prestamos",
    "situacion_crediticia",
    "sucursales",
    "resumen_tarjeta",
    "conversacion",
    "fallback",
    "unknown",
}
TOPIC_CATEGORY_ALIASES = {
    "saludo": "saludo",
    "beneficios": "beneficios",
    "prestamos": "prestamos",
    "situacion_crediticia_bcra": "situacion_crediticia",
    "situacion_crediticia": "situacion_crediticia",
    "sucursales_cercanas": "sucursales",
    "sucursales": "sucursales",
    "resumen_tarjeta": "resumen_tarjeta",
    "conversacion": "conversacion",
    "capacidades": "conversacion",
    "fallback": "fallback",
}
ALLOWED_CHANNEL_CATEGORIES = {"whatsapp", "api", "local", "unknown"}
ALLOWED_ENVIRONMENT_CATEGORIES = {"dev", "staging", "prod", "default", "unknown"}
HIGH_HALLUCINATION_RISK_THRESHOLD = 0.65
CATEGORICAL_SCORE_DEFINITIONS = {
    "conversation_route": {
        "data_type": "CATEGORICAL",
        "comment": "Ruta conversacional normalizada para dashboards de Langfuse.",
    },
    "conversation_topic": {
        "data_type": "CATEGORICAL",
        "comment": "Tema conversacional normalizado para dashboards de Langfuse.",
    },
    "conversation_channel": {
        "data_type": "CATEGORICAL",
        "comment": "Canal de entrada normalizado de la conversacion.",
    },
    "conversation_environment": {
        "data_type": "CATEGORICAL",
        "comment": "Ambiente de ejecucion normalizado de la conversacion.",
    },
    "conversation_rag_status": {
        "data_type": "CATEGORICAL",
        "comment": "Estado del uso de RAG durante la conversacion.",
    },
    "conversation_tool_status": {
        "data_type": "CATEGORICAL",
        "comment": "Estado del uso de tools o integraciones durante la conversacion.",
    },
    "conversation_quality_status": {
        "data_type": "CATEGORICAL",
        "comment": "Estado sintetico de calidad para la conversacion.",
    },
    "conversation_outcome": {
        "data_type": "CATEGORICAL",
        "comment": "Resultado final normalizado combinando ruta y outcome.",
    },
}


def safe_len(value: Any) -> int:
    if value is None:
        return 0

    try:
        return len(value)
    except TypeError:
        return 0


def bool_score(value: Any) -> int:
    return 1 if bool(value) else 0


def now_ms() -> int:
    return int(time.time() * 1000)


def duration_ms(start_ms: int | float | None) -> int:
    try:
        return max(0, now_ms() - int(start_ms or 0))
    except (TypeError, ValueError):
        return 0


def is_langfuse_observability_enabled() -> bool:
    raw_value = os.getenv("LANGFUSE_OBSERVABILITY_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_langfuse_node_observability_enabled() -> bool:
    if not is_langfuse_observability_enabled():
        return False

    raw_value = os.getenv("LANGFUSE_NODE_OBSERVABILITY_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_langfuse_node_score_enabled() -> bool:
    if not is_langfuse_node_observability_enabled():
        return False

    raw_value = os.getenv("LANGFUSE_NODE_SCORE_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_langfuse_categorical_scores_enabled() -> bool:
    if not is_langfuse_observability_enabled():
        return False

    raw_value = os.getenv("LANGFUSE_CATEGORICAL_SCORES_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_langfuse_session_metrics_enabled() -> bool:
    if not is_langfuse_observability_enabled():
        return False

    raw_value = os.getenv("LANGFUSE_SESSION_METRICS_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def is_langfuse_session_categorical_enabled() -> bool:
    if not is_langfuse_observability_enabled():
        return False

    raw_value = os.getenv("LANGFUSE_SESSION_CATEGORICAL_ENABLED")
    if raw_value is None:
        return True

    return raw_value.strip().lower() not in FALSE_ENV_VALUES


def build_initial_trace_metadata(
    *,
    question: str | None = None,
    user_location: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
    channel: str | None = None,
    entrypoint: str | None = None,
    environment: str | None = None,
    graph_version: str | None = None,
    app_version: str | None = None,
    langfuse_tags: list[str] | None = None,
    observation_name: str | None = None,
) -> dict[str, Any]:
    inferred_channel = _infer_channel(
        channel=channel,
        langfuse_tags=langfuse_tags,
        observation_name=observation_name,
    )
    inferred_entrypoint = _infer_entrypoint(
        entrypoint=entrypoint,
        langfuse_tags=langfuse_tags,
        observation_name=observation_name,
    )

    return {
        "channel": inferred_channel,
        "environment": _clean_string(environment) or os.getenv("APP_ENV", "dev"),
        "entrypoint": inferred_entrypoint,
        "graph_version": _clean_string(graph_version) or os.getenv(
            "GALA_GRAPH_VERSION",
            "gala_graph_v1",
        ),
        "app_version": _clean_string(app_version) or os.getenv("APP_VERSION", "unknown"),
        "has_location": _has_location(user_location),
        "has_media": bool(media),
        "media_type": _infer_media_type(media),
        "input_chars": safe_len(question or ""),
    }


def build_final_trace_metadata(
    state: dict[str, Any] | None,
    total_latency_ms: int | None = None,
    error: Any = None,
) -> dict[str, Any]:
    safe_state = _safe_state(state)
    final_route = _final_route(safe_state)
    final_topic = _final_topic(safe_state, final_route)
    used_tool = _used_tool(safe_state, final_route)
    tool_name = _tool_name(safe_state, final_route) if used_tool else ""

    return {
        "final_route": final_route,
        "final_topic": final_topic,
        "used_rag": _used_rag(safe_state, final_route),
        "used_tool": used_tool,
        "tool_name": tool_name,
        "fallback": _fallback_used(safe_state, final_route),
        "guardrail_blocked": _guardrail_blocked(safe_state, final_route),
        "needs_clarification": bool(safe_state.get("needs_clarification")),
        "answer_chars": safe_len(_final_answer(safe_state)),
        "documents_count": safe_len(safe_state.get("documents") or []),
        "error": _safe_error_value(error if error is not None else safe_state.get("error")),
        "total_latency_ms": max(0, int(total_latency_ms or 0)),
    }


def build_basic_scores(
    state: dict[str, Any] | None,
    total_latency_ms: int | None = None,
) -> dict[str, int]:
    safe_state = _safe_state(state)
    final_route = _final_route(safe_state)

    return {
        "total_latency_ms": max(0, int(total_latency_ms or 0)),
        "answer_length": safe_len(_final_answer(safe_state)),
        "retrieval_docs_count": safe_len(safe_state.get("documents") or []),
        "has_context": bool_score(safe_state.get("context")),
        "fallback_used": bool_score(_fallback_used(safe_state, final_route)),
        "guardrail_blocked": bool_score(_guardrail_blocked(safe_state, final_route)),
        "used_rag": bool_score(_used_rag(safe_state, final_route)),
        "used_tool": bool_score(_used_tool(safe_state, final_route)),
        "needs_clarification": bool_score(safe_state.get("needs_clarification")),
    }


def normalize_category_value(value: Any, default: str = "unknown") -> str:
    fallback = str(default or "unknown").strip() or "unknown"

    if isinstance(value, bool):
        return "true" if value else "false"

    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or fallback


def build_categorical_scores(
    state: dict[str, Any],
    initial_metadata: dict[str, Any] | None = None,
    final_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    safe_state = _safe_state(state)
    safe_initial_metadata = initial_metadata or {}
    safe_final_metadata = final_metadata or {}

    return {
        "conversation_route": _conversation_route_category(
            safe_state,
            safe_final_metadata,
        ),
        "conversation_topic": _conversation_topic_category(
            safe_state,
            safe_final_metadata,
        ),
        "conversation_channel": _conversation_channel_category(
            safe_initial_metadata,
            safe_final_metadata,
        ),
        "conversation_environment": _conversation_environment_category(
            safe_initial_metadata,
            safe_final_metadata,
        ),
        "conversation_rag_status": _conversation_rag_status(
            safe_state,
            safe_final_metadata,
        ),
        "conversation_tool_status": _conversation_tool_status(
            safe_state,
            safe_final_metadata,
        ),
        "conversation_quality_status": build_quality_status(
            safe_state,
            final_metadata=safe_final_metadata,
        ),
        "conversation_outcome": build_conversation_outcome(
            safe_state,
            final_metadata=safe_final_metadata,
        ),
    }


def build_conversation_outcome(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    safe_state = _safe_state(state)
    safe_final_metadata = final_metadata or {}
    route_category = _conversation_route_category(safe_state, safe_final_metadata)
    used_rag = _resolve_used_rag(safe_state, safe_final_metadata)
    has_context = _resolve_has_context(safe_state, safe_final_metadata)
    documents_count = _resolve_documents_count(safe_state, safe_final_metadata)
    fallback_used = _resolve_fallback_used(safe_state, safe_final_metadata)
    guardrail_blocked = _resolve_guardrail_blocked(safe_state, safe_final_metadata)
    needs_clarification = _resolve_needs_clarification(safe_state, safe_final_metadata)
    missing_fields = _resolve_missing_fields(safe_state, safe_final_metadata)

    if guardrail_blocked:
        return "guardrail_blocked"

    if fallback_used:
        if route_category and route_category not in {"fallback", "unknown"}:
            return f"{route_category}_fallback"
        return "fallback"

    if route_category in {"rag", "answer"} and used_rag and not (has_context or documents_count > 0):
        return "rag_no_context"

    if needs_clarification:
        if route_category == "credit_card_statement" and "pdf" in missing_fields:
            return "credit_card_statement_needs_file"
        if route_category in {
            "benefits",
            "bcra_agent",
            "branch_locator",
            "credit_card_statement",
        }:
            return f"{route_category}_needs_clarification"

    if route_category in ALLOWED_ROUTE_CATEGORIES and route_category not in {"fallback", "unknown"}:
        return f"{route_category}_success"

    if route_category == "fallback":
        return "fallback"

    return "unknown"


def build_quality_status(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    safe_state = _safe_state(state)
    safe_final_metadata = final_metadata or {}
    guardrail_blocked = _resolve_guardrail_blocked(safe_state, safe_final_metadata)
    fallback_used = _resolve_fallback_used(safe_state, safe_final_metadata)
    needs_human_review = _resolve_bool_signal(
        safe_state,
        safe_final_metadata,
        "needs_human_review",
    )
    dataset_candidate = _resolve_bool_signal(
        safe_state,
        safe_final_metadata,
        "dataset_candidate",
    )
    likely_low_value_answer = _resolve_bool_signal(
        safe_state,
        safe_final_metadata,
        "likely_low_value_answer",
    )
    hallucination_risk = _resolve_float_signal(
        safe_state,
        safe_final_metadata,
        "hallucination_risk",
    )

    if guardrail_blocked:
        return "risky"

    if hallucination_risk is not None and hallucination_risk >= HIGH_HALLUCINATION_RISK_THRESHOLD:
        return "risky"

    if fallback_used:
        return "fallback"

    if needs_human_review or likely_low_value_answer:
        return "needs_review"

    if dataset_candidate:
        return "dataset_candidate"

    return "healthy"


def build_node_start_metadata(
    node_name: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_state = _safe_state(state)
    route = _current_route(safe_state)
    topic = _current_topic(safe_state, route)
    used_tool = _used_tool(safe_state, route)
    metadata = {
        "node_name": node_name,
        "node_phase": "start",
        "route": route,
        "topic": topic,
        "used_rag": _used_rag(safe_state, route),
        "used_tool": used_tool,
        "tool_name": _tool_name(safe_state, route) if used_tool else "",
        "documents_count": safe_len(safe_state.get("documents") or []),
        "fallback": _fallback_used(safe_state, route),
        "guardrail_blocked": _guardrail_blocked(safe_state, route),
        "needs_clarification": bool(safe_state.get("needs_clarification")),
    }
    metadata.update(_build_node_specific_metadata(node_name, safe_state))
    return metadata


def build_node_end_metadata(
    node_name: str,
    state: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    error: Any = None,
) -> dict[str, Any]:
    safe_state = _safe_state(state)
    route = _current_route(safe_state)
    topic = _current_topic(safe_state, route)
    used_tool = _used_tool(safe_state, route)
    metadata = {
        "node_name": node_name,
        "node_phase": "end",
        "node_status": "error" if error is not None else "success",
        "node_latency_ms": max(0, int(latency_ms or 0)),
        "route": route,
        "topic": topic,
        "used_rag": _used_rag(safe_state, route),
        "used_tool": used_tool,
        "tool_name": _tool_name(safe_state, route) if used_tool else "",
        "documents_count": safe_len(safe_state.get("documents") or []),
        "fallback": _fallback_used(safe_state, route),
        "guardrail_blocked": _guardrail_blocked(safe_state, route),
        "needs_clarification": bool(safe_state.get("needs_clarification")),
        "error": _safe_error_value(error if error is not None else safe_state.get("error")),
    }
    metadata.update(_build_node_specific_metadata(node_name, safe_state))
    return metadata


def build_node_scores(
    node_name: str,
    state: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    error: Any = None,
) -> dict[str, int]:
    safe_state = _safe_state(state)
    route = _current_route(safe_state)
    scores = {
        f"{node_name}_latency_ms": max(0, int(latency_ms or 0)),
        f"{node_name}_success": 0 if error is not None else 1,
        f"{node_name}_error": 1 if error is not None else 0,
    }
    scores.update(_build_node_specific_scores(node_name, safe_state, route))
    return scores


def _conversation_route_category(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    raw_route = _clean_string((final_metadata or {}).get("final_route")) or _final_route(state)
    return _normalize_route_category(raw_route)


def _conversation_topic_category(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    raw_route = _clean_string((final_metadata or {}).get("final_route")) or _final_route(state)
    raw_topic = _clean_string((final_metadata or {}).get("final_topic")) or _final_topic(state, raw_route)
    return _normalize_topic_category(raw_topic)


def _conversation_channel_category(
    initial_metadata: dict[str, Any] | None = None,
    final_metadata: dict[str, Any] | None = None,
) -> str:
    raw_channel = _clean_string((final_metadata or {}).get("channel")) or _clean_string(
        (initial_metadata or {}).get("channel")
    )
    normalized_channel = normalize_category_value(raw_channel)
    if normalized_channel in ALLOWED_CHANNEL_CATEGORIES:
        return normalized_channel
    return "unknown"


def _conversation_environment_category(
    initial_metadata: dict[str, Any] | None = None,
    final_metadata: dict[str, Any] | None = None,
) -> str:
    raw_environment = _clean_string((final_metadata or {}).get("environment")) or _clean_string(
        (initial_metadata or {}).get("environment")
    )
    if not raw_environment:
        raw_environment = "default"

    normalized_environment = normalize_category_value(raw_environment, default="default")
    if normalized_environment in ALLOWED_ENVIRONMENT_CATEGORIES:
        return normalized_environment
    return "unknown"


def _conversation_rag_status(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    used_rag = _resolve_used_rag(state, final_metadata)
    has_context = _resolve_has_context(state, final_metadata)
    documents_count = _resolve_documents_count(state, final_metadata)

    if not used_rag:
        return "rag_not_used"
    if has_context or documents_count > 0:
        return "rag_with_context"
    return "rag_without_context"


def _conversation_tool_status(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None = None,
) -> str:
    route = _clean_string((final_metadata or {}).get("final_route")) or _final_route(state)
    used_tool = _resolve_used_tool(state, final_metadata)
    needs_clarification = _resolve_needs_clarification(state, final_metadata)

    if needs_clarification and route in TOOL_ROUTES:
        return "tool_needs_clarification"
    if used_tool:
        return "tool_used"
    return "tool_not_used"


def _safe_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    return {}


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _normalized_tags(tags: list[str] | None) -> set[str]:
    return {
        str(tag).strip().lower()
        for tag in (tags or [])
        if str(tag).strip()
    }


def _infer_channel(
    *,
    channel: str | None,
    langfuse_tags: list[str] | None,
    observation_name: str | None,
) -> str:
    if _clean_string(channel):
        return _clean_string(channel)

    tags = _normalized_tags(langfuse_tags)
    observation = _clean_string(observation_name).lower()

    if "whatsapp" in tags or "whatsapp" in observation:
        return "whatsapp"
    if "api" in tags or "api" in observation:
        return "api"
    if "local" in tags or "local-prototype" in tags or "local" in observation:
        return "local"

    return "unknown"


def _infer_entrypoint(
    *,
    entrypoint: str | None,
    langfuse_tags: list[str] | None,
    observation_name: str | None,
) -> str:
    if _clean_string(entrypoint):
        return _clean_string(entrypoint)

    tags = _normalized_tags(langfuse_tags)
    observation = _clean_string(observation_name).lower()

    if "webhook" in tags or "webhook" in observation or "whatsapp" in observation:
        return "webhook"
    if "api" in tags or "api" in observation:
        return "api"
    if "local" in tags or "local-prototype" in tags or "local" in observation:
        return "local"

    return "unknown"


def _has_location(user_location: dict[str, Any] | None) -> bool:
    if not isinstance(user_location, dict) or not user_location:
        return False

    latitude = _clean_string(user_location.get("latitude"))
    longitude = _clean_string(user_location.get("longitude"))
    return bool((latitude and longitude) or user_location)


def _infer_media_type(media: dict[str, Any] | None) -> str:
    if not isinstance(media, dict) or not media:
        return "none"

    content_type = _clean_string(media.get("content_type")).lower()
    filename = _clean_string(media.get("filename")).lower()

    if "pdf" in content_type or filename.endswith(".pdf"):
        return "pdf"
    if content_type.startswith("image/") or filename.endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"),
    ):
        return "image"

    return "other"


def _final_route(state: dict[str, Any]) -> str:
    memory = state.get("memory") or {}
    route = _clean_string(state.get("route"))
    if route:
        return route

    if isinstance(memory, dict):
        return _clean_string(memory.get("last_route"))

    return ""


def _final_topic(state: dict[str, Any], route: str) -> str:
    topic = _clean_string(state.get("topic"))
    if topic:
        return topic

    memory = state.get("memory") or {}
    if isinstance(memory, dict):
        memory_topic = _clean_string(memory.get("last_topic"))
        if memory_topic:
            return memory_topic

    return TOPIC_BY_ROUTE.get(route, route or "")


def _current_route(state: dict[str, Any]) -> str:
    return _clean_string(state.get("route"))


def _current_topic(state: dict[str, Any], route: str) -> str:
    topic = _clean_string(state.get("topic"))
    if topic:
        return topic

    return TOPIC_BY_ROUTE.get(route, "")


def _tool_name(state: dict[str, Any], route: str) -> str:
    tool_name = _clean_string(state.get("tool_name"))
    return tool_name or route


def _used_rag(state: dict[str, Any], route: str) -> bool:
    if route in RAG_ROUTES:
        return True

    return bool(
        _clean_string(state.get("search_query"))
        or state.get("documents")
        or _clean_string(state.get("context"))
    )


def _used_tool(state: dict[str, Any], route: str) -> bool:
    if route in TOOL_ROUTES:
        return True

    return bool(
        _clean_string(state.get("tool_name"))
        or state.get("tool_input")
        or state.get("tool_output")
    )


def _guardrail_blocked(state: dict[str, Any], route: str) -> bool:
    answer = _clean_string(state.get("answer"))
    final_answer = _clean_string(state.get("final_answer"))

    if route == "sensitive":
        return True

    return bool(final_answer and final_answer != answer)


def _fallback_used(state: dict[str, Any], route: str) -> bool:
    return route == "fallback" or _final_answer(state) == FALLBACK_ANSWER


def _final_answer(state: dict[str, Any]) -> str:
    return _clean_string(state.get("final_answer")) or _clean_string(state.get("answer"))


def _safe_error_value(error: Any) -> str | None:
    if error is None:
        return None

    if isinstance(error, Exception):
        error_text = f"{type(error).__name__}: {str(error)}".strip(": ")
    else:
        error_text = _clean_string(error)

    if not error_text:
        return None

    return error_text[:200]


def _tool_output(state: dict[str, Any]) -> dict[str, Any]:
    tool_output = state.get("tool_output")
    if isinstance(tool_output, dict):
        return tool_output
    return {}


def _tool_output_int(tool_output: dict[str, Any], key: str) -> int:
    value = tool_output.get(key)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_route_category(route: Any) -> str:
    normalized_route = normalize_category_value(route)
    route_category = ROUTE_CATEGORY_ALIASES.get(normalized_route, normalized_route)
    if route_category in ALLOWED_ROUTE_CATEGORIES:
        return route_category
    return "unknown"


def _normalize_topic_category(topic: Any) -> str:
    normalized_topic = normalize_category_value(topic)
    topic_category = TOPIC_CATEGORY_ALIASES.get(normalized_topic, normalized_topic)
    if topic_category in ALLOWED_TOPIC_CATEGORIES:
        return topic_category
    return "unknown"


def _resolve_bool_signal(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
    key: str,
) -> bool:
    value = _resolve_signal(state, final_metadata, key)
    parsed = _as_optional_bool(value)
    return bool(parsed)


def _resolve_float_signal(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
    key: str,
) -> float | None:
    return _as_optional_float(_resolve_signal(state, final_metadata, key))


def _resolve_signal(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
    key: str,
) -> Any:
    if isinstance(final_metadata, dict) and key in final_metadata:
        return final_metadata.get(key)
    return state.get(key)


def _resolve_used_rag(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "used_rag"))
    if parsed is not None:
        return parsed
    return _used_rag(state, _final_route(state))


def _resolve_used_tool(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "used_tool"))
    if parsed is not None:
        return parsed
    return _used_tool(state, _final_route(state))


def _resolve_has_context(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    has_context = _as_optional_bool(_resolve_signal(state, final_metadata, "has_context"))
    if has_context is not None:
        return has_context
    return bool(_clean_string(state.get("context")))


def _resolve_documents_count(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> int:
    documents_count = _as_optional_int(_resolve_signal(state, final_metadata, "documents_count"))
    if documents_count is not None:
        return documents_count

    retrieval_docs_count = _as_optional_int(_resolve_signal(state, final_metadata, "retrieval_docs_count"))
    if retrieval_docs_count is not None:
        return retrieval_docs_count

    return safe_len(state.get("documents") or [])


def _resolve_fallback_used(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "fallback_used"))
    if parsed is not None:
        return parsed

    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "fallback"))
    if parsed is not None:
        return parsed

    return _fallback_used(state, _final_route(state))


def _resolve_guardrail_blocked(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "guardrail_blocked"))
    if parsed is not None:
        return parsed
    return _guardrail_blocked(state, _final_route(state))


def _resolve_needs_clarification(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> bool:
    parsed = _as_optional_bool(_resolve_signal(state, final_metadata, "needs_clarification"))
    if parsed is not None:
        return parsed
    return bool(state.get("needs_clarification"))


def _resolve_missing_fields(
    state: dict[str, Any],
    final_metadata: dict[str, Any] | None,
) -> list[str]:
    raw_missing_fields = _resolve_signal(state, final_metadata, "missing_fields")
    if isinstance(raw_missing_fields, list):
        return [normalize_category_value(field) for field in raw_missing_fields if str(field).strip()]
    return []


def _as_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_ENV_VALUES:
            return True
        if lowered in FALSE_ENV_VALUES:
            return False
    return None


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_missing_field(state: dict[str, Any], field_name: str) -> bool:
    missing_fields = state.get("missing_fields") or []
    return field_name in missing_fields


def _build_node_specific_metadata(
    node_name: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    if node_name == "router":
        metadata = {
            "selected_route": _current_route(state),
            "is_followup": bool(state.get("is_followup")),
        }
        route_source = _clean_string(state.get("route_source"))
        if route_source:
            metadata["route_source"] = route_source
        return metadata

    if node_name == "retriever":
        context = _clean_string(state.get("context"))
        documents_count = safe_len(state.get("documents") or [])
        return {
            "context_chars": safe_len(context),
            "has_context": bool(context),
            "empty_retrieval": documents_count == 0,
        }

    if node_name == "answer":
        answer_text = _clean_string(state.get("answer"))
        return {
            "answer_chars": safe_len(answer_text),
            "answer_fallback": answer_text == FALLBACK_ANSWER,
        }

    if node_name == "benefits":
        tool_output = _tool_output(state)
        return {
            "results_count": _tool_output_int(tool_output, "results_count"),
            "needs_location": _has_missing_field(state, "user_location"),
        }

    if node_name == "guardrail":
        final_answer = _clean_string(state.get("final_answer"))
        answer = _clean_string(state.get("answer"))
        return {
            "replaced_with_fallback": final_answer == FALLBACK_ANSWER and answer != FALLBACK_ANSWER,
            "replaced_answer": bool(final_answer and final_answer != answer),
        }

    if node_name in {"branch_locator", "credit_card_statement", "bcra_agent"}:
        tool_output = _tool_output(state)
        return {
            "results_count": _tool_output_int(tool_output, "results_count"),
        }

    return {}


def _build_node_specific_scores(
    node_name: str,
    state: dict[str, Any],
    route: str,
) -> dict[str, int]:
    if node_name == "router":
        return {
            "router_fallback_used": bool_score(route == "fallback"),
        }

    if node_name == "retriever":
        documents_count = safe_len(state.get("documents") or [])
        return {
            "retriever_docs_count": documents_count,
            "retriever_has_context": bool_score(state.get("context")),
            "retriever_empty_retrieval": bool_score(documents_count == 0),
            "retriever_context_chars": safe_len(_clean_string(state.get("context"))),
        }

    if node_name == "answer":
        answer_text = _clean_string(state.get("answer"))
        return {
            "answer_length": safe_len(answer_text),
            "answer_fallback_used": bool_score(answer_text == FALLBACK_ANSWER),
            "answer_used_rag": bool_score(_used_rag(state, route)),
        }

    if node_name == "benefits":
        tool_output = _tool_output(state)
        return {
            "benefits_used_tool": bool_score(_used_tool(state, route)),
            "benefits_results_count": _tool_output_int(tool_output, "results_count"),
            "benefits_needs_clarification": bool_score(state.get("needs_clarification")),
        }

    if node_name == "guardrail":
        return {
            "guardrail_blocked": bool_score(_guardrail_blocked(state, route)),
            "guardrail_fallback_used": bool_score(_final_answer(state) == FALLBACK_ANSWER),
        }

    return {}
