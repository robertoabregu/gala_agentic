import re
import unicodedata
from datetime import datetime, timezone

from agents.state import AgentState
from core.constants import FALLBACK_ANSWER
from core.privacy import mask_sensitive_text
from memory.local_memory import clear_pending, save_memory, set_pending
from observability.logger import log_step


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


def _infer_topic(route: str) -> str:
    return TOPIC_BY_ROUTE.get(route, route or "")


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    lowered = without_accents.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _is_location_placeholder(text: str) -> bool:
    return _normalize_text(text) == "ubicacion compartida por whatsapp"


def _get_memory_question(state: AgentState, memory: dict) -> str:
    route = state.get("route", "")
    tool_input = state.get("tool_input") or {}
    resolved_query = str(tool_input.get("resolved_query") or "").strip()
    question = str(state.get("question") or "").strip()

    if route == "benefits":
        if resolved_query:
            return resolved_query

        if _is_location_placeholder(question):
            return str(memory.get("pending_query") or memory.get("last_user_question", "")).strip()

        return (
            question
            or state.get("original_question")
            or memory.get("last_user_question", "")
        )

    return (
        state.get("standalone_question")
        or state.get("question")
        or memory.get("last_user_question", "")
    )


def save_memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "demo-local")
    memory = state.get("memory") or {}
    route = state.get("route", "")
    missing_fields = state.get("missing_fields", [])
    final_answer = state.get("final_answer", "")
    tool_input = state.get("tool_input") or {}
    user_location = state.get("user_location") or {}
    pending_query = ""

    if route == "benefits" and "user_location" in missing_fields:
        pending_query = str(
            tool_input.get("resolved_query")
            or state.get("standalone_question")
            or state.get("question")
            or memory.get("last_user_question", "")
        ).strip()

    if state.get("needs_clarification"):
        updated_memory = set_pending(
            memory=memory,
            route=route,
            missing_fields=missing_fields,
            pending_query=pending_query,
        )
    else:
        updated_memory = clear_pending(memory)

    latitude = user_location.get("latitude")
    longitude = user_location.get("longitude")
    if latitude and longitude:
        updated_memory["user_location"] = dict(user_location)

    if final_answer != FALLBACK_ANSWER:
        last_user_question = _get_memory_question(state, memory)
        updated_memory["last_user_question"] = mask_sensitive_text(last_user_question)
        updated_memory["last_assistant_answer"] = mask_sensitive_text(final_answer)

    parsed_statement = state.get("credit_card_statement")
    if isinstance(parsed_statement, dict) and parsed_statement:
        updated_memory["credit_card_statement"] = parsed_statement

    updated_memory["last_route"] = route
    updated_memory["last_topic"] = _infer_topic(route)
    updated_memory["updated_at"] = datetime.now(timezone.utc).isoformat()

    save_memory(session_id, updated_memory)

    log_step(
        "SAVE_MEMORY",
        "Memoria local actualizada",
        {
            "session_id": session_id,
            "pending_route": updated_memory.get("pending_route", ""),
            "pending_query": updated_memory.get("pending_query", ""),
            "missing_fields": updated_memory.get("missing_fields", []),
            "last_route": updated_memory.get("last_route", ""),
            "last_topic": updated_memory.get("last_topic", ""),
            "has_user_location": bool(updated_memory.get("user_location")),
            "has_credit_card_statement": bool(updated_memory.get("credit_card_statement")),
        },
    )

    return {
        **state,
        "memory": updated_memory,
        "pending_route": updated_memory.get("pending_route", ""),
        "missing_fields": updated_memory.get("missing_fields", []),
    }
