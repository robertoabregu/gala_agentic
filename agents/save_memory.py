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
    "chitchat": "conversacion",
    "fallback": "fallback",
}


def _infer_topic(route: str) -> str:
    return TOPIC_BY_ROUTE.get(route, route or "")


def save_memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "demo-local")
    memory = state.get("memory") or {}
    route = state.get("route", "")
    missing_fields = state.get("missing_fields", [])
    final_answer = state.get("final_answer", "")

    if state.get("needs_clarification"):
        updated_memory = set_pending(
            memory=memory,
            route=route,
            missing_fields=missing_fields,
        )
    else:
        updated_memory = clear_pending(memory)

    if final_answer != FALLBACK_ANSWER:
        last_user_question = (
            state.get("standalone_question")
            or state.get("question")
            or memory.get("last_user_question", "")
        )
        updated_memory["last_user_question"] = mask_sensitive_text(last_user_question)
        updated_memory["last_assistant_answer"] = mask_sensitive_text(final_answer)

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
            "missing_fields": updated_memory.get("missing_fields", []),
            "last_route": updated_memory.get("last_route", ""),
            "last_topic": updated_memory.get("last_topic", ""),
        },
    )

    return {
        **state,
        "memory": updated_memory,
        "pending_route": updated_memory.get("pending_route", ""),
        "missing_fields": updated_memory.get("missing_fields", []),
    }
