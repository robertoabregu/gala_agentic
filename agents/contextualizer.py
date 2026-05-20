import json
import re
import unicodedata
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

from agents.state import AgentState
from core.privacy import extract_identification, mask_sensitive_text
from observability.logger import log_step


SYSTEM_PROMPT = """
Sos un modulo de contextualizacion conversacional.
Tu tarea NO es responder al usuario.
Tu tarea es decidir si la nueva pregunta depende del turno anterior.
Si depende, converti la nueva pregunta en una pregunta completa y autonoma.
Si no depende, dejala igual.

Devolve unicamente JSON valido con este formato:
{
  "is_followup": true,
  "standalone_question": "..."
}

Reglas:
- No respondas la pregunta.
- No inventes informacion nueva.
- Conserva el idioma del usuario.
- Si la nueva pregunta es independiente, deja standalone_question igual que current_question.
- Si la nueva pregunta depende del turno anterior, explicita el referente de pronombres como "eso", "asi", "ahi", "lo", "la".
- Si el turno previo fue sobre BCRA o situacion crediticia, y el follow-up depende de ese resultado, menciona Central de Deudores del BCRA o la situacion informada si corresponde.
- Si last_route es "credit_card_statement" o last_topic es "resumen_tarjeta", y la pregunta actual depende del resumen ya analizado, converti el follow-up en una pregunta completa sobre el resumen de tarjeta analizado previamente.

Ejemplos para resumen de tarjeta:
- current_question: y en dolares?
  standalone_question: Mostrame los consumos en dolares del resumen de tarjeta analizado previamente.
- current_question: cual fue el mas grande?
  standalone_question: Cual fue el consumo mas grande del resumen de tarjeta analizado previamente.
- current_question: y de impuestos?
  standalone_question: Cuanto me cobraron de impuestos en el resumen de tarjeta analizado previamente.
- current_question: mostrame los de Maria
  standalone_question: Mostrame los consumos de Maria en el resumen de tarjeta analizado previamente.
"""

LOCATION_PLACEHOLDER_PATTERNS = {
    "ubicacion compartida por whatsapp",
    "ubicación compartida por whatsapp",
}


def _has_useful_memory(memory: dict) -> bool:
    return bool(memory.get("last_user_question") or memory.get("last_assistant_answer"))


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


def _has_credit_card_statement_context(memory: dict) -> bool:
    statement = memory.get("credit_card_statement")
    if not isinstance(statement, dict) or not statement:
        return False

    return (
        memory.get("last_route") == "credit_card_statement"
        or memory.get("last_topic") == "resumen_tarjeta"
        or memory.get("pending_route") == "credit_card_statement"
    )


def _extract_statement_holders(memory: dict) -> list[str]:
    statement = memory.get("credit_card_statement")
    if not isinstance(statement, dict):
        return []

    holders = {
        str(item.get("titular") or "").strip()
        for item in statement.get("transactions", [])
        if str(item.get("titular") or "").strip()
    }
    return sorted(holders)


def _is_location_share_handoff(state: AgentState, memory: dict) -> bool:
    user_location = state.get("user_location") or {}
    latitude = user_location.get("latitude")
    longitude = user_location.get("longitude")
    if not latitude or not longitude:
        return False

    normalized_question = _normalize_text(state.get("question", ""))
    if normalized_question not in LOCATION_PLACEHOLDER_PATTERNS:
        return False

    pending_route = state.get("pending_route") or memory.get("pending_route", "")
    return pending_route in {"benefits", "branch_locator"}


def _rewrite_credit_card_followup(question: str, memory: dict) -> str | None:
    if not _has_credit_card_statement_context(memory):
        return None

    normalized_question = _normalize_text(question)
    if not normalized_question:
        return None

    if "resumen" in normalized_question and "tarjeta" in normalized_question:
        return None

    if normalized_question in {"y en dolares", "en dolares"}:
        return "Mostrame los consumos en dolares del resumen de tarjeta analizado previamente."

    if normalized_question in {"y en pesos", "en pesos"}:
        return "Mostrame los consumos en pesos del resumen de tarjeta analizado previamente."

    if normalized_question in {"y de impuestos", "de impuestos"} or (
        "impuesto" in normalized_question and len(normalized_question.split()) <= 5
    ):
        return "Cuanto me cobraron de impuestos en el resumen de tarjeta analizado previamente."

    if any(
        pattern in normalized_question
        for pattern in ("mas grande", "el mas grande", "la mas grande", "mayor gasto", "mayor consumo")
    ):
        return "Cual fue el consumo mas grande del resumen de tarjeta analizado previamente."

    holders = _extract_statement_holders(memory)
    for holder in holders:
        normalized_holder = _normalize_text(holder)
        if normalized_holder and normalized_holder in normalized_question:
            return (
                f"Mostrame los consumos de {holder} en el resumen de tarjeta analizado previamente."
            )

    if normalized_question.startswith("y en "):
        detail = normalized_question[5:].strip()
        if detail:
            return (
                f"Mostrame los consumos en {detail} del resumen de tarjeta analizado previamente."
            )

    if normalized_question.startswith("cuanto gaste en ") or normalized_question.startswith("cuanto gaste de "):
        merchant = normalized_question.split(" en ", 1)[-1].strip()
        if merchant and merchant != normalized_question:
            return (
                f"Cuanto gaste en {merchant} en el resumen de tarjeta analizado previamente."
            )

    if normalized_question.startswith("mostrame los de "):
        detail = normalized_question.replace("mostrame los de ", "", 1).strip()
        if detail:
            return (
                f"Mostrame los consumos de {detail} en el resumen de tarjeta analizado previamente."
            )

    return None


def _parse_response(content: str) -> dict:
    cleaned = (content or "").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta del contextualizador no es un objeto JSON.")

    is_followup = bool(parsed.get("is_followup"))
    standalone_question = str(parsed.get("standalone_question", "")).strip()

    return {
        "is_followup": is_followup,
        "standalone_question": standalone_question,
    }


def contextualizer_node(
    state: AgentState,
    llm: ChatOpenAI,
) -> AgentState:
    question = state.get("question", "").strip()
    memory = state.get("memory") or {}

    default_state = {
        **state,
        "original_question": question,
        "standalone_question": question,
        "is_followup": False,
    }

    if not question:
        log_step("CONTEXTUALIZER", "Pregunta vacia, no se contextualiza")
        return default_state

    if _is_location_share_handoff(state, memory):
        pending_route = state.get("pending_route") or memory.get("pending_route", "")
        pending_query = str(memory.get("pending_query") or "").strip()
        standalone_question = pending_query if pending_route == "benefits" and pending_query else question
        is_followup = standalone_question != question

        log_step(
            "CONTEXTUALIZER",
            "Handoff de ubicacion resuelto sin LLM",
            {
                "pending_route": pending_route,
                "is_followup": is_followup,
                "standalone": mask_sensitive_text(standalone_question),
            },
        )
        return {
            **state,
            "original_question": question,
            "standalone_question": standalone_question,
            "is_followup": is_followup,
        }

    if extract_identification(question):
        log_step(
            "CONTEXTUALIZER",
            "Se conserva la pregunta original por contener identificacion",
            {
                "original": mask_sensitive_text(question),
                "is_followup": False,
                "standalone": mask_sensitive_text(question),
            },
        )
        return default_state

    if not _has_useful_memory(memory):
        log_step(
            "CONTEXTUALIZER",
            "Sin memoria util, no se contextualiza",
            {
                "original": mask_sensitive_text(question),
                "is_followup": False,
                "standalone": mask_sensitive_text(question),
            },
        )
        return default_state

    deterministic_rewrite = _rewrite_credit_card_followup(question, memory)
    if deterministic_rewrite:
        log_step(
            "CONTEXTUALIZER",
            "Follow-up de resumen de tarjeta contextualizado por reglas",
            {
                "original": mask_sensitive_text(question),
                "is_followup": True,
                "standalone": mask_sensitive_text(deterministic_rewrite),
            },
        )
        return {
            **state,
            "original_question": question,
            "standalone_question": deterministic_rewrite,
            "is_followup": True,
        }

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"last_user_question: {mask_sensitive_text(memory.get('last_user_question', ''))}\n"
                        f"last_assistant_answer: {mask_sensitive_text(memory.get('last_assistant_answer', ''))}\n"
                        f"last_route: {memory.get('last_route', '')}\n"
                        f"last_topic: {memory.get('last_topic', '')}\n"
                        f"current_question: {mask_sensitive_text(question)}"
                    )
                ),
            ]
        )
        parsed = _parse_response(response.content)
        standalone_question = parsed["standalone_question"] or question
        is_followup = parsed["is_followup"] and standalone_question != question
    except Exception as exc:
        log_step(
            "CONTEXTUALIZER",
            "Error contextualizando, se usa la pregunta original",
            {"error": str(exc)},
        )
        return default_state

    log_step(
        "CONTEXTUALIZER",
        "Pregunta contextualizada",
        {
            "original": mask_sensitive_text(question),
            "is_followup": is_followup,
            "standalone": mask_sensitive_text(standalone_question),
        },
    )

    return {
        **state,
        "original_question": question,
        "standalone_question": standalone_question,
        "is_followup": is_followup,
    }
