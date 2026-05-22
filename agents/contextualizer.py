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
Tu tarea es decidir si la nueva pregunta depende semanticamente del turno anterior.
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
- Considera follow-up solo cuando la nueva pregunta no se entiende bien sin el turno anterior.
- Si la nueva pregunta introduce un tema, producto o tramite distinto al del turno anterior, tratala como independiente.
- No arrastres entidades o productos del turno previo salvo que sean estrictamente necesarios para entender la pregunta actual.
- Si hay duda, devolve is_followup = false.
- Si el turno previo fue sobre BCRA o situacion crediticia, y el follow-up depende de ese resultado, menciona Central de Deudores del BCRA o la situacion informada si corresponde.
- Si el turno previo fue sobre un resumen de tarjeta ya analizado, y la nueva pregunta depende de ese resumen, converti el follow-up en una pregunta completa sobre ese resumen.
- Si el turno previo fue sobre prestamos, y la nueva pregunta depende claramente del ultimo producto mencionado, converti el follow-up en una pregunta completa que mencione ese producto.

Ejemplos:
- last_user_question: Que es un prestamo express?
  last_assistant_answer: Un prestamo express permite pedir un monto menor y devolverlo en una sola cuota.
  current_question: cuanto es un monto menor?
  output: {"is_followup": true, "standalone_question": "En el prestamo express, cuanto es un monto menor?"}

- last_user_question: En los adelantos de sueldo, hasta que porcentaje adelantan?
  last_assistant_answer: Podes pedir hasta el 50% de tu sueldo.
  current_question: y no se puede pedir el 70%?
  output: {"is_followup": true, "standalone_question": "En el adelanto de sueldo, no se puede pedir el 70%?"}

- last_user_question: En los adelantos de sueldo, hasta que porcentaje adelantan?
  last_assistant_answer: Podes pedir hasta el 50% de tu sueldo.
  current_question: mostrame donde esta la sucu mas cerca
  output: {"is_followup": false, "standalone_question": "mostrame donde esta la sucu mas cerca"}

- last_user_question: En los adelantos de sueldo, hasta que porcentaje adelantan?
  last_assistant_answer: Podes pedir hasta el 50% de tu sueldo.
  current_question: quiero saber cual es mi situacion crediticia
  output: {"is_followup": false, "standalone_question": "quiero saber cual es mi situacion crediticia"}

- last_user_question: Te pase mi resumen de tarjeta y lo analizaste.
  last_assistant_answer: Encontre consumos en pesos y dolares.
  current_question: y en dolares?
  output: {"is_followup": true, "standalone_question": "Mostrame los consumos en dolares del resumen de tarjeta analizado previamente."}

- last_user_question: Te pase mi resumen de tarjeta y lo analizaste.
  last_assistant_answer: Encontre consumos en pesos y dolares.
  current_question: quiero ver promociones en supermercados
  output: {"is_followup": false, "standalone_question": "quiero ver promociones en supermercados"}
""".strip()

LOCATION_PLACEHOLDER_PATTERNS = {
    "ubicacion compartida por whatsapp",
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


def _extract_statement_holders(statement: dict[str, Any]) -> list[str]:
    holders = {
        str(item.get("titular") or "").strip()
        for item in statement.get("transactions", [])
        if str(item.get("titular") or "").strip()
    }
    return sorted(holders)


def _build_credit_card_statement_context(memory: dict) -> dict[str, Any]:
    statement = memory.get("credit_card_statement")
    if not isinstance(statement, dict) or not statement:
        return {"has_statement": False}

    metadata = statement.get("metadata") or {}
    return {
        "has_statement": True,
        "holders": _extract_statement_holders(statement),
        "transactions_count": int(metadata.get("transactions_count") or 0),
        "taxes_and_fees_count": int(metadata.get("taxes_and_fees_count") or 0),
    }


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


def _build_prompt_payload(question: str, memory: dict) -> dict[str, Any]:
    return {
        "last_user_question": mask_sensitive_text(memory.get("last_user_question", "")),
        "last_assistant_answer": mask_sensitive_text(memory.get("last_assistant_answer", "")),
        "last_route": str(memory.get("last_route", "")),
        "last_topic": str(memory.get("last_topic", "")),
        "current_question": mask_sensitive_text(question),
        "credit_card_statement_context": _build_credit_card_statement_context(memory),
    }


def _parse_response(content: str) -> dict[str, Any]:
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

    try:
        payload = _build_prompt_payload(question, memory)
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        parsed = _parse_response(response.content)
        standalone_question = parsed["standalone_question"] or question
        is_followup = parsed["is_followup"] and standalone_question != question

        if not is_followup:
            standalone_question = question
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
