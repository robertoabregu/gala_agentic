import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
"""


def _has_useful_memory(memory: dict) -> bool:
    return bool(memory.get("last_user_question") or memory.get("last_assistant_answer"))


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
