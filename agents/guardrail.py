import re
import unicodedata

from agents.state import AgentState
from core.constants import FALLBACK_ANSWER
from observability.logger import log_step


UNSAFE_ANSWER_PATTERNS = [
    "creo que",
    "probablemente",
    "podria ser",
    "en general",
    "normalmente",
    "seguramente",
]

SENSITIVE_TERMS = [
    "clave",
    "contrasena",
    "password",
    "token",
    "pin",
    "cvv",
    "codigo de seguridad",
]

SENSITIVE_VERBS = ["decime", "dame", "mostrar", "mostrame", "ver", "saber"]

SENSITIVE_STATEMENTS = [
    "mi clave es",
    "mi contrasena es",
    "mi password es",
    "mi token es",
    "mi pin es",
    "mi cvv es",
    "te paso mi clave",
    "te paso mi contrasena",
]


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


def guardrail_node(state: AgentState) -> AgentState:
    question = _normalize_text(state.get("question", ""))
    answer = state.get("answer", "").strip()
    route = state.get("route", "")

    if route == "sensitive":
        log_step("GUARDRAIL", "Bloqueo por ruta sensitive")
        return {
            **state,
            "final_answer": (
                "Por seguridad, no puedo ayudarte a consultar, mostrar, recuperar o revelar "
                "claves, contrasenas, PIN, tokens, CVV ni datos sensibles. "
                "Opera siempre desde los canales oficiales del banco."
            ),
        }

    if not answer:
        log_step("GUARDRAIL", "Fallback por respuesta vacia")
        return {
            **state,
            "final_answer": FALLBACK_ANSWER,
        }

    has_sensitive_term = any(term in question for term in SENSITIVE_TERMS)
    asks_to_reveal = any(word in question for word in SENSITIVE_VERBS)
    shares_sensitive_data = any(pattern in question for pattern in SENSITIVE_STATEMENTS)

    if has_sensitive_term and (asks_to_reveal or shares_sensitive_data):
        log_step("GUARDRAIL", "Bloqueo por intento de datos sensibles")
        return {
            **state,
            "final_answer": (
                "Por seguridad, no puedo ayudarte a consultar, mostrar o revelar claves, "
                "tokens, PIN, contrasenas, CVV ni datos sensibles. "
                "Te recomiendo operar siempre desde los canales oficiales del banco."
            ),
        }

    answer_lower = _normalize_text(answer)

    if any(pattern in answer_lower for pattern in UNSAFE_ANSWER_PATTERNS):
        log_step("GUARDRAIL", "Fallback por respuesta especulativa")
        return {
            **state,
            "final_answer": FALLBACK_ANSWER,
        }

    log_step("GUARDRAIL", "Respuesta aprobada")
    return {
        **state,
        "final_answer": answer,
    }
