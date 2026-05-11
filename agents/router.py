import re
import unicodedata

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from core.privacy import mask_sensitive_text
from observability.logger import log_step


VALID_ROUTES = {
    "chitchat",
    "loans_rag",
    "bcra_credit_status",
    "fallback",
    "sensitive",
}

SENSITIVE_TERMS = (
    "clave",
    "contrasena",
    "password",
    "token",
    "pin",
    "cvv",
    "codigo de seguridad",
)

SENSITIVE_VERBS = (
    "decime",
    "dame",
    "mostrar",
    "mostrame",
    "ver",
    "saber",
    "recuperar",
    "adivinar",
    "compartir",
)

SENSITIVE_STATEMENTS = (
    "mi clave es",
    "mi contrasena es",
    "mi password es",
    "mi token es",
    "mi pin es",
    "mi cvv es",
    "te paso mi clave",
    "te paso mi contrasena",
)

BCRA_PATTERNS = (
    "situacion crediticia",
    "estado crediticio",
    "central de deudores",
    "central deudores",
    "deudores bcra",
    "deudores del bcra",
    "deudor bcra",
    "situacion en el bcra",
    "situacion en bcra",
    "deuda bancaria",
    "deuda con bancos",
    "deuda en bancos",
    "deuda en el bcra",
    "deuda en bcra",
    "deuda bcra",
    "historial crediticio",
)

LOANS_PATTERNS = (
    "prestamo",
    "prestamos",
    "adelanto de sueldo",
    "prestamo personal",
    "prestamos personales",
    "prestamo hipotecario",
    "prestamos hipotecarios",
    "hipotecario uva",
    "hipotecario",
    "cuota del prestamo",
    "cuotas del prestamo",
    "cuota",
    "precancel",
    "prendario",
    "deuda vencida",
    "adelanto",
)

CHITCHAT_EXACT = {
    "hola",
    "buen dia",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "gracias",
    "muchas gracias",
    "mil gracias",
    "quien sos",
    "quien eres",
    "como estas",
}


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


def _normalize_route(route: str) -> str:
    normalized = route.strip().lower()
    if normalized == "rag":
        return "loans_rag"
    return normalized


def _is_sensitive_request(normalized_question: str) -> bool:
    has_sensitive_term = any(term in normalized_question for term in SENSITIVE_TERMS)
    asks_to_reveal = any(verb in normalized_question for verb in SENSITIVE_VERBS)
    shares_sensitive_data = any(
        pattern in normalized_question for pattern in SENSITIVE_STATEMENTS
    )

    return has_sensitive_term and (asks_to_reveal or shares_sensitive_data)


def _is_bcra_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in BCRA_PATTERNS)


def _is_loans_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in LOANS_PATTERNS)


def _is_chitchat_request(normalized_question: str) -> bool:
    if _is_loans_request(normalized_question) or _is_bcra_request(normalized_question):
        return False

    if normalized_question in CHITCHAT_EXACT:
        return True

    if normalized_question.startswith("gracias") and len(normalized_question.split()) <= 4:
        return True

    if normalized_question.startswith("hola") and len(normalized_question.split()) <= 4:
        return True

    return False


def router_node(
    state: AgentState,
    llm: ChatOpenAI,
) -> AgentState:
    question = state.get("question", "").strip()
    routing_question = (state.get("standalone_question") or question).strip()
    memory = state.get("memory") or {}
    pending_route = state.get("pending_route") or memory.get("pending_route", "")

    if not question:
        log_step("ROUTER", "Fallback por pregunta vacia")
        return {
            **state,
            "route": "fallback",
            "error": "Pregunta vacia.",
        }

    if pending_route == "bcra_credit_status":
        log_step(
            "ROUTER",
            "Ruta recuperada desde memoria local",
            {"pending_route": pending_route},
        )
        return {
            **state,
            "route": pending_route,
            "error": None,
        }

    normalized_question = _normalize_text(routing_question)

    if _is_sensitive_request(normalized_question):
        route = "sensitive"
    elif _is_bcra_request(normalized_question):
        route = "bcra_credit_status"
    elif _is_loans_request(normalized_question):
        route = "loans_rag"
    elif _is_chitchat_request(normalized_question):
        route = "chitchat"
    else:
        masked_question = mask_sensitive_text(routing_question)

        try:
            response = llm.invoke(
                [
                    (
                        "system",
                        (
                            "Sos un clasificador de intencion para un chatbot bancario. "
                            "Tu unica tarea es decidir la ruta correcta. "
                            "Responde solamente una de estas palabras: "
                            "chitchat, loans_rag, bcra_credit_status, fallback, sensitive.\n\n"
                            "Usa chitchat para saludos, agradecimientos o charla simple.\n"
                            "Usa loans_rag para consultas sobre prestamos, adelanto de sueldo, "
                            "cuotas, precancelacion, prestamos hipotecarios o prendarios.\n"
                            "Usa bcra_credit_status cuando el usuario quiera consultar situacion "
                            "crediticia, Central de Deudores, BCRA o deudas bancarias.\n"
                            "Usa sensitive si el usuario pide revelar, mostrar, recuperar o "
                            "compartir claves, contrasenas, PIN, tokens, CVV o datos privados.\n"
                            "Usa fallback si la consulta no encaja en ninguna ruta anterior."
                        ),
                    ),
                    ("user", masked_question),
                ]
            )

            route = _normalize_route(response.content)
        except Exception as exc:
            log_step(
                "ROUTER",
                "Error clasificando con LLM",
                {"error": str(exc)},
            )
            route = "fallback"

    if route not in VALID_ROUTES:
        route = "fallback"

    log_step("ROUTER", "Ruta seleccionada", {"route": route})

    return {
        **state,
        "route": route,
        "error": None,
    }
