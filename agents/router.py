import re
import unicodedata

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from core.privacy import extract_identification, mask_sensitive_text
from observability.logger import log_step


VALID_ROUTES = {
    "chitchat",
    "loans_rag",
    "bcra_credit_status",
    "branch_locator",
    "benefits",
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

BRANCH_LOCATOR_PATTERNS = (
    "sucursal cercana",
    "sucursales cercanas",
    "sucursal mas cercana",
    "sucursales mas cercanas",
    "sucursal cerca",
    "sucursales cerca",
    "buscar sucursal",
    "encontrar sucursal",
    "donde hay una sucursal",
    "donde tengo una sucursal",
    "que sucursal tengo cerca",
    "sucursal tengo cerca",
    "galicia cerca",
    "banco cerca",
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

BCRA_FOLLOWUP_PATTERNS = (
    "malo",
    "mala",
    "bueno",
    "buena",
    "grave",
    "normal",
    "significa",
    "riesgo",
    "afecta",
    "credito",
    "deuda",
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
    "cuota del credito",
    "cuotas del credito",
    "precancel",
    "prendario",
    "deuda vencida",
    "adelanto",
)

BENEFITS_PATTERNS = (
    "beneficio",
    "beneficios",
    "promo",
    "promos",
    "promocion",
    "promociones",
    "descuento",
    "descuentos",
    "ahorro",
    "oferta",
    "ofertas",
    "cuota",
    "cuotas",
    "sin interes",
    "qr",
    "nfc",
    "contactless",
    "contact less",
    "sin contacto",
    "eminent",
    "eminent black",
    "seleccion exclusiva",
    "gastronomia",
    "supermercados",
    "supermercado",
    "super",
    "indumentaria",
    "ropa",
    "electronica",
    "tecnologia",
    "hogar",
    "casa",
)

BENEFITS_CONTEXT_PATTERNS = (
    "beneficio",
    "beneficios",
    "promo",
    "promos",
    "promocion",
    "promociones",
    "descuento",
    "descuentos",
    "ahorro",
    "oferta",
    "ofertas",
    "qr",
    "nfc",
    "contactless",
    "contact less",
    "sin contacto",
    "eminent",
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


def _contains_normalized_term(text: str, term: str) -> bool:
    return f" {term} " in f" {text} "


def _is_sensitive_request(normalized_question: str) -> bool:
    has_sensitive_term = any(term in normalized_question for term in SENSITIVE_TERMS)
    asks_to_reveal = any(verb in normalized_question for verb in SENSITIVE_VERBS)
    shares_sensitive_data = any(
        pattern in normalized_question for pattern in SENSITIVE_STATEMENTS
    )

    return has_sensitive_term and (asks_to_reveal or shares_sensitive_data)


def _is_branch_locator_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in BRANCH_LOCATOR_PATTERNS)


def _is_bcra_request(normalized_question: str) -> bool:
    return any(pattern in normalized_question for pattern in BCRA_PATTERNS)


def _is_bcra_followup_request(
    normalized_question: str,
    *,
    is_followup: bool,
    last_route: str,
    last_topic: str,
) -> bool:
    if not is_followup:
        return False

    if last_route != "bcra_credit_status" and last_topic != "situacion_crediticia_bcra":
        return False

    return any(pattern in normalized_question for pattern in BCRA_FOLLOWUP_PATTERNS)


def _is_bcra_identification_followup(
    question: str,
    *,
    pending_route: str,
    missing_fields: list[str],
    last_route: str,
    last_topic: str,
    last_assistant_answer: str,
) -> bool:
    if not extract_identification(question):
        return False

    if pending_route == "bcra_credit_status":
        return True

    if "identificacion" in missing_fields:
        return True

    if last_route == "bcra_credit_status":
        return True

    if last_topic == "situacion_crediticia_bcra":
        return True

    normalized_last_answer = _normalize_text(last_assistant_answer)

    return (
        "cuit" in normalized_last_answer
        or "cuil" in normalized_last_answer
        or "identificacion" in normalized_last_answer
    )


def _is_loans_request(normalized_question: str) -> bool:
    if _has_benefits_context(normalized_question):
        return False

    return any(pattern in normalized_question for pattern in LOANS_PATTERNS)


def _is_benefits_request(normalized_question: str) -> bool:
    if "caja de ahorro" in normalized_question:
        return False

    return any(
        _contains_normalized_term(normalized_question, pattern)
        for pattern in BENEFITS_PATTERNS
    )


def _has_benefits_context(normalized_question: str) -> bool:
    return any(
        _contains_normalized_term(normalized_question, pattern)
        for pattern in BENEFITS_CONTEXT_PATTERNS
    )


def _is_chitchat_request(normalized_question: str) -> bool:
    if (
        _is_loans_request(normalized_question)
        or _is_bcra_request(normalized_question)
        or _is_branch_locator_request(normalized_question)
        or _is_benefits_request(normalized_question)
    ):
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
    missing_fields = state.get("missing_fields") or memory.get("missing_fields", [])
    last_route = state.get("last_route") or memory.get("last_route", "")
    last_topic = state.get("last_topic") or memory.get("last_topic", "")
    last_assistant_answer = memory.get("last_assistant_answer", "")

    user_location = state.get("user_location") or {}
    latitude = user_location.get("latitude")
    longitude = user_location.get("longitude")

    if not question:
        log_step("ROUTER", "Fallback por pregunta vacia")
        return {
            **state,
            "route": "fallback",
            "error": "Pregunta vacia.",
        }

    if latitude and longitude and (
        pending_route == "branch_locator"
        or last_route == "branch_locator"
        or last_topic == "sucursales_cercanas"
    ):
        log_step(
            "ROUTER",
            "Ubicacion recibida para branch_locator",
            {
                "pending_route": pending_route,
                "last_route": last_route,
                "last_topic": last_topic,
            },
        )
        return {
            **state,
            "route": "branch_locator",
            "pending_route": "",
            "error": None,
        }

    if _is_bcra_identification_followup(
        question,
        pending_route=pending_route,
        missing_fields=missing_fields,
        last_route=last_route,
        last_topic=last_topic,
        last_assistant_answer=last_assistant_answer,
    ):
        log_step(
            "ROUTER",
            "Identificacion recibida para flujo BCRA",
            {
                "pending_route": pending_route,
                "missing_fields": missing_fields,
                "last_route": last_route,
                "last_topic": last_topic,
            },
        )
        return {
            **state,
            "route": "bcra_credit_status",
            "pending_route": "",
            "error": None,
        }

    if pending_route in {"bcra_credit_status", "branch_locator"}:
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
    elif _is_branch_locator_request(normalized_question):
        route = "branch_locator"
    elif _is_bcra_followup_request(
        normalized_question,
        is_followup=bool(state.get("is_followup")),
        last_route=last_route,
        last_topic=last_topic,
    ):
        route = "bcra_credit_status"
    elif _is_bcra_request(normalized_question):
        route = "bcra_credit_status"
    elif _is_loans_request(normalized_question):
        route = "loans_rag"
    elif _is_benefits_request(normalized_question):
        route = "benefits"
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
                            "chitchat, loans_rag, bcra_credit_status, branch_locator, benefits, fallback, sensitive.\n\n"
                            "Usa chitchat para saludos, agradecimientos o charla simple.\n"
                            "Usa loans_rag para consultas sobre prestamos, adelanto de sueldo, "
                            "cuotas de prestamos, precancelacion, prestamos hipotecarios o prendarios.\n"
                            "Usa bcra_credit_status cuando el usuario quiera consultar situacion "
                            "crediticia, Central de Deudores, BCRA o deudas bancarias.\n"
                            "Usa branch_locator cuando el usuario quiera buscar sucursales "
                            "Galicia cercanas a su ubicacion actual.\n"
                            "Usa benefits cuando el usuario pregunte por beneficios, promociones, "
                            "descuentos, ofertas, categorias de beneficios, cuotas en promociones, "
                            "promociones sin interes, pago QR, pago NFC o beneficios del "
                            "segmento Eminent o Eminent Black.\n"
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
