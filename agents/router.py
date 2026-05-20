import re
import unicodedata
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]

from agents.state import AgentState
from core.privacy import extract_identification, mask_sensitive_text
from observability.logger import log_step
from services.twilio_media import looks_like_pdf_media


VALID_ROUTES = {
    "chitchat",
    "loans_rag",
    "bcra_credit_status",
    "branch_locator",
    "benefits",
    "credit_card_statement",
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

CREDIT_CARD_STATEMENT_PATTERNS = (
    "resumen de tarjeta",
    "resumen de la tarjeta",
    "resumen tarjeta",
    "gastos del resumen",
    "consumos del resumen",
    "analizar resumen",
    "analizar mi resumen",
    "leer resumen",
    "te paso mi resumen de visa",
    "te paso mi resumen",
    "quiero analizar mi resumen",
    "resumen de visa",
    "pago minimo",
    "total a pagar",
    "vencimiento del resumen",
    "cierre del resumen",
)

CREDIT_CARD_SUMMARY_TERMS = (
    "resumen",
    "pdf",
    "visa",
    "mastercard",
    "items",
    "gastos",
    "consumos",
    "movimientos",
    "pago minimo",
    "total a pagar",
    "vencimiento",
    "cierre",
    "impuestos",
)

CREDIT_CARD_FOLLOWUP_PATTERNS = (
    "gasto",
    "gastos",
    "consumo",
    "consumos",
    "movimiento",
    "movimientos",
    "dolar",
    "dolares",
    "usd",
    "peso",
    "pesos",
    "ars",
    "impuesto",
    "impuestos",
    "cargo",
    "cargos",
    "interes",
    "intereses",
    "cuota",
    "cuotas",
    "mas grande",
    "mayor",
    "netflix",
    "amazon",
)

BANKING_FALLBACK_PATTERNS = (
    "mi cuenta",
    "cuenta bancaria",
    "caja de ahorro",
    "cuenta corriente",
    "saldo",
    "plata en mi cuenta",
    "abrir una cuenta",
    "abrir cuenta",
    "tarjeta nueva",
    "mi tarjeta",
    "tarjeta de credito",
    "tarjeta de debito",
    "limite de mi tarjeta",
    "aumentar el limite",
    "comprar dolares",
    "dolar",
    "dolares",
    "transferir plata",
    "transferencia",
    "transferencias",
    "home banking",
    "homebanking",
    "cambiar mi clave",
    "desbloqueo mi home banking",
    "sacar efectivo",
    "extraer efectivo",
    "resumen de la tarjeta",
    "pago el resumen",
    "desconocer un consumo",
    "desconocer consumo",
    "consumo",
    "inversion",
    "inversiones",
    "seguro",
    "seguros",
    "cbu",
    "alias",
)

GENERAL_CHITCHAT_PATTERNS = (
    "boca",
    "river",
    "partido",
    "libertadores",
    "mundial",
    "futbol",
    "clima",
    "pronostico",
    "llover",
    "pelicula",
    "peliculas",
    "serie",
    "series",
    "chiste",
    "receta",
    "torta",
    "cafe",
    "como preparo",
    "que significa",
    "significa esta palabra",
    "palabra",
)

CHITCHAT_CAPABILITIES_PATTERNS = (
    "que sabes hacer",
    "que podes hacer",
    "que puedes hacer",
    "en que me sabes ayudar",
    "en que me podes ayudar",
    "en que me puedes ayudar",
    "como me sabes ayudar",
    "como me podes ayudar",
    "como me puedes ayudar",
    "ayuda",
    "menu",
    "opciones",
)

CHITCHAT_EXACT = {
    "hola",
    "que tal",
    "buen dia",
    "buenos dias",
    "buenas",
    "buenas tardes",
    "buenas noches",
    "hello",
    "hi",
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


def _has_credit_card_statement_memory(memory: dict) -> bool:
    statement = memory.get("credit_card_statement")
    return isinstance(statement, dict) and bool(statement)


def _is_credit_card_statement_request(
    normalized_question: str,
    *,
    memory: dict,
    pending_route: str,
    last_route: str,
    last_topic: str,
    is_followup: bool,
    media: dict,
) -> bool:
    has_media = bool(media)
    has_pdf_media = looks_like_pdf_media(media)
    has_statement_memory = _has_credit_card_statement_memory(memory)
    has_summary_context = any(pattern in normalized_question for pattern in CREDIT_CARD_SUMMARY_TERMS)
    has_explicit_pattern = any(pattern in normalized_question for pattern in CREDIT_CARD_STATEMENT_PATTERNS)
    has_card_context = any(
        term in normalized_question
        for term in (
            "tarjeta",
            "tarjeta de credito",
            "tarjeta credito",
            "visa",
            "mastercard",
        )
    )
    has_analysis_intent = any(
        term in normalized_question
        for term in (
            "analizar",
            "analiza",
            "entender",
            "leer",
            "revisar",
            "ayudame",
            "ayudame a leer",
            "podrias analizar",
        )
    )
    has_followup_detail = any(
        pattern in normalized_question
        for pattern in CREDIT_CARD_FOLLOWUP_PATTERNS
    )

    if pending_route == "credit_card_statement":
        return True

    if has_pdf_media and (
        has_summary_context
        or has_card_context
        or has_analysis_intent
        or normalized_question == "analizar resumen de tarjeta adjunto"
    ):
        return True

    if has_explicit_pattern:
        return True

    if has_summary_context and (has_card_context or has_analysis_intent):
        return True

    if has_summary_context and any(
        term in normalized_question
        for term in ("visa", "mastercard", "pdf", "items", "vencimiento", "cierre")
    ):
        return True

    if (
        has_statement_memory
        and (last_route == "credit_card_statement" or last_topic == "resumen_tarjeta")
        and (has_followup_detail or "resumen" in normalized_question or is_followup)
    ):
        return True

    if has_media and has_card_context and has_summary_context:
        return True

    return False


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


def _is_banking_fallback_request(normalized_question: str) -> bool:
    return any(
        _contains_normalized_term(normalized_question, pattern)
        for pattern in BANKING_FALLBACK_PATTERNS
    )


def _is_general_non_banking_request(normalized_question: str) -> bool:
    return any(
        _contains_normalized_term(normalized_question, pattern)
        for pattern in GENERAL_CHITCHAT_PATTERNS
    )


def _is_capabilities_chitchat_request(normalized_question: str) -> bool:
    if any(
        _contains_normalized_term(normalized_question, pattern)
        for pattern in CHITCHAT_CAPABILITIES_PATTERNS
    ):
        return True

    tokens = normalized_question.split()
    if not tokens:
        return False

    if any(token in {"ayuda", "menu", "opciones"} for token in tokens):
        return True

    if len(tokens) <= 5 and "hacer" in tokens and any(
        token.startswith("pod") or token.startswith("sab")
        for token in tokens
    ):
        return True

    if any(token.startswith("ayud") for token in tokens) and any(
        token in {"que", "como", "en"}
        for token in tokens
    ):
        return True

    return False


def _is_chitchat_request(normalized_question: str) -> bool:
    if (
        _is_loans_request(normalized_question)
        or _is_bcra_request(normalized_question)
        or _is_branch_locator_request(normalized_question)
        or _is_benefits_request(normalized_question)
        or _is_banking_fallback_request(normalized_question)
    ):
        return False

    if normalized_question in CHITCHAT_EXACT:
        return True

    if normalized_question.startswith("gracias") and len(normalized_question.split()) <= 4:
        return True

    if normalized_question.startswith("hola") and len(normalized_question.split()) <= 4:
        return True

    if normalized_question.startswith("buenas") and len(normalized_question.split()) <= 4:
        return True

    if _is_capabilities_chitchat_request(normalized_question):
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
    media = state.get("media") or {}

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
        pending_route == "benefits"
    ):
        log_step(
            "ROUTER",
            "Ubicacion recibida para benefits",
            {
                "pending_route": pending_route,
                "last_route": last_route,
                "last_topic": last_topic,
            },
        )
        return {
            **state,
            "route": "benefits",
            "pending_route": "",
            "error": None,
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

    if pending_route in {"bcra_credit_status", "branch_locator", "benefits", "credit_card_statement"}:
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
    elif _is_credit_card_statement_request(
        normalized_question,
        memory=memory,
        pending_route=pending_route,
        last_route=last_route,
        last_topic=last_topic,
        is_followup=bool(state.get("is_followup")),
        media=media,
    ):
        route = "credit_card_statement"
    elif _is_loans_request(normalized_question):
        route = "loans_rag"
    elif _is_benefits_request(normalized_question):
        route = "benefits"
    elif _is_banking_fallback_request(normalized_question):
        route = "fallback"
    elif _is_general_non_banking_request(normalized_question):
        route = "chitchat"
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
                            "chitchat, loans_rag, bcra_credit_status, branch_locator, benefits, credit_card_statement, fallback, sensitive.\n\n"
                            "Definicion de chitchat: mensajes sociales, saludos, agradecimientos, "
                            "despedidas, preguntas sobre capacidades del bot, preguntas generales o temas no bancarios fuera del alcance del asistente.\n"
                            "Definicion de fallback: consultas bancarias o financieras que parecen "
                            "relevantes para Galicia pero que no estan cubiertas por los flujos disponibles.\n"
                            "Usa chitchat para saludos, agradecimientos, charla simple o temas no bancarios.\n"
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
                            "Usa credit_card_statement cuando el usuario quiera analizar un PDF del resumen "
                            "de su tarjeta, entender consumos, impuestos, total a pagar, pago minimo, "
                            "vencimiento o gastos en pesos/dolares del resumen ya analizado.\n"
                            "Usa sensitive si el usuario pide revelar, mostrar, recuperar o "
                            "compartir claves, contrasenas, PIN, tokens, CVV o datos privados.\n"
                            "Usa fallback solo para consultas bancarias o financieras no soportadas.\n\n"
                            "Ejemplos:\n"
                            "Usuario: que sabes hacer\n"
                            "Intent: chitchat\n"
                            "Usuario: me podes ayudar a hacer una torta?\n"
                            "Intent: chitchat\n"
                            "Usuario: cuando juega Boca por Libertadores?\n"
                            "Intent: chitchat\n"
                            "Usuario: quiero una tarjeta nueva\n"
                            "Intent: fallback\n"
                            "Usuario: cuanta plata tengo en mi cuenta?\n"
                            "Intent: fallback\n"
                            "Usuario: beneficios con QR\n"
                            "Intent: benefits\n"
                            "Usuario: promos de indumentaria\n"
                            "Intent: benefits\n"
                            "Usuario: quiero saber sobre prestamos personales\n"
                            "Intent: loans_rag\n"
                            "Usuario: tengo dudas con mi resumen de la tarjeta\n"
                            "Intent: credit_card_statement\n"
                            "Usuario: cuanto me cobraron de impuestos en el resumen\n"
                            "Intent: credit_card_statement"
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
            route = "fallback" if _is_banking_fallback_request(normalized_question) else "chitchat"

    if route not in VALID_ROUTES:
        route = "fallback" if _is_banking_fallback_request(normalized_question) else "chitchat"

    log_step("ROUTER", "Ruta seleccionada", {"route": route})

    return {
        **state,
        "route": route,
        "error": None,
    }
