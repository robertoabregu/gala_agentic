import os
import re
import unicodedata

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from observability.logger import log_step


CHITCHAT_MODEL = os.getenv("OPENAI_MODEL_CHITCHAT", "gpt-4o-mini")

CAPABILITIES_PATTERNS = {
    "que sabes hacer",
    "que podes hacer",
    "que puedes hacer",
    "en que me sabes ayudar",
    "en que me podes ayudar",
    "en que me puedes ayudar",
    "como me podes ayudar",
    "como me puedes ayudar",
    "ayuda",
    "menu",
    "opciones",
}

IDENTITY_PATTERNS = {"quien sos", "quien eres", "que sos"}
THANKS_PATTERNS = {"gracias", "muchas gracias", "mil gracias"}
STATUS_PATTERNS = {"como estas", "como andas"}
FAREWELL_PATTERNS = {"chau", "adios", "hasta luego", "nos vemos"}


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


def _get_user_name(state: AgentState) -> str:
    memory = state.get("memory") or {}
    return (memory.get("user_name") or memory.get("nombre") or "").strip()


def _is_greeting(normalized_question: str) -> bool:
    greetings = {
        "hola",
        "que tal",
        "buen dia",
        "buenos dias",
        "buenas",
        "buenas tardes",
        "buenas noches",
        "hello",
        "hi",
    }

    if normalized_question in greetings:
        return True

    if normalized_question.startswith("hola ") and len(normalized_question.split()) <= 5:
        return True

    if normalized_question.startswith("buenas ") and len(normalized_question.split()) <= 5:
        return True

    return False


def _build_greeting_response(user_name: str = "") -> str:
    name_part = f", *{user_name}*" if user_name else ""

    return (
        f"👋 Hola{name_part}, ¿cómo estás? Soy *Gala*.\n\n"
        "Podés escribirme lo que necesitás o elegir un tema y avanzamos.\n\n"
        "Actualmente puedo ayudarte con:\n\n"
        "💰 Consultas sobre *préstamos*\n"
        "📊 Consulta de *situación crediticia*\n"
        "📍 Búsqueda de *sucursales cercanas*\n"
        "🎁 Búsqueda de *beneficios*"
    )


def _is_capabilities_request(normalized_question: str) -> bool:
    if normalized_question in CAPABILITIES_PATTERNS:
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
        token in {"que", "qu", "como", "en"}
        for token in tokens
    ):
        return True

    return False


def _build_capabilities_response(user_name: str = "") -> str:
    return _build_greeting_response(user_name)


def _fallback_chitchat_response(
    normalized_question: str,
    *,
    user_name: str = "",
) -> str:
    name_part = f", {user_name}" if user_name else ""

    if normalized_question in THANKS_PATTERNS or "gracias" in normalized_question:
        return (
            "De nada. Cuando quieras, puedo ayudarte con préstamos, "
            "situación crediticia, sucursales o beneficios."
        )

    if normalized_question in IDENTITY_PATTERNS:
        return (
            "Soy Gala, el asistente virtual de Banco Galicia. "
            "Puedo ayudarte con consultas sobre *préstamos*, *situación crediticia*, "
            "*sucursales cercanas* y *beneficios*."
        )

    if _is_capabilities_request(normalized_question):
        return _build_capabilities_response(user_name=user_name)

    if normalized_question in STATUS_PATTERNS:
        return (
            f"Muy bien{name_part}, gracias. "
            "Puedo ayudarte con consultas sobre *préstamos*, *situación crediticia*, "
            "*sucursales cercanas* y *beneficios*."
        )

    if normalized_question in FAREWELL_PATTERNS:
        return (
            "Hasta luego. Cuando quieras, puedo ayudarte con préstamos, "
            "situación crediticia, sucursales o beneficios."
        )

    return (
        "No puedo ayudarte con eso, pero sí puedo darte una mano con consultas sobre "
        "préstamos, situación crediticia, sucursales o beneficios."
    )


def _strip_leading_greeting(answer: str) -> str:
    cleaned = re.sub(
        r"^\W*(hola|buenas|buen dia|buenos dias|buenas tardes|buenas noches)\b[\W_]*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def chitchat_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    normalized_question = _normalize_text(question)
    user_name = _get_user_name(state)

    if _is_greeting(normalized_question):
        answer = _build_greeting_response(user_name)

        log_step("CHITCHAT", "Saludo inicial generado", {
            "has_user_name": bool(user_name),
        })

        return {
            **state,
            "answer": answer,
            "topic": "saludo",
            "error": None,
        }

    if _is_capabilities_request(normalized_question):
        answer = _build_capabilities_response(user_name=user_name)

        log_step("CHITCHAT", "Menu de capacidades generado", {
            "has_user_name": bool(user_name),
        })

        return {
            **state,
            "answer": answer,
            "topic": "capacidades",
            "error": None,
        }

    llm = ChatOpenAI(
        model=CHITCHAT_MODEL,
        temperature=0.5,
    )

    system_prompt = """
Sos Gala, el asistente virtual de Banco Galicia.

Respondés mensajes conversacionales breves, como saludos, agradecimientos,
despedidas, preguntas casuales o comentarios simples.

Tono:
- claro
- amable
- natural
- profesional
- argentino, con voseo cuando corresponda

Reglas:
- Respondé en 1 o 2 frases como máximo.
- No des información bancaria si el usuario no la pidió.
- No inventes datos.
- No pidas claves, tokens, PIN, contraseñas, CVV ni datos sensibles.
- Si el usuario hace charla casual fuera del alcance bancario, respondé amable pero NO continúes el tema.
- No hagas preguntas para seguir conversaciones casuales.
- No opines sobre fútbol, política, famosos, noticias, clima u otros temas fuera del alcance.
- Redirigí suavemente hacia consultas sobre préstamos, situación crediticia, sucursales o beneficios cuando corresponda.
- Si el mensaje es un tema no bancario fuera de alcance, una respuesta válida es:
  "No puedo ayudarte con eso, pero sí puedo darte una mano con consultas sobre préstamos, situación crediticia, sucursales o beneficios."
- Si el usuario solo saluda, saludá y ofrecé ayuda bancaria de forma natural.
- Si el usuario pregunta qué podés hacer, respondé con el menú de capacidades del asistente.
- Si agradece, respondé de forma breve y cálida.
- Si se despide, despedite de forma breve y amable.
- Si pregunta quién sos, explicá que sos Gala y mencioná brevemente en qué podés ayudar.
- Si la conversación ya está empezada y el usuario no saludó ni pidió el menú, no empieces con "Hola", "Buenas" ni similares.
- Nunca cierres una respuesta casual con preguntas como “¿Te gusta Boca?”, “¿Querés hablar de eso?” o similares.
- Si se informa un nombre del usuario, podés usarlo naturalmente en saludos o despedidas.
- No uses el nombre del usuario en todas las respuestas.
- Actualmente podés ayudar con préstamos, situación crediticia, búsqueda de sucursales cercanas y beneficios.
""".strip()

    user_context = ""
    if user_name:
        user_context = f"Nombre del usuario: {user_name}\n"

    user_prompt = f"""
{user_context}
Saludo habilitado: no

Mensaje del usuario:
{question}
""".strip()

    try:
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        answer = response.content.strip()
        answer = _strip_leading_greeting(answer)
        if not answer:
            answer = _fallback_chitchat_response(
                normalized_question,
                user_name=user_name,
            )

    except Exception as error:
        log_step("CHITCHAT", "Error generando respuesta con LLM", {
            "error": str(error),
        })

        answer = _fallback_chitchat_response(
            normalized_question,
            user_name=user_name,
        )

    log_step("CHITCHAT", "Respuesta conversacional generada", {
        "model": CHITCHAT_MODEL,
        "has_user_name": bool(user_name),
    })

    return {
        **state,
        "answer": answer,
        "error": None,
    }
