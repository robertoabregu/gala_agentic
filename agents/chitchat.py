import os
import re
import unicodedata

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from observability.logger import log_step


CHITCHAT_MODEL = os.getenv("OPENAI_MODEL_CHITCHAT", "gpt-4o-mini")

CAPABILITIES_PATTERNS = {
    "que podes hacer",
    "que puedes hacer",
    "en que me podes ayudar",
    "en que me puedes ayudar",
    "como me podes ayudar",
    "como me puedes ayudar",
    "ayuda",
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


def _conversation_started(state: AgentState) -> bool:
    memory = state.get("memory") or {}
    return bool(
        state.get("is_followup")
        or memory.get("last_user_question")
        or memory.get("last_assistant_answer")
    )


def _is_greeting(normalized_question: str) -> bool:
    greetings = {
        "hola",
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

    return f"¡Hola{name_part}! Puedo ayudarte con consultas sobre *préstamos* y *beneficios*."


def _is_capabilities_request(normalized_question: str) -> bool:
    return normalized_question in CAPABILITIES_PATTERNS


def _should_greet(state: AgentState, normalized_question: str) -> bool:
    if _is_greeting(normalized_question):
        return True

    return not _conversation_started(state) and _is_capabilities_request(normalized_question)


def _build_capabilities_response(*, should_greet: bool, user_name: str = "") -> str:
    if should_greet:
        return _build_greeting_response(user_name)

    return "Puedo ayudarte con consultas sobre *préstamos* y *beneficios*."


def _fallback_chitchat_response(
    normalized_question: str,
    *,
    should_greet: bool,
    user_name: str = "",
) -> str:
    name_part = f", {user_name}" if user_name else ""

    if normalized_question in THANKS_PATTERNS or "gracias" in normalized_question:
        return "De nada. Cuando quieras, puedo ayudarte con préstamos o beneficios."

    if normalized_question in IDENTITY_PATTERNS:
        return (
            "Soy Gala, el asistente virtual de Banco Galicia. "
            "Puedo ayudarte con consultas sobre *préstamos* y *beneficios*."
        )

    if _is_capabilities_request(normalized_question):
        return _build_capabilities_response(should_greet=should_greet, user_name=user_name)

    if normalized_question in STATUS_PATTERNS:
        return (
            f"Muy bien{name_part}, gracias. "
            "Puedo ayudarte con consultas sobre *préstamos* y *beneficios*."
        )

    if normalized_question in FAREWELL_PATTERNS:
        return "Hasta luego. Cuando quieras, puedo ayudarte con préstamos o beneficios."

    return "No puedo ayudarte con eso, pero sí puedo darte una mano con consultas sobre préstamos o beneficios."


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
    should_greet = _should_greet(state, normalized_question)

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
- Redirigí suavemente hacia consultas sobre préstamos o beneficios cuando corresponda.
- Si el mensaje es un tema no bancario fuera de alcance, una respuesta válida es:
  "No puedo ayudarte con eso, pero sí puedo darte una mano con consultas sobre préstamos o beneficios."
- Si el usuario solo saluda, saludá y ofrecé ayuda bancaria de forma natural.
- Si agradece, respondé de forma breve y cálida.
- Si se despide, despedite de forma breve y amable.
- Si pregunta quién sos, explicá que sos Gala y mencioná brevemente en qué podés ayudar.
- Solo saludá si es el primer turno o si el usuario saludó explícitamente.
- Si la conversación ya está empezada y el usuario no saludó, no empieces con "Hola", "Buenas" ni similares.
- Nunca cierres una respuesta casual con preguntas como “¿Te gusta Boca?”, “¿Querés hablar de eso?” o similares.
- Si se informa un nombre del usuario, podés usarlo naturalmente en saludos o despedidas.
- No uses el nombre del usuario en todas las respuestas.
- Actualmente podés ayudar con préstamos y beneficios.
""".strip()

    user_context = ""
    if user_name:
        user_context = f"Nombre del usuario: {user_name}\n"

    user_prompt = f"""
{user_context}
Saludo habilitado: {"si" if should_greet else "no"}

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
        if not should_greet:
            answer = _strip_leading_greeting(answer)
        if not answer:
            answer = _fallback_chitchat_response(
                normalized_question,
                should_greet=should_greet,
                user_name=user_name,
            )

    except Exception as error:
        log_step("CHITCHAT", "Error generando respuesta con LLM", {
            "error": str(error),
        })

        answer = _fallback_chitchat_response(
            normalized_question,
            should_greet=should_greet,
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
