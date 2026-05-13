import os
import re
import unicodedata

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from observability.logger import log_step


CHITCHAT_MODEL = os.getenv("OPENAI_MODEL_CHITCHAT", "gpt-4o-mini")


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
        "Podés *escribirme lo que necesitás* o elegir un tema y avanzamos.\n\n"
        "Actualmente puedo ayudarte con:\n\n"
        "💰 Consultas sobre *préstamos*\n"
        "📊 Consulta de *situación crediticia*\n"
        "📍 Búsqueda de *sucursales cercanas*"
    )


def _fallback_chitchat_response(normalized_question: str, user_name: str = "") -> str:
    name_part = f", {user_name}" if user_name else ""

    if "gracias" in normalized_question:
        return f"De nada{name_part}. Cuando quieras, seguimos."

    if normalized_question in {"quien sos", "quien eres", "que sos"}:
        return (
            "Soy Gala, el asistente virtual de Banco Galicia.\n\n"
            "Actualmente puedo ayudarte con consultas sobre *préstamos*, "
            "*situación crediticia* o *búsqueda de sucursales cercanas*."
        )

    if normalized_question in {"como estas", "como andas"}:
        return (
            f"Muy bien{name_part}, gracias 😊\n\n"
            "Actualmente puedo ayudarte con consultas sobre *préstamos*, "
            "*situación crediticia* o *búsqueda de sucursales cercanas*."
        )

    return (
        f"Hola{name_part}. Actualmente puedo ayudarte con consultas sobre "
        "*préstamos*, *situación crediticia* o *búsqueda de sucursales cercanas*."
    )


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
- Redirigí suavemente hacia consultas sobre préstamos, situación crediticia o búsqueda de sucursales cercanas cuando corresponda.
- No digas siempre la misma frase de redirección.
- Si el usuario solo saluda, saludá y ofrecé ayuda bancaria de forma natural.
- Si agradece, respondé de forma breve y cálida.
- Si se despide, despedite de forma breve y amable.
- Si pregunta quién sos, explicá que sos Gala y mencioná brevemente en qué podés ayudar.
- Nunca cierres una respuesta casual con preguntas como “¿Te gusta Boca?”, “¿Querés hablar de eso?” o similares.
- Si se informa un nombre del usuario, podés usarlo naturalmente en saludos o despedidas.
- No uses el nombre del usuario en todas las respuestas.
- Actualmente podés ayudar con préstamos, situación crediticia y búsqueda de sucursales cercanas.
""".strip()

    user_context = ""
    if user_name:
        user_context = f"Nombre del usuario: {user_name}\n"

    user_prompt = f"""
{user_context}
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

    except Exception as error:
        log_step("CHITCHAT", "Error generando respuesta con LLM", {
            "error": str(error),
        })

        answer = _fallback_chitchat_response(normalized_question, user_name)

    log_step("CHITCHAT", "Respuesta conversacional generada", {
        "model": CHITCHAT_MODEL,
        "has_user_name": bool(user_name),
    })

    return {
        **state,
        "answer": answer,
        "error": None,
    }
