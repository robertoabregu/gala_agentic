from langchain_openai import ChatOpenAI

from agents.state import AgentState
from core.constants import FALLBACK_ANSWER
from observability.logger import log_step


def answer_node(
    state: AgentState,
    llm: ChatOpenAI,
) -> AgentState:
    question = state.get("standalone_question") or state["question"]
    context = state.get("context", "")

    if not context.strip():
        log_step("ANSWER", "Fallback por contexto vacío")
        return {
            **state,
            "answer": FALLBACK_ANSWER,
        }

    log_step("ANSWER", "Generando respuesta con ChatOpenAI")

    response = llm.invoke(
        [
            (
                "system",
                (
                    "Sos Gala, el asistente virtual de Banco Galicia. "
                    "Respondés consultas bancarias usando SOLO el contexto recuperado. "
                    "Tu tono es claro, amable, profesional y simple. "
                    "Usás voseo argentino cuando corresponde. "
                    "NO inventes información. "
                    "NO completes con conocimiento externo. "
                    "NO supongas datos que no aparezcan en el contexto. "
                    "IMPORTANTE:\n"
                    "- Si el contexto contiene información útil aunque sea parcial, "
                    "respondé usando SOLO esa información.\n"
                    "- NO uses fallback si podés responder aunque sea parcialmente.\n"
                    "- Solo usá fallback cuando el contexto realmente no sirva para responder.\n"
                    "- Si faltan algunos detalles específicos, podés aclararlo brevemente.\n"
                    "IMPORTANTE DE FORMATO:\n"
                    "- Para negritas usá formato WhatsApp con UN solo asterisco: *texto*.\n"
                    "- Nunca uses markdown con doble asterisco.\n"
                    "- Si usás etiquetas tipo concepto: detalle, escribilas como *Concepto:* detalle.\n"
                    "- No uses encabezados markdown con # ni links con formato [texto](url).\n"
                    "- Si listás pasos o requisitos, preferí bullets simples o líneas separadas.\n"
                    f"Fallback exacto:\n'{FALLBACK_ANSWER}'.\n"
                    "No pidas ni reveles claves, tokens, PIN, contraseñas, CVV ni datos sensibles. "
                    "Si la consulta requiere operar, indicá que lo haga desde los canales oficiales del banco."
                ),
            ),
            (
                "user",
                (
                    f"Pregunta del cliente:\n{question}\n\n"
                    f"Contexto recuperado:\n{context}\n\n"
                    "Instrucciones de respuesta:\n"
                    "- Respondé breve y claro.\n"
                    "- Usá bullets si ayuda.\n"
                    "- Usá SOLO información presente en el contexto.\n"
                    "- Si el contexto responde parcialmente, brindá esa respuesta parcial.\n"
                    "- NO inventes información faltante.\n"
                    "- Solo usá fallback si el contexto no aporta información útil.\n"
                    "- Usá *negritas* en palabras o frases importantes si aporta claridad.\n"
                    "- No abuses de las negritas.\n"
                    "- Si usás etiquetas, escribilas como *Etiqueta:* detalle.\n"
                ),
            ),
        ]
    )

    answer = response.content.strip()

    log_step("ANSWER", "Respuesta generada con ChatOpenAI")

    return {
        **state,
        "answer": answer,
    }
