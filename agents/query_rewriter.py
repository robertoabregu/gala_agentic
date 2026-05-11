from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from core.privacy import mask_sensitive_text


SYSTEM_PROMPT = """
Sos un experto en optimizar consultas para motores RAG.

Tu tarea es transformar preguntas de usuarios en consultas cortas,
claras y específicas para búsqueda semántica.

Reglas:
- No respondas la pregunta.
- Solo devolvé la query optimizada.
- Expandí términos implícitos.
- Agregá palabras clave útiles.
- Mantené la intención original.

Ejemplos:

Usuario:
"Qué necesito para sacar una tarjeta?"

Query:
"requisitos solicitar tarjeta crédito débito documentación"

Usuario:
"Cómo hago para pedir un préstamo personal?"

Query:
"requisitos préstamo personal documentación condiciones solicitar préstamo"

Usuario:
"Cómo desbloqueo mi tarjeta?"

Query:
"desbloquear tarjeta pasos app online banking"
"""


def query_rewriter_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    question = state.get("standalone_question") or state["question"]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ])

    rewritten_query = response.content.strip().strip('"').strip("'")

    print("\n[QUERY_REWRITER]")
    print(f"- original: {mask_sensitive_text(question)}")
    print(f"- rewritten: {mask_sensitive_text(rewritten_query)}")

    state["search_query"] = rewritten_query

    return state
