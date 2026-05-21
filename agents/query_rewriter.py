import re
import unicodedata
from typing import Any

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    ChatOpenAI = Any  # type: ignore[misc,assignment]

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - fallback solo para tests sin dependencias
    class HumanMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

from agents.state import AgentState
from core.privacy import mask_sensitive_text


SYSTEM_PROMPT = """
Sos un experto en optimizar consultas para el RAG de prestamos de Banco Galicia.

Tu tarea es reescribir preguntas de usuarios sobre prestamos para que sirvan mejor
en una busqueda semantica.
A este nodo solo le llegan consultas de prestamos.

Reglas:
- No respondas la pregunta.
- Devolve solo la query optimizada, sin comillas ni explicaciones.
- Mantene la intencion original.
- Conserva siempre el producto mencionado por el usuario si existe.
- Si la pregunta ya nombra un producto o atributo concreto, no lo reemplaces por otro.
- No agregues temas no mencionados por el usuario.
- No inventes politicas, condiciones de empresa, topes, requisitos o documentacion
  si el usuario no los pidio.
- Podes expandir con sinonimos utiles solo si mantienen exactamente la misma intencion.
- Si la pregunta ya es clara y especifica, hace cambios minimos.

Ejemplos:

Usuario:
"Que es un prestamo express?"

Query:
"prestamo express definicion caracteristicas como funciona"

Usuario:
"Hasta que porcentaje adelantan en adelanto de sueldo?"

Query:
"adelanto de sueldo porcentaje maximo monto disponible"

Usuario:
"Que documentacion necesito para un hipotecario uva?"

Query:
"prestamo hipotecario uva documentacion requisitos"
"""


PRODUCT_PATTERNS = {
    "prestamo_express": (
        "prestamo express",
        "prestamos express",
    ),
    "adelanto_sueldo": (
        "adelanto de sueldo",
        "adelantos de sueldo",
    ),
    "prestamo_personal": (
        "prestamo personal",
        "prestamos personales",
    ),
    "prestamo_hipotecario_uva": (
        "prestamo hipotecario uva",
        "prestamos hipotecarios",
        "hipotecario uva",
        "prestamo hipotecario",
    ),
    "prestamo_prendario": (
        "prestamo prendario",
        "prestamos prendarios",
        "prendario",
    ),
    "cuotificacion": (
        "cuotificacion",
    ),
    "refinanciacion": (
        "refinanciacion",
    ),
}

CANONICAL_PRODUCT_TERMS = {
    "prestamo_express": "prestamo express",
    "adelanto_sueldo": "adelanto de sueldo",
    "prestamo_personal": "prestamo personal",
    "prestamo_hipotecario_uva": "prestamo hipotecario uva",
    "prestamo_prendario": "prestamo prendario",
    "cuotificacion": "cuotificacion",
    "refinanciacion": "refinanciacion",
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


def _detect_products(text: str) -> set[str]:
    normalized_text = _normalize_text(text)
    return {
        product_key
        for product_key, patterns in PRODUCT_PATTERNS.items()
        if any(pattern in normalized_text for pattern in patterns)
    }


def _clean_rewritten_query(
    original_question: str,
    rewritten_query: str,
) -> str:
    cleaned = " ".join((rewritten_query or "").strip().strip('"').strip("'").split())
    if not cleaned:
        cleaned = original_question.strip()

    original_products = _detect_products(original_question)
    rewritten_products = _detect_products(cleaned)

    missing_products = [
        CANONICAL_PRODUCT_TERMS[product_key]
        for product_key in original_products
        if product_key not in rewritten_products
    ]

    if missing_products:
        cleaned = f"{cleaned} {' '.join(missing_products)}".strip()

    return cleaned


def query_rewriter_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    question = state.get("standalone_question") or state["question"]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ])

    rewritten_query = _clean_rewritten_query(
        original_question=question,
        rewritten_query=response.content,
    )

    print("\n[QUERY_REWRITER]")
    print(f"- original: {mask_sensitive_text(question)}")
    print(f"- rewritten: {mask_sensitive_text(rewritten_query)}")

    state["search_query"] = rewritten_query

    return state
