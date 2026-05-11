from openai import OpenAI

from agents.state import AgentState
from core.constants import FALLBACK_ANSWER
from core.context import build_context
from core.privacy import mask_sensitive_text
from observability.logger import log_step
import unicodedata


PRODUCT_TYPE_BOOSTS = {
    "prestamo_personal": 0.12,
    "prestamos_general": 0.06,
}

NEGATIVE_PRODUCT_BOOSTS = {
    "prestamo_prendario": -0.10,
    "prestamo_hipotecario_uva": -0.10,
    "adelanto_sueldo": -0.08,
    "prestamo_express": -0.05,
}


PURPOSE_BOOSTS = {
    "general_info": 0.06,
    "condiciones": 0.05,
    "requisitos": 0.05,
    "documentacion": 0.05,
    "solicitud": 0.05,
    "cuotas": 0.03,
    "cancelacion": 0.02,
}


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


def _calculate_metadata_boost(query: str, result: dict) -> float:
    boost = 0.0

    normalized_query = _normalize(query)

    product_type = result.get("product_type", "otro")
    document_purpose = result.get("document_purpose", "otro")

    # ==========================================
    # PRÉSTAMOS PERSONALES
    # ==========================================

    if "prestamo personal" in normalized_query:
        boost += PRODUCT_TYPE_BOOSTS.get(product_type, 0.0)
        boost += PURPOSE_BOOSTS.get(document_purpose, 0.0)
        boost += NEGATIVE_PRODUCT_BOOSTS.get(product_type, 0.0)

    # ==========================================
    # CUOTIFICACIÓN
    # ==========================================

    if "cuotificacion" in normalized_query:
        if product_type == "cuotificacion":
            boost += 0.20
        else:
            boost -= 0.08

    # ==========================================
    # PRENDARIO
    # ==========================================

    if "prendario" in normalized_query:
        if product_type == "prestamo_prendario":
            boost += 0.18
        else:
            boost -= 0.08

    # ==========================================
    # HIPOTECARIO UVA
    # ==========================================

    if "hipotecario" in normalized_query or "uva" in normalized_query:
        if product_type == "prestamo_hipotecario_uva":
            boost += 0.18
        else:
            boost -= 0.08

    return boost


def retriever_node(
    state: AgentState,
    client: OpenAI,
    retriever,
    top_k: int,
    score_threshold: float,
) -> AgentState:
    query = (
        state.get("search_query")
        or state.get("standalone_question")
        or state["question"]
    )

    print("\n🔎 Query usada para retrieval:")
    print(mask_sensitive_text(query))

    candidate_top_k = max(top_k, 15)

    results = retriever.search(
        client=client,
        query=query,
        top_k=candidate_top_k,
    )

    print("\n🔎 Resultados crudos del retriever:")
    for idx, result in enumerate(results, 1):
        print(
            f"{idx}. {result.get('title')} "
            f"(score={result.get('score', 0):.3f})"
        )

    filtered_results = [
        result for result in results
        if result.get("score", 0.0) >= score_threshold
    ]

    # ==========================================
    # METADATA RERANK
    # ==========================================

    reranked_results = []

    for result in filtered_results:
        semantic_score = result.get("score", 0.0)

        metadata_boost = _calculate_metadata_boost(
            query=query,
            result=result,
        )

        final_score = semantic_score + metadata_boost

        result["semantic_score"] = semantic_score
        result["metadata_boost"] = metadata_boost
        result["final_score"] = final_score

        reranked_results.append(result)

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x.get("final_score", 0),
        reverse=True,
    )

    relevant_results = reranked_results[:3]

    if not relevant_results:
        return {
            **state,
            "documents": [],
            "context": "",
            "answer": FALLBACK_ANSWER,
            "final_answer": FALLBACK_ANSWER,
            "route": "fallback",
        }

    context = build_context(relevant_results)

    log_step("RETRIEVER", "Resultados recuperados", {
        "total": len(results),
    })

    log_step("RETRIEVER", "Resultados relevantes", {
        "relevantes": len(relevant_results),
    })

    print("\n📄 Contexto usado:")
    for idx, doc in enumerate(relevant_results, 1):
        print(
            f"{idx}. {doc.get('title')} "
            f"(semantic={doc.get('semantic_score', 0):.3f}, "
            f"boost={doc.get('metadata_boost', 0):+.3f}, "
            f"final={doc.get('final_score', 0):.3f}) "
            f"[{doc.get('product_type')}]"
        )

    return {
        **state,
        "documents": relevant_results,
        "context": context,
    }