import unicodedata

from openai import OpenAI

from agents.state import AgentState
from core.constants import FALLBACK_ANSWER
from core.context import build_context
from core.privacy import mask_sensitive_text
from observability.logger import log_step


PRODUCT_TYPE_BOOSTS = {
    "prestamo_personal": 0.12,
    "prestamos_general": 0.06,
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

SPECIFIC_LOAN_PRODUCT_TYPES = {
    "prestamo_personal",
    "prestamo_prendario",
    "prestamo_hipotecario_uva",
    "adelanto_sueldo",
    "prestamo_express",
    "cuotificacion",
}

EXACT_PRODUCT_BOOST = 0.18
NEGATIVE_EXACT_PRODUCT_BOOST = -0.08


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _boost_specific_product(
    normalized_query: str,
    product_type: str,
    document_purpose: str,
    *,
    patterns: tuple[str, ...],
    expected_product_type: str,
) -> float:
    if not _has_any(normalized_query, patterns):
        return 0.0

    if product_type == expected_product_type:
        return EXACT_PRODUCT_BOOST + PURPOSE_BOOSTS.get(document_purpose, 0.0)

    if product_type in SPECIFIC_LOAN_PRODUCT_TYPES:
        return NEGATIVE_EXACT_PRODUCT_BOOST

    return 0.0


def _calculate_metadata_boost(query: str, result: dict) -> float:
    boost = 0.0

    normalized_query = _normalize(query)
    product_type = result.get("product_type", "otro")
    document_purpose = result.get("document_purpose", "otro")

    if _has_any(normalized_query, ("prestamo personal", "prestamos personales")):
        boost += PRODUCT_TYPE_BOOSTS.get(product_type, 0.0)
        if product_type in {"prestamo_personal", "prestamos_general"}:
            boost += PURPOSE_BOOSTS.get(document_purpose, 0.0)
        elif product_type in SPECIFIC_LOAN_PRODUCT_TYPES:
            boost += NEGATIVE_EXACT_PRODUCT_BOOST

    boost += _boost_specific_product(
        normalized_query,
        product_type,
        document_purpose,
        patterns=("prestamo express", "prestamos express"),
        expected_product_type="prestamo_express",
    )

    boost += _boost_specific_product(
        normalized_query,
        product_type,
        document_purpose,
        patterns=("adelanto de sueldo", "adelantos de sueldo"),
        expected_product_type="adelanto_sueldo",
    )

    boost += _boost_specific_product(
        normalized_query,
        product_type,
        document_purpose,
        patterns=("cuotificacion",),
        expected_product_type="cuotificacion",
    )

    boost += _boost_specific_product(
        normalized_query,
        product_type,
        document_purpose,
        patterns=("prendario", "prestamo prendario", "prestamos prendarios"),
        expected_product_type="prestamo_prendario",
    )

    boost += _boost_specific_product(
        normalized_query,
        product_type,
        document_purpose,
        patterns=("hipotecario", "uva", "prestamo hipotecario"),
        expected_product_type="prestamo_hipotecario_uva",
    )

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

    print("\n[RETRIEVER] Query usada para retrieval:")
    print(mask_sensitive_text(query))

    candidate_top_k = max(top_k, 15)

    results = retriever.search(
        client=client,
        query=query,
        top_k=candidate_top_k,
    )

    print("\n[RETRIEVER] Resultados crudos:")
    for idx, result in enumerate(results, 1):
        print(
            f"{idx}. {result.get('title')} "
            f"(score={result.get('score', 0):.3f})"
        )

    reranked_results = []

    for result in results:
        semantic_score = result.get("score", 0.0)

        metadata_boost = _calculate_metadata_boost(
            query=query,
            result=result,
        )

        final_score = semantic_score + metadata_boost

        result["semantic_score"] = semantic_score
        result["metadata_boost"] = metadata_boost
        result["final_score"] = final_score

        if semantic_score >= score_threshold or final_score >= score_threshold:
            reranked_results.append(result)

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x.get("final_score", 0),
        reverse=True,
    )

    relevant_results = reranked_results[:3]

    if not relevant_results:
        log_step(
            "RETRIEVER",
            "Sin resultados relevantes para construir contexto",
            {
                "total": len(results),
                "threshold": score_threshold,
            },
        )
        return {
            **state,
            "documents": [],
            "context": "",
            "answer": FALLBACK_ANSWER,
            "final_answer": FALLBACK_ANSWER,
        }

    context = build_context(relevant_results)

    log_step("RETRIEVER", "Resultados recuperados", {
        "total": len(results),
    })

    log_step("RETRIEVER", "Resultados relevantes", {
        "relevantes": len(relevant_results),
    })

    print("\n[RETRIEVER] Contexto usado:")
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
