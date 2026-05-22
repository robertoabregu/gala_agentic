from __future__ import annotations

import unittest

from agents.contextualizer import contextualizer_node
from agents.query_rewriter import query_rewriter_node
from agents.retriever_node import retriever_node
from core.constants import FALLBACK_ANSWER


class StaticLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, *_args, **_kwargs):
        class Response:
            pass

        response = Response()
        response.content = self.content
        return response


class ExplodingLLM:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("No deberia invocarse el LLM en este caso")


class DummyRetriever:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def search(self, client, query: str, top_k: int = 4) -> list[dict]:
        _ = client
        _ = query
        _ = top_k
        return [dict(result) for result in self.results]


def _base_state(question: str, **overrides):
    state = {
        "session_id": "demo",
        "memory": {},
        "pending_route": "",
        "question": question,
        "original_question": question,
        "standalone_question": question,
        "is_followup": False,
        "route": "loans_rag",
        "search_query": "",
        "documents": [],
        "context": "",
        "answer": "",
        "final_answer": "",
        "error": None,
        "missing_fields": [],
        "user_location": {},
        "media": {},
        "credit_card_statement": {},
    }
    state.update(overrides)
    return state


class LoansFlowTests(unittest.TestCase):
    def test_query_rewriter_preserves_explicit_loan_product(self) -> None:
        result = query_rewriter_node(
            _base_state("Hasta que porcentaje adelantan en adelanto de sueldo?"),
            llm=StaticLLM("porcentaje maximo monto disponible"),
        )

        self.assertIn("adelanto de sueldo", result["search_query"].lower())

    def test_contextualizer_rewrites_short_loan_followup_using_previous_product(self) -> None:
        result = contextualizer_node(
            _base_state(
                "cuanto es un monto menor?",
                memory={
                    "last_route": "loans_rag",
                    "last_topic": "prestamos",
                    "last_user_question": "Que es un prestamo express?",
                    "last_assistant_answer": (
                        "Un prestamo express te permite pedir un monto menor "
                        "y devolverlo en una sola cuota."
                    ),
                },
            ),
            llm=StaticLLM(
                (
                    '{"is_followup": true, '
                    '"standalone_question": "En el prestamo express, cuanto es un monto menor?"}'
                )
            ),
        )

        self.assertTrue(result["is_followup"])
        self.assertIn("prestamo express", result["standalone_question"].lower())

    def test_contextualizer_uses_loan_history_even_if_last_route_is_fallback(self) -> None:
        result = contextualizer_node(
            _base_state(
                "y la documentacion?",
                memory={
                    "last_route": "fallback",
                    "last_topic": "fallback",
                    "last_user_question": "Que documentacion necesito para un hipotecario uva?",
                    "last_assistant_answer": (
                        "Para el prestamo hipotecario uva vas a necesitar documentacion "
                        "de la propiedad."
                    ),
                },
            ),
            llm=StaticLLM(
                (
                    '{"is_followup": true, '
                    '"standalone_question": "Cual es la documentacion necesaria para el prestamo hipotecario uva?"}'
                )
            ),
        )

        self.assertTrue(result["is_followup"])
        self.assertIn("hipotecario uva", result["standalone_question"].lower())

    def test_contextualizer_does_not_stick_to_loans_when_topic_changes_to_branch_locator(self) -> None:
        question = "mostrame donde esta la sucu mas cerca"
        result = contextualizer_node(
            _base_state(
                question,
                memory={
                    "last_route": "loans_rag",
                    "last_topic": "prestamos",
                    "last_user_question": "En el adelanto de sueldo, no se puede pedir el 70%?",
                    "last_assistant_answer": "Podes pedir hasta el 50% de tu sueldo.",
                },
            ),
            llm=StaticLLM(
                (
                    '{"is_followup": false, '
                    '"standalone_question": "mostrame donde esta la sucu mas cerca"}'
                )
            ),
        )

        self.assertFalse(result["is_followup"])
        self.assertEqual(result["standalone_question"], question)

    def test_contextualizer_does_not_stick_to_loans_when_topic_changes_to_bcra(self) -> None:
        question = "quiero saber cual es mi situacion crediticia"
        result = contextualizer_node(
            _base_state(
                question,
                memory={
                    "last_route": "loans_rag",
                    "last_topic": "prestamos",
                    "last_user_question": "En el adelanto de sueldo, no se puede pedir el 70%?",
                    "last_assistant_answer": "Podes pedir hasta el 50% de tu sueldo.",
                },
            ),
            llm=StaticLLM(
                (
                    '{"is_followup": false, '
                    '"standalone_question": "quiero saber cual es mi situacion crediticia"}'
                )
            ),
        )

        self.assertFalse(result["is_followup"])
        self.assertEqual(result["standalone_question"], question)

    def test_contextualizer_supports_credit_card_followup_via_llm(self) -> None:
        result = contextualizer_node(
            _base_state(
                "y en dolares?",
                memory={
                    "last_route": "credit_card_statement",
                    "last_topic": "resumen_tarjeta",
                    "last_user_question": "Te pase mi resumen de tarjeta",
                    "last_assistant_answer": "Encontre consumos en pesos y dolares.",
                    "credit_card_statement": {
                        "metadata": {"transactions_count": 12},
                        "transactions": [{"titular": "Maria"}],
                    },
                },
            ),
            llm=StaticLLM(
                (
                    '{"is_followup": true, '
                    '"standalone_question": "Mostrame los consumos en dolares del resumen de tarjeta analizado previamente."}'
                )
            ),
        )

        self.assertTrue(result["is_followup"])
        self.assertIn("resumen de tarjeta", result["standalone_question"].lower())

    def test_retriever_rescues_borderline_exact_product_match(self) -> None:
        result = retriever_node(
            _base_state(
                "Hasta que porcentaje adelantan en adelanto de sueldo?",
                search_query="adelanto de sueldo porcentaje maximo monto disponible",
            ),
            client=None,
            retriever=DummyRetriever(
                [
                    {
                        "title": "Prestamo personal",
                        "content": "Info general de prestamos personales.",
                        "product_type": "prestamo_personal",
                        "document_purpose": "general_info",
                        "score": 0.49,
                    },
                    {
                        "title": "Adelanto de sueldo",
                        "content": "Podes pedir hasta el 50% de tu sueldo.",
                        "product_type": "adelanto_sueldo",
                        "document_purpose": "solicitud",
                        "score": 0.47,
                    },
                ]
            ),
            top_k=4,
            score_threshold=0.5,
        )

        self.assertEqual(result["documents"][0]["product_type"], "adelanto_sueldo")
        self.assertIn("50% de tu sueldo", result["context"])

    def test_retriever_keeps_loans_route_when_no_context_is_found(self) -> None:
        result = retriever_node(
            _base_state(
                "Hasta que porcentaje adelantan en adelanto de sueldo?",
                search_query="adelanto de sueldo porcentaje maximo monto disponible",
            ),
            client=None,
            retriever=DummyRetriever([]),
            top_k=4,
            score_threshold=0.5,
        )

        self.assertEqual(result["route"], "loans_rag")
        self.assertEqual(result["answer"], FALLBACK_ANSWER)
        self.assertEqual(result["final_answer"], FALLBACK_ANSWER)


if __name__ == "__main__":
    unittest.main()
