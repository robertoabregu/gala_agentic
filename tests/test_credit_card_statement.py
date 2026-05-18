from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.contextualizer import contextualizer_node
from agents.credit_card_statement import credit_card_statement_node
from agents.router import router_node
from tools.credit_card_pdf_parser import parse_credit_card_statement_text_pages


SAMPLE_PAGES = [
    """
    Tarjeta Crédito VISA
    ROBERTO ARIEL ABREGU
    VISA
    2.202.255,21
    617,59
    20-Nov-25
    01-Dic-25
    24-Dic-25
    05-Ene-26
    22-Ene-26
    02-Feb-26
    PAGO MINIMO
    LÍMITES
    En pesos
    $ 528.260,00
    De compras en un pago y en cuotas
    $ 10.000.000,00
    De financiación
    $ 9.000.000,00
    DETALLE DEL CONSUMO
    FECHA
    REFERENCIA
    CUOTA
    COMPROBANTE
    PESOS
    DÓLARES
    20-11-25
    K
    AMAZON RETA* B002C2QU1    USD      245,61
    173215
    245,61
    21-11-25
    *
    SUPERMERCADO CENTRAL
    173216
    25.000,00
    TARJETA 9550 Total Consumos de ROBERTO ARIEL ABREGU
    25.000,00
    245,61
    22-11-25
    *
    NETFLIX.COM      570991852USD       10,43
    173217
    10,43
    23-11-25
    *
    TIENDA ONLINE
    03/06
    173218
    50.000,00
    TARJETA 1603 Total Consumos de MARIA DOLOR VASQUEZ
    50.000,00
    10,43
    24-12-25
    IMPUESTO DE SELLOS        $
    25.976,93
    24-12-25
    DB.RG 5617  30% (   874358,09 )
    262.307,42
    TOTAL A PAGAR
    2.202.255,21
    617,59
    Plan V: texto legal e informativo que no deberia ser tomado como movimiento.
    """,
]


def build_statement() -> dict:
    return parse_credit_card_statement_text_pages(SAMPLE_PAGES)


class DummyLLM:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("El LLM no deberia usarse en este test.")


class RouterFallbackLLM:
    def invoke(self, *_args, **_kwargs):
        class Response:
            content = "fallback"

        return Response()


class CreditCardStatementTests(unittest.TestCase):
    def test_parser_extracts_summary_and_movements(self) -> None:
        statement = build_statement()
        summary = statement["summary"]
        metadata = statement["metadata"]

        self.assertEqual(summary["issuer"], "VISA")
        self.assertEqual(summary["fecha_cierre_actual"], "2025-12-24")
        self.assertEqual(summary["fecha_vencimiento_actual"], "2026-01-05")
        self.assertEqual(summary["proximo_cierre"], "2026-01-22")
        self.assertEqual(summary["proximo_vencimiento"], "2026-02-02")
        self.assertAlmostEqual(summary["total_pesos"], 2202255.21)
        self.assertAlmostEqual(summary["total_dolares"], 617.59)
        self.assertAlmostEqual(summary["pago_minimo"], 528260.00)
        self.assertEqual(metadata["transactions_count"], 4)
        self.assertEqual(metadata["taxes_and_fees_count"], 2)
        self.assertEqual(statement["transactions"][0]["tarjeta"], "9550")
        self.assertEqual(statement["transactions"][0]["titular"], "ROBERTO")
        self.assertEqual(statement["transactions"][-1]["tarjeta"], "1603")
        self.assertEqual(statement["transactions"][-1]["titular"], "MARIA")

    def test_node_requests_pdf_when_missing(self) -> None:
        result = credit_card_statement_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "Quiero analizar mi resumen de tarjeta",
                "original_question": "Quiero analizar mi resumen de tarjeta",
                "standalone_question": "Quiero analizar mi resumen de tarjeta",
                "is_followup": False,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
                "media": {},
            },
            llm=None,
        )

        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["missing_fields"], ["pdf"])
        self.assertEqual(result["route"], "credit_card_statement")

    def test_node_rejects_non_pdf_media(self) -> None:
        result = credit_card_statement_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "credit_card_statement",
                "question": "Te paso mi resumen",
                "original_question": "Te paso mi resumen",
                "standalone_question": "Te paso mi resumen",
                "is_followup": False,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
                "media": {
                    "num_media": "1",
                    "url": "https://example.test/file.jpg",
                    "content_type": "image/jpeg",
                    "filename": "file.jpg",
                },
            },
            llm=None,
        )

        self.assertTrue(result["needs_clarification"])
        self.assertIn("PDF", result["answer"])

    @patch("agents.credit_card_statement.parse_credit_card_statement_pdf")
    @patch("agents.credit_card_statement.download_twilio_pdf_to_tempfile")
    def test_node_parses_pdf_and_builds_initial_summary(
        self,
        mock_download,
        mock_parse,
    ) -> None:
        mock_download.return_value = "temporary.pdf"
        mock_parse.return_value = build_statement()

        result = credit_card_statement_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "credit_card_statement",
                "question": "Analizar resumen de tarjeta adjunto",
                "original_question": "Analizar resumen de tarjeta adjunto",
                "standalone_question": "Analizar resumen de tarjeta adjunto",
                "is_followup": False,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
                "media": {
                    "num_media": "1",
                    "url": "https://example.test/statement.pdf",
                    "content_type": "application/pdf",
                    "filename": "statement.pdf",
                },
            },
            llm=None,
        )

        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["route"], "credit_card_statement")
        self.assertIn("Analicé tu resumen", result["answer"])
        self.assertIn("Total a pagar", result["answer"])

    def test_contextualizer_rewrites_credit_card_followup(self) -> None:
        statement = build_statement()
        result = contextualizer_node(
            {
                "session_id": "demo",
                "memory": {
                    "last_user_question": "Mostrame los consumos del resumen",
                    "last_assistant_answer": "Analicé tu resumen de tarjeta.",
                    "last_route": "credit_card_statement",
                    "last_topic": "resumen_tarjeta",
                    "credit_card_statement": statement,
                },
                "pending_route": "",
                "question": "y en dólares?",
                "original_question": "",
                "standalone_question": "",
                "is_followup": False,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
            },
            llm=DummyLLM(),
        )

        self.assertTrue(result["is_followup"])
        self.assertEqual(
            result["standalone_question"],
            "Mostrame los consumos en dolares del resumen de tarjeta analizado previamente.",
        )

    def test_router_routes_credit_card_followup(self) -> None:
        statement = build_statement()
        result = router_node(
            {
                "session_id": "demo",
                "memory": {
                    "last_route": "credit_card_statement",
                    "last_topic": "resumen_tarjeta",
                    "credit_card_statement": statement,
                },
                "pending_route": "",
                "missing_fields": [],
                "question": "y en dólares?",
                "original_question": "y en dólares?",
                "standalone_question": "Mostrame los consumos en dolares del resumen de tarjeta analizado previamente.",
                "is_followup": True,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
                "media": {},
            },
            llm=RouterFallbackLLM(),
        )

        self.assertEqual(result["route"], "credit_card_statement")

    def test_node_answers_largest_transaction_and_taxes_from_memory(self) -> None:
        statement = build_statement()
        base_state = {
            "session_id": "demo",
            "memory": {
                "credit_card_statement": statement,
            },
            "pending_route": "",
            "route": "",
            "search_query": "",
            "documents": [],
            "context": "",
            "answer": "",
            "final_answer": "",
            "error": None,
            "media": {},
        }

        largest = credit_card_statement_node(
            {
                **base_state,
                "question": "Cual fue el consumo mas grande del resumen de tarjeta analizado previamente.",
                "original_question": "Cual fue el consumo mas grande del resumen de tarjeta analizado previamente.",
                "standalone_question": "Cual fue el consumo mas grande del resumen de tarjeta analizado previamente.",
                "is_followup": True,
            },
            llm=None,
        )
        self.assertIn("AMAZON", largest["answer"])
        self.assertIn("USD 245,61", largest["answer"])

        taxes = credit_card_statement_node(
            {
                **base_state,
                "question": "Cuanto me cobraron de impuestos en el resumen de tarjeta analizado previamente.",
                "original_question": "Cuanto me cobraron de impuestos en el resumen de tarjeta analizado previamente.",
                "standalone_question": "Cuanto me cobraron de impuestos en el resumen de tarjeta analizado previamente.",
                "is_followup": True,
            },
            llm=None,
        )
        self.assertIn("288.284,35", taxes["answer"])


if __name__ == "__main__":
    unittest.main()
