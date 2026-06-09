from __future__ import annotations

import unittest

from experiments.evaluation import evaluate_scores, extract_dataset_case


class ExperimentEvaluatorsTests(unittest.TestCase):
    def test_evaluate_scores_calculates_route_match_with_aliases(self) -> None:
        scores = evaluate_scores(
            item_input={
                "question": "Quiero consultar mi situacion crediticia",
                "expected_route": "bcra_agent",
            },
            output={
                "route": "bcra_credit_status",
                "topic": "situacion_crediticia_bcra",
                "answer": "Necesito tu DNI para consultarlo.",
                "used_tool": True,
            },
        )

        self.assertTrue(scores["route_match"])

    def test_evaluate_scores_calculates_topic_match(self) -> None:
        scores = evaluate_scores(
            item_input={
                "question": "Donde tengo una sucursal cerca?",
                "expected_topic": "sucursales",
            },
            output={
                "route": "branch_locator",
                "topic": "sucursales_cercanas",
                "answer": "Tengo una sucursal a 3 cuadras.",
            },
        )

        self.assertTrue(scores["topic_match"])

    def test_evaluate_scores_calculates_must_include_coverage(self) -> None:
        scores = evaluate_scores(
            item_input={
                "question": "Puedo solicitar un prendario si ya tengo un auto?",
                "must_include": ["prestamo prendario", "vehiculo", "garantia"],
            },
            output={
                "answer": "El prestamo prendario usa el vehiculo como respaldo.",
            },
        )

        self.assertAlmostEqual(scores["must_include_coverage"], 2 / 3, places=4)

    def test_evaluate_scores_detects_forbidden_terms(self) -> None:
        scores = evaluate_scores(
            item_input={
                "question": "Puedo pedir un prestamo?",
                "must_not_include": [
                    "No tengo informacion suficiente para resolver esa consulta bancaria desde aca"
                ],
            },
            output={
                "answer": (
                    "No tengo informacion suficiente para resolver esa consulta bancaria desde aca."
                ),
            },
        )

        self.assertFalse(scores["forbidden_terms_avoided"])

    def test_evaluate_scores_handles_missing_expected_route(self) -> None:
        scores = evaluate_scores(
            item_input={"question": "Hola"},
            output={"answer": "Hola, como estas?"},
        )

        self.assertIsNone(scores["route_match"])
        self.assertTrue(scores["answer_non_empty"])

    def test_evaluate_scores_supports_string_expected_output(self) -> None:
        scores = evaluate_scores(
            item_input={"question": "Hola, que podes hacer?"},
            expected_output="Explica las capacidades del bot sin comparar texto exacto.",
            output={
                "route": "chitchat",
                "topic": "conversacion",
                "answer": "Puedo ayudarte con prestamos, beneficios y sucursales.",
            },
        )

        self.assertTrue(scores["answer_non_empty"])
        self.assertTrue(scores["whatsapp_format_basic_ok"])

    def test_extract_dataset_case_reads_mixed_flow_expectations(self) -> None:
        benefits_case = extract_dataset_case(
            input_value={
                "question": "Que beneficios tengo en gastronomia?",
                "expected_route": "benefits",
            },
            expected_output={"expected_topic": "beneficios"},
            metadata={"dataset_name": "Gala Regression Cases"},
            item_id="item-benefits",
        )
        branch_case = extract_dataset_case(
            input_value={
                "question": "Donde tengo una sucursal cerca?",
                "expected_route": "branch_locator",
            },
            expected_output={"expected_topic": "sucursales"},
            metadata={"dataset_name": "Gala Regression Cases"},
            item_id="item-branch",
        )

        self.assertEqual(benefits_case["question"], "Que beneficios tengo en gastronomia?")
        self.assertEqual(benefits_case["expected_route"], "benefits")
        self.assertEqual(branch_case["expected_route"], "branch_locator")
        self.assertEqual(branch_case["expected_topic"], "sucursales")


if __name__ == "__main__":
    unittest.main()
