from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.benefits import benefits_node
from agents.router import router_node
from services.benefits_ranker import rank_locales


class RouterFallbackLLM:
    def invoke(self, *_args, **_kwargs):
        class Response:
            content = "chitchat"

        return Response()


def _base_state(question: str, **overrides):
    state = {
        "session_id": "demo",
        "memory": {},
        "pending_route": "",
        "question": question,
        "original_question": question,
        "standalone_question": question,
        "is_followup": False,
        "route": "",
        "search_query": "",
        "documents": [],
        "context": "",
        "answer": "",
        "final_answer": "",
        "error": None,
        "missing_fields": [],
        "user_location": {},
        "media": {},
    }
    state.update(overrides)
    return state


class BenefitsFlowTests(unittest.TestCase):
    def test_benefits_without_location_requests_whatsapp_location(self) -> None:
        result = benefits_node(_base_state("beneficios cerca"))

        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["pending_route"], "benefits")
        self.assertEqual(result["missing_fields"], ["user_location"])
        self.assertIn("WhatsApp", result["answer"])

    def test_router_routes_back_to_benefits_when_location_arrives(self) -> None:
        result = router_node(
            _base_state(
                "Ubicacion compartida por WhatsApp",
                pending_route="benefits",
                memory={"pending_route": "benefits"},
                user_location={"latitude": "-34.5", "longitude": "-58.4"},
            ),
            llm=RouterFallbackLLM(),
        )

        self.assertEqual(result["route"], "benefits")
        self.assertEqual(result["pending_route"], "")

    def test_rank_locales_prioritizes_supermarkets(self) -> None:
        ranked = rank_locales(
            [
                {
                    "local_id": 1,
                    "brand": "Burger Place",
                    "category": "Gastronomía",
                    "distance_km": 0.1,
                },
                {
                    "local_id": 2,
                    "brand": "Jumbo",
                    "category": "Supermercados",
                    "distance_km": 0.8,
                },
            ],
            query="supermercados cerca",
        )

        self.assertEqual(ranked[0]["category"], "Supermercados")

    def test_rank_locales_prioritizes_sports_brands_for_zapatillas(self) -> None:
        ranked = rank_locales(
            [
                {
                    "local_id": 1,
                    "brand": "Farmacity",
                    "category": "Salud y Bienestar",
                    "distance_km": 0.1,
                },
                {
                    "local_id": 2,
                    "brand": "Nike",
                    "category": "Indumentaria",
                    "distance_km": 0.7,
                },
                {
                    "local_id": 3,
                    "brand": "Zara",
                    "category": "Indumentaria",
                    "distance_km": 0.2,
                },
            ],
            query="zapatillas para mi papá",
        )

        self.assertEqual(ranked[0]["category"], "Indumentaria")
        self.assertEqual(ranked[0]["brand"], "Nike")

    @patch("agents.benefits.rank_locales")
    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_flow_fetches_detail_only_for_top_candidates(
        self,
        nearby_mock,
        detail_mock,
        rank_mock,
    ) -> None:
        locals_payload = [
            {
                "local_id": index,
                "brand": f"Local {index}",
                "category": "Supermercados",
                "distance_km": float(index),
                "city": "San Fernando",
                "province": "Buenos Aires",
            }
            for index in range(1, 13)
        ]
        nearby_mock.return_value = locals_payload
        rank_mock.return_value = locals_payload
        detail_mock.side_effect = [
            {
                "local_id": index,
                "brand": f"Local {index}",
                "address": f"Calle {index}, San Fernando",
                "city": "San Fernando",
                "province": "Buenos Aires",
                "promotions": [],
            }
            for index in range(1, 9)
        ]

        benefits_node(
            _base_state(
                "supermercados cerca",
                route="benefits",
                user_location={"latitude": "-34.45", "longitude": "-58.55"},
            )
        )

        self.assertEqual(detail_mock.call_count, 8)

    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_eminent_only_promo_is_clarified(self, nearby_mock, detail_mock) -> None:
        nearby_mock.return_value = [
            {
                "local_id": 10,
                "brand": "DIA",
                "category": "Supermercados",
                "distance_km": 0.1,
                "city": "San Fernando",
                "province": "Buenos Aires",
            }
        ]
        detail_mock.return_value = {
            "local_id": 10,
            "brand": "DIA",
            "address": "Av. Peron 2201, San Fernando",
            "city": "San Fernando",
            "province": "Buenos Aires",
            "promotions": [
                {
                    "discount_percent": 20,
                    "cashback_cap": 10000,
                    "days": "Viernes",
                    "is_eminent": True,
                    "attention_model": "Eminent",
                    "payment_summary": "Tarjetas Galicia",
                }
            ],
        }

        result = benefits_node(
            _base_state(
                "beneficios cerca",
                route="benefits",
                user_location={"latitude": "-34.45", "longitude": "-58.55"},
            )
        )

        self.assertIn("Aplica para clientes Eminent", result["answer"])

    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_response_does_not_invent_discount(self, nearby_mock, detail_mock) -> None:
        nearby_mock.return_value = [
            {
                "local_id": 20,
                "brand": "Jumbo",
                "category": "Supermercados",
                "distance_km": 0.2,
                "city": "San Fernando",
                "province": "Buenos Aires",
            }
        ]
        detail_mock.return_value = {
            "local_id": 20,
            "brand": "Jumbo",
            "address": "Av. Libertador 1234, San Fernando",
            "city": "San Fernando",
            "province": "Buenos Aires",
            "promotions": [
                {
                    "discount_percent": None,
                    "cashback_cap": 15000,
                    "days": "Domingo",
                    "is_eminent": False,
                    "payment_summary": "Tarjetas Galicia",
                }
            ],
        }

        result = benefits_node(
            _base_state(
                "beneficios cerca",
                route="benefits",
                user_location={"latitude": "-34.45", "longitude": "-58.55"},
            )
        )

        self.assertNotIn("% de ahorro", result["answer"])

    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_detail_failure_continues_with_other_locales(self, nearby_mock, detail_mock) -> None:
        nearby_mock.return_value = [
            {
                "local_id": 30,
                "brand": "Local roto",
                "category": "Supermercados",
                "distance_km": 0.1,
                "city": "San Fernando",
                "province": "Buenos Aires",
            },
            {
                "local_id": 31,
                "brand": "DIA",
                "category": "Supermercados",
                "distance_km": 0.2,
                "city": "San Fernando",
                "province": "Buenos Aires",
            },
        ]
        detail_mock.side_effect = [
            RuntimeError("boom"),
            {
                "local_id": 31,
                "brand": "DIA",
                "address": "Sarmiento 1, Tigre",
                "city": "Tigre",
                "province": "Buenos Aires",
                "promotions": [
                    {
                        "discount_percent": 20,
                        "cashback_cap": 10000,
                        "days": "Domingo",
                        "is_eminent": False,
                        "payment_summary": "Tarjetas Galicia",
                    }
                ],
            },
        ]

        result = benefits_node(
            _base_state(
                "beneficios cerca",
                route="benefits",
                user_location={"latitude": "-34.45", "longitude": "-58.55"},
            )
        )

        self.assertIn("*DIA*", result["answer"])
        self.assertNotIn("No pude consultar los beneficios", result["answer"])

    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_response_includes_detail_address(self, nearby_mock, detail_mock) -> None:
        nearby_mock.return_value = [
            {
                "local_id": 40,
                "brand": "Mcdonalds",
                "category": "Gastronomía",
                "distance_km": 4.0,
                "city": "Tigre",
                "province": "Buenos Aires",
            }
        ]
        detail_mock.return_value = {
            "local_id": 40,
            "brand": "Mcdonalds",
            "address": "Sarmiento 1, Tigre",
            "city": "Tigre",
            "province": "Buenos Aires",
            "promotions": [
                {
                    "discount_percent": 20,
                    "cashback_cap": 10000,
                    "days": "Domingo",
                    "is_eminent": False,
                    "payment_summary": "Tarjetas Galicia",
                }
            ],
        }

        result = benefits_node(
            _base_state(
                "beneficios cerca",
                route="benefits",
                user_location={"latitude": "-34.45", "longitude": "-58.55"},
            )
        )

        self.assertIn("Sarmiento 1, Tigre", result["answer"])

    def test_router_regression_for_other_routes(self) -> None:
        cases = [
            ("Quiero saber sobre préstamos personales", "loans_rag"),
            ("Necesito ver mi situación crediticia", "bcra_credit_status"),
            ("Necesito una sucursal cercana", "branch_locator"),
            ("Quiero analizar mi resumen de tarjeta", "credit_card_statement"),
            ("hola", "chitchat"),
        ]

        for question, expected_route in cases:
            with self.subTest(question=question):
                result = router_node(_base_state(question), llm=RouterFallbackLLM())
                self.assertEqual(result["route"], expected_route)


if __name__ == "__main__":
    unittest.main()
