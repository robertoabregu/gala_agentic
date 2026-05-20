from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from agents.benefits import benefits_node
from services.benefits_intelligence import extract_benefits_intent
from services.benefits_ranker import rank_enriched_locales, select_primary_promotion
from services.commercial_calendar import get_active_commercial_context


class FakeBenefitsIntentLLM:
    def invoke(self, messages, *_args, **_kwargs):
        user_payload = messages[-1][1]
        if "campera" in user_payload:
            content = """
            {
              "intent": "benefits_planning",
              "needs_location": true,
              "commercial_intent": "gift_planning",
              "product_interest": "campera",
              "recipient": "padre",
              "occasion": "dia_del_padre",
              "category_candidates": ["Indumentaria"],
              "store_type_hints": ["ropa deportiva"],
              "brand_or_store_hints": ["Dexter", "Nike"],
              "sort_strategy": "relevance_promotion_distance",
              "audience_hint": "general"
            }
            """
        else:
            content = "{}"

        class Response:
            def __init__(self, text):
                self.content = text

        return Response(content)


def _base_state(question: str, **overrides):
    state = {
        "session_id": "demo",
        "memory": {},
        "pending_route": "",
        "question": question,
        "original_question": question,
        "standalone_question": question,
        "is_followup": False,
        "route": "benefits",
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


class BenefitsIntelligenceTests(unittest.TestCase):
    def test_extract_intent_for_zapatillas_para_mi_papa(self) -> None:
        result = extract_benefits_intent("zapatillas para mi papá", reference_date=date(2026, 6, 10))

        self.assertEqual(result["product_interest"], "zapatillas")
        self.assertEqual(result["recipient"], "padre")
        self.assertIn("Indumentaria", result["category_candidates"])

    def test_extract_intent_for_supermercados_cerca(self) -> None:
        result = extract_benefits_intent("supermercados cerca")

        self.assertIn("Supermercados", result["category_candidates"])

    def test_active_commercial_context_includes_response_hint_when_nearby(self) -> None:
        context = get_active_commercial_context(
            reference_date=date(2026, 6, 10),
            categories=["Indumentaria"],
        )

        self.assertIsNotNone(context["active_occasion"])
        self.assertEqual(context["active_occasion"]["id"], "dia_del_padre")
        self.assertIn("Día del Padre", context["active_occasion"]["response_hint"])

    def test_active_commercial_context_is_empty_when_not_nearby(self) -> None:
        context = get_active_commercial_context(
            reference_date=date(2026, 8, 5),
            categories=["VehÃ­culos"],
        )

        self.assertIsNone(context["active_occasion"])

    def test_llm_can_enrich_structured_intent(self) -> None:
        result = extract_benefits_intent(
            "quiero comprar una campera para mi papá",
            llm=FakeBenefitsIntentLLM(),
            reference_date=date(2026, 6, 10),
        )

        self.assertEqual(result["intent"], "benefits_planning")
        self.assertEqual(result["product_interest"], "campera")
        self.assertEqual(result["occasion"], "dia_del_padre")
        self.assertIn("Nike", result["brand_or_store_hints"])

    def test_ranker_prioritizes_massive_promotion_over_eminent_by_default(self) -> None:
        ranked = rank_enriched_locales(
            [
                {
                    "local_id": 1,
                    "brand": "Marca Eminent",
                    "category": "Indumentaria",
                    "distance_km": 0.1,
                    "promotions": [
                        {
                            "discount_percent": 30,
                            "cashback_cap": 20000,
                            "is_eminent": True,
                            "valid_from": "01/06/2026",
                            "valid_to": "30/06/2026",
                        }
                    ],
                },
                {
                    "local_id": 2,
                    "brand": "Marca Masiva",
                    "category": "Indumentaria",
                    "distance_km": 0.2,
                    "promotions": [
                        {
                            "discount_percent": 20,
                            "cashback_cap": 10000,
                            "is_eminent": False,
                            "valid_from": "01/06/2026",
                            "valid_to": "30/06/2026",
                        }
                    ],
                },
            ],
            query="ropa cerca",
            intent_context={"audience_hint": "general", "category_candidates": ["Indumentaria"]},
            max_results=2,
            reference_date=date(2026, 6, 10),
        )

        self.assertEqual(ranked[0]["brand"], "Marca Masiva")

    def test_select_primary_promotion_prefers_massive_and_flags_additional_eminent(self) -> None:
        local = {
            "brand": "Dexter",
            "promotions": [
                {
                    "discount_percent": 20,
                    "cashback_cap": 10000,
                    "is_eminent": False,
                    "valid_from": "01/06/2026",
                    "valid_to": "30/06/2026",
                },
                {
                    "discount_percent": 30,
                    "cashback_cap": 15000,
                    "is_eminent": True,
                    "valid_from": "01/06/2026",
                    "valid_to": "30/06/2026",
                },
            ],
        }

        selected = select_primary_promotion(
            local,
            audience_hint="general",
            reference_date=date(2026, 6, 10),
        )

        self.assertIsNotNone(selected)
        self.assertFalse(selected["is_eminent"])

    @patch("agents.benefits.get_local_promotions_detail")
    @patch("agents.benefits.get_nearby_locales")
    def test_response_uses_active_occasion_hint_when_available(self, nearby_mock, detail_mock) -> None:
        nearby_mock.return_value = [
            {
                "local_id": 1,
                "brand": "Dexter",
                "category": "Indumentaria",
                "distance_km": 1.5,
                "city": "San Fernando",
                "province": "Buenos Aires",
            }
        ]
        detail_mock.return_value = {
            "local_id": 1,
            "brand": "Dexter",
            "address": "Av. Perón 2201, San Fernando",
            "city": "San Fernando",
            "province": "Buenos Aires",
            "promotions": [
                {
                    "discount_percent": 20,
                    "cashback_cap": 10000,
                    "days": "Viernes y sábado",
                    "payment_summary": "Tarjetas Galicia",
                    "is_eminent": False,
                    "valid_from": "01/06/2026",
                    "valid_to": "30/06/2026",
                }
            ],
        }

        with patch(
            "agents.benefits.extract_benefits_intent",
            return_value={
                "intent": "benefits_planning",
                "commercial_intent": "gift_planning",
                "product_interest": "zapatillas",
                "recipient": "padre",
                "occasion": "dia_del_padre",
                "category_candidates": ["Indumentaria"],
                "store_type_hints": ["calzado", "ropa deportiva"],
                "brand_or_store_hints": ["Dexter", "Moov"],
                "sort_strategy": "relevance_promotion_distance",
                "audience_hint": "general",
                "active_occasion": {
                    "id": "dia_del_padre",
                    "name": "Día del Padre",
                    "days_until": 11,
                    "related_categories": ["Indumentaria"],
                    "response_hint": "Justo se viene el Día del Padre, así que te dejo opciones que pueden servir para regalar.",
                },
            },
        ):
            result = benefits_node(
                _base_state(
                    "Quiero comprarle unas zapatillas a mi papá",
                    user_location={"latitude": "-34.45", "longitude": "-58.55"},
                )
            )

        self.assertIn("Día del Padre", result["answer"])
        self.assertIn("Av. Perón 2201, San Fernando", result["answer"])


if __name__ == "__main__":
    unittest.main()
