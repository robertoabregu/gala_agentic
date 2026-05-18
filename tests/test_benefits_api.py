from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.benefits import benefits_node
from tools import benefits_api


SAMPLE_CATEGORIES_PAYLOAD = {
    "data": {
        "list": [
            {"descripcion": "Indumentaria"},
            {"descripcion": "Gastronomía"},
        ]
    },
    "errors": None,
}

SAMPLE_CATALOG_PAYLOAD = {
    "data": {
        "list": [
            {
                "id": 101,
                "titulo": "Paruolo",
                "promocion": "20% de ahorro y hasta 3 cuotas sin interés",
                "subtitulo": "Indumentaria",
                "leyendaDiasAplicacion": "Viernes",
                "adicional": "",
                "eminent": False,
                "pagoQR": False,
                "contactLess": False,
                "pagoNFC": False,
                "proximamente": False,
                "fechaHasta": "2026-12-31",
                "tipoPromocion": "Marca",
                "mediosDePago": [{"tipoTarjeta": "credito"}],
                "modeloAtencion": {},
            },
            {
                "id": 102,
                "titulo": "Viamo",
                "promocion": "25% de ahorro y hasta 3 cuotas sin interés",
                "subtitulo": "Indumentaria",
                "leyendaDiasAplicacion": "Viernes",
                "adicional": "",
                "eminent": True,
                "pagoQR": False,
                "contactLess": False,
                "pagoNFC": False,
                "proximamente": False,
                "fechaHasta": "2026-12-31",
                "tipoPromocion": "Marca",
                "mediosDePago": [{"tipoTarjeta": "credito"}],
                "modeloAtencion": {"nombre": "Eminent Black", "exclusivo": True},
            },
        ]
    },
    "errors": None,
}

SAMPLE_EMPTY_CAROUSEL_PAYLOAD = {
    "data": {
        "idCarrusel": 152,
        "titulo": "Ofertas Galicia",
        "promociones": {
            "list": [],
            "totalSize": 0,
        },
    },
    "errors": None,
}


def fake_fetch_json(path: str, *, params: dict) -> dict:
    if path == "categorias":
        return SAMPLE_CATEGORIES_PAYLOAD
    if path == "promociones/catalogo":
        return SAMPLE_CATALOG_PAYLOAD
    if path == "promociones/list/carrusel/152":
        return SAMPLE_EMPTY_CAROUSEL_PAYLOAD
    raise AssertionError(f"Path inesperado en test: {path!r} con params={params!r}")


class BenefitsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        benefits_api._clear_benefits_cache()

    def tearDown(self) -> None:
        benefits_api._clear_benefits_cache()

    @patch("tools.benefits_api._fetch_json", side_effect=fake_fetch_json)
    def test_fetch_live_promotions_uses_catalog_when_available(self, _mock_fetch_json) -> None:
        promotions = benefits_api._fetch_live_promotions()

        self.assertEqual(len(promotions), 2)
        self.assertEqual(promotions[0]["comercio"], "Paruolo")
        self.assertEqual(promotions[0]["categoria"], "Indumentaria")
        self.assertFalse(promotions[0]["exclusivoEminent"])
        self.assertTrue(promotions[1]["exclusivoEminent"])

    @patch("tools.benefits_api._fetch_json", side_effect=fake_fetch_json)
    def test_benefits_node_returns_indumentaria_results_for_zapatillas_query(
        self,
        _mock_fetch_json,
    ) -> None:
        result = benefits_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "che tengo ganas de comprarme unas zapatillas nuevas, hay promos?",
                "original_question": "che tengo ganas de comprarme unas zapatillas nuevas, hay promos?",
                "standalone_question": "Che, tengo ganas de comprarme unas zapatillas nuevas, ¿hay promos?",
                "is_followup": False,
                "route": "",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
            }
        )

        self.assertEqual(result["route"], "benefits")
        self.assertEqual(result["tool_input"]["category"], "Indumentaria")
        self.assertEqual(result["tool_output"]["results_count"], 2)
        self.assertIn("Paruolo", result["answer"])
        self.assertNotIn("No encontré", result["answer"])
