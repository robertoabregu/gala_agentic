from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.benefits import benefits_node
from tools import benefits_api


SAMPLE_CATEGORIES_PAYLOAD = {
    "data": {
        "list": [
            {"id": 8, "descripcion": "Supermercados"},
            {"id": 1, "descripcion": "Gastronomía"},
            {"id": 7, "descripcion": "Indumentaria"},
            {"id": 9, "descripcion": "Electrónica"},
        ]
    },
    "errors": None,
}

SAMPLE_BASE_CATALOG_PAYLOAD = {
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
        ],
        "totalSize": 2,
    },
    "errors": None,
}

SAMPLE_FRAVEGA_SEARCH_PAYLOAD = {
    "data": [
        {
            "ids": [211],
            "nombre": "Fravega",
            "idCategoria": 9,
            "navegacion": "marca/211|Fravega|Marca",
            "tipo": "Marca",
        }
    ],
    "errors": None,
}

SAMPLE_COMIDA_SEARCH_PAYLOAD = {
    "data": [
        {
            "ids": [101],
            "nombre": "Rappi",
            "idCategoria": 1,
            "navegacion": "marca/101|Rappi|Marca",
            "tipo": "Marca",
        },
        {
            "ids": [103118],
            "nombre": "The Foodbox",
            "idCategoria": 8,
            "navegacion": "marca/103118|The Foodbox|Marca",
            "tipo": "Marca",
        },
    ],
    "errors": None,
}

SAMPLE_GASTRONOMIA_SEARCH_PAYLOAD = {
    "data": [
        {
            "ids": [38],
            "nombre": "Promos Gastronomía",
            "idCategoria": 0,
            "navegacion": "uri=/promociones/categoria=1",
            "tipo": "Especial",
        }
    ],
    "errors": None,
}

SAMPLE_FRAVEGA_PROMOS_PAYLOAD = {
    "data": {
        "list": [
            {
                "id": 201,
                "titulo": "Fravega",
                "promocion": "Hasta 12 cuotas sin interés",
                "subtitulo": "Electrónica",
                "leyendaDiasAplicacion": "Todos los días",
                "adicional": "",
                "eminent": False,
                "pagoQR": False,
                "contactLess": False,
                "pagoNFC": False,
                "proximamente": False,
                "fechaHasta": "2026-12-31",
                "tipoPromocion": "Marca",
                "mediosDePago": [{"tipoTarjeta": "credito"}],
                "modeloAtencion": {"nombre": "Masivo", "exclusivo": False},
            },
            {
                "id": 202,
                "titulo": "Fravega",
                "promocion": "Hasta 18 cuotas sin interés",
                "subtitulo": "Electrónica",
                "leyendaDiasAplicacion": "Todos los días",
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
        ],
        "totalSize": 2,
    },
    "errors": None,
}

SAMPLE_RAPPI_PROMOS_PAYLOAD = {
    "data": {
        "list": [
            {
                "id": 301,
                "titulo": "Rappi",
                "promocion": "30% de ahorro",
                "subtitulo": "Gastronomía",
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
                "modeloAtencion": {"nombre": "Masivo", "exclusivo": False},
            }
        ],
        "totalSize": 1,
    },
    "errors": None,
}

SAMPLE_SUPERMERCADOS_PROMOS_PAYLOAD = {
    "data": {
        "list": [
            {
                "id": 401,
                "titulo": "¿Te toca ir al súper?",
                "promocion": "Aprovechá hasta 20% de ahorro",
                "subtitulo": "Supermercados",
                "leyendaDiasAplicacion": "",
                "adicional": "",
                "eminent": False,
                "pagoQR": False,
                "contactLess": False,
                "pagoNFC": False,
                "proximamente": False,
                "fechaHasta": "2026-12-31",
                "tipoPromocion": "Especial",
                "mediosDePago": [{"tipoTarjeta": "credito"}],
                "modeloAtencion": {"nombre": "todos", "exclusivo": False},
            }
        ],
        "totalSize": 1,
    },
    "errors": None,
}

SAMPLE_EMPTY_CATEGORY_PAYLOAD = {
    "data": {
        "list": [],
        "totalSize": 0,
    },
    "errors": None,
}


def _normalize_params(params: object) -> tuple[tuple[str, object], ...]:
    if isinstance(params, dict):
        return tuple(sorted(params.items()))
    if isinstance(params, list):
        return tuple(params)
    if isinstance(params, tuple):
        return params
    return ()


def fake_fetch_json(path: str, *, params: object) -> dict:
    normalized_params = _normalize_params(params)

    if path == "categorias":
        return SAMPLE_CATEGORIES_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == ():
        return SAMPLE_BASE_CATALOG_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == (
        ("IdsMarca", 211),
        ("TipoPromocion", "Marca"),
    ):
        return SAMPLE_FRAVEGA_PROMOS_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == (
        ("IdsMarca", 101),
        ("TipoPromocion", "Marca"),
    ):
        return SAMPLE_RAPPI_PROMOS_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == (
        ("IdCategoria", 8),
        ("TipoPromocion", "categoria"),
    ):
        return SAMPLE_SUPERMERCADOS_PROMOS_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == (
        ("IdCategoria", 1),
        ("TipoPromocion", "categoria"),
    ):
        return SAMPLE_EMPTY_CATEGORY_PAYLOAD

    if path == "promociones/catalogo" and normalized_params == (
        ("IdCategoria", 9),
        ("TipoPromocion", "categoria"),
    ):
        return SAMPLE_EMPTY_CATEGORY_PAYLOAD

    if path == "buscador/search/fravega":
        return SAMPLE_FRAVEGA_SEARCH_PAYLOAD

    if path == "buscador/search/gastronomia":
        return SAMPLE_GASTRONOMIA_SEARCH_PAYLOAD

    if path == "buscador/search/comida":
        return SAMPLE_COMIDA_SEARCH_PAYLOAD

    if path == "buscador/search/comer":
        return {"data": [], "errors": None}

    if path == "buscador/search/delivery":
        return {"data": [], "errors": None}

    if path == "buscador/search/super":
        return {"data": [], "errors": None}

    if path == "buscador/search/supermercados":
        return {"data": [], "errors": None}

    if path == "buscador/search/electronica":
        return {"data": [], "errors": None}

    if path == "buscador/search/electrodomesticos":
        return {"data": [], "errors": None}

    if path.startswith("buscador/search/"):
        return {"data": [], "errors": None}

    raise AssertionError(f"Path inesperado en test: {path!r} con params={params!r}")


class BenefitsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        benefits_api._clear_benefits_cache()

    def tearDown(self) -> None:
        benefits_api._clear_benefits_cache()

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
                "standalone_question": "Che, tengo ganas de comprarme unas zapatillas nuevas, hay promos?",
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

    @patch("tools.benefits_api._fetch_json", side_effect=fake_fetch_json)
    def test_benefits_node_finds_fravega_as_brand_search(
        self,
        _mock_fetch_json,
    ) -> None:
        result = benefits_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "hay promos para fravega?",
                "original_question": "hay promos para fravega?",
                "standalone_question": "hay promos para fravega?",
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
        self.assertEqual(result["tool_output"]["results_count"], 2)
        self.assertIn("Fravega", result["answer"])
        self.assertNotIn("No encontré", result["answer"])

    @patch("tools.benefits_api._fetch_json", side_effect=fake_fetch_json)
    def test_benefits_node_uses_search_fallback_for_comida_query(
        self,
        _mock_fetch_json,
    ) -> None:
        result = benefits_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "tenés promos para pedir comida?",
                "original_question": "tenés promos para pedir comida?",
                "standalone_question": "tenés promos para pedir comida?",
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
        self.assertEqual(result["tool_input"]["category"], "Gastronomía")
        self.assertEqual(result["tool_output"]["results_count"], 1)
        self.assertIn("Rappi", result["answer"])
        self.assertNotIn("No encontré", result["answer"])

    @patch("tools.benefits_api._fetch_json", side_effect=fake_fetch_json)
    def test_benefits_node_uses_category_fallback_for_supermercados_query(
        self,
        _mock_fetch_json,
    ) -> None:
        result = benefits_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "y tenés beneficios para supermercados?",
                "original_question": "y tenés beneficios para supermercados?",
                "standalone_question": "y tenés beneficios para supermercados?",
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
        self.assertEqual(result["tool_input"]["category"], "Supermercados")
        self.assertEqual(result["tool_output"]["results_count"], 1)
        self.assertIn("súper", result["answer"])
