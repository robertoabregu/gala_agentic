from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.benefits_location_api import (
    _clear_location_api_caches,
    get_local_promotions_detail,
    get_nearby_locales,
)


class MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class BenefitsLocationApiCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_location_api_caches()

    @patch("tools.benefits_location_api.requests.get")
    def test_nearby_locales_uses_short_cache(self, requests_get_mock) -> None:
        requests_get_mock.return_value = MockResponse(
            [
                {
                    "id": 10,
                    "nombre": "DIA",
                    "categoriaMarca": "Supermercados",
                    "localidad": "San Fernando",
                    "provincia": "Buenos Aires",
                    "distancia": 0.2,
                    "latitud": -34.44,
                    "longitud": -58.54,
                    "clienteLatitud": -34.45,
                    "clienteLongitud": -58.55,
                    "emoji": "cart",
                }
            ]
        )

        first_result = get_nearby_locales(-34.45, -58.55)
        second_result = get_nearby_locales(-34.45, -58.55)

        self.assertEqual(requests_get_mock.call_count, 1)
        self.assertEqual(len(first_result), 1)
        self.assertEqual(first_result, second_result)

    @patch("tools.benefits_location_api.requests.get")
    def test_local_detail_uses_short_cache(self, requests_get_mock) -> None:
        requests_get_mock.return_value = MockResponse(
            {
                "idLocal": 99,
                "nombreMarca": "Dexter",
                "calle": "Constitucion",
                "numero": 729,
                "localidadNombre": "San Fernando",
                "provinciaNombre": "Buenos Aires",
                "promociones": [
                    {
                        "id": 1,
                        "porcentajeAhorro": 20,
                        "tipoTope": "Cliente",
                        "topeReintegro": 10000,
                        "leyendaDiasAplicacion": "Viernes",
                    }
                ],
            }
        )

        first_result = get_local_promotions_detail(99)
        second_result = get_local_promotions_detail(99)

        self.assertEqual(requests_get_mock.call_count, 1)
        self.assertEqual(first_result["local_id"], 99)
        self.assertEqual(first_result, second_result)


if __name__ == "__main__":
    unittest.main()
