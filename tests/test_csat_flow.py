from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


def _load_app_module():
    fake_bot_runner = types.ModuleType("core.bot_runner")
    fake_bot_runner.BotRuntime = object
    fake_bot_runner.prepare_runtime = lambda *args, **kwargs: object()
    fake_bot_runner.run_bot_query = lambda *args, **kwargs: {}

    with patch.dict(sys.modules, {"core.bot_runner": fake_bot_runner}):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


class CsatFlowTests(unittest.TestCase):
    def test_defaults_to_empty_content_variables(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            whatsapp_app = _load_app_module()

        self.assertEqual(whatsapp_app._build_csat_flow_content_variables(), {})

    def test_allows_empty_content_variables_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"CSAT_FLOW_CONTENT_VARIABLES_JSON": "{}"},
            clear=False,
        ):
            whatsapp_app = _load_app_module()

        self.assertEqual(whatsapp_app._build_csat_flow_content_variables(), {})

    def test_replaces_flow_token_placeholder_from_env_json(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CSAT_FLOW_CONTENT_VARIABLES_JSON": (
                    '{"2":"__FLOW_TOKEN__","5":"encuesta_csat"}'
                )
            },
            clear=False,
        ):
            whatsapp_app = _load_app_module()

        content_variables = whatsapp_app._build_csat_flow_content_variables()

        self.assertIn("2", content_variables)
        self.assertIn("5", content_variables)
        self.assertNotEqual(
            content_variables["2"],
            whatsapp_app.CSAT_FLOW_TOKEN_PLACEHOLDER,
        )
        self.assertEqual(content_variables["5"], "encuesta_csat")
