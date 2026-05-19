from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

from agents.goodbye import CSAT_FLOW_TEMPLATE_SID, GOODBYE_ANSWER, goodbye_node
from agents.router import router_node


class RouterFallbackLLM:
    def invoke(self, *_args, **_kwargs):
        class Response:
            content = "chitchat"

        return Response()


def _load_app_module():
    fake_bot_runner = types.ModuleType("core.bot_runner")
    fake_bot_runner.BotRuntime = object
    fake_bot_runner.prepare_runtime = lambda *args, **kwargs: object()
    fake_bot_runner.run_bot_query = lambda *args, **kwargs: {}

    with patch.dict(sys.modules, {"core.bot_runner": fake_bot_runner}):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


class GoodbyeTests(unittest.TestCase):
    def test_router_routes_goodbye_for_chau(self) -> None:
        result = router_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "chau",
                "original_question": "chau",
                "standalone_question": "chau",
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
            llm=RouterFallbackLLM(),
        )

        self.assertEqual(result["route"], "goodbye")

    def test_router_routes_goodbye_for_chau_gracias(self) -> None:
        result = router_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "Chau gracias",
                "original_question": "Chau gracias",
                "standalone_question": "Chau gracias",
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
            llm=RouterFallbackLLM(),
        )

        self.assertEqual(result["route"], "goodbye")

    def test_router_prioritizes_goodbye_over_pending_route(self) -> None:
        result = router_node(
            {
                "session_id": "demo",
                "memory": {"pending_route": "credit_card_statement"},
                "pending_route": "credit_card_statement",
                "question": "chau",
                "original_question": "chau",
                "standalone_question": "chau",
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
            llm=RouterFallbackLLM(),
        )

        self.assertEqual(result["route"], "goodbye")

    def test_router_does_not_route_goodbye_for_technical_mention(self) -> None:
        result = router_node(
            {
                "session_id": "demo",
                "memory": {},
                "pending_route": "",
                "question": "qué significa chau?",
                "original_question": "qué significa chau?",
                "standalone_question": "qué significa chau?",
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
            llm=RouterFallbackLLM(),
        )

        self.assertNotEqual(result["route"], "goodbye")

    def test_goodbye_node_sets_csat_on_first_goodbye(self) -> None:
        result = goodbye_node(
            {
                "session_id": "demo",
                "memory": {"csat_sent": False},
                "pending_route": "",
                "question": "chau",
                "original_question": "chau",
                "standalone_question": "chau",
                "is_followup": False,
                "route": "goodbye",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
            }
        )

        self.assertEqual(result["answer"], GOODBYE_ANSWER)
        self.assertTrue(result["send_csat"])
        self.assertEqual(result["csat_template_sid"], CSAT_FLOW_TEMPLATE_SID)

    def test_goodbye_node_does_not_resend_csat_if_already_sent(self) -> None:
        result = goodbye_node(
            {
                "session_id": "demo",
                "memory": {"csat_sent": True},
                "pending_route": "",
                "question": "chau",
                "original_question": "chau",
                "standalone_question": "chau",
                "is_followup": False,
                "route": "goodbye",
                "search_query": "",
                "documents": [],
                "context": "",
                "answer": "",
                "final_answer": "",
                "error": None,
            }
        )

        self.assertFalse(result["send_csat"])
        self.assertEqual(result["answer"], "\u00a1Gracias por escribirme! \U0001F60A")

    def test_app_still_replies_if_template_send_fails(self) -> None:
        whatsapp_app = _load_app_module()
        client = whatsapp_app.app.test_client()
        with patch.object(whatsapp_app, "get_runtime", return_value=object()):
            with patch.object(
                whatsapp_app,
                "run_bot_query",
                return_value={
                    "final_answer": GOODBYE_ANSWER,
                    "topic": "despedida",
                    "route": "goodbye",
                    "send_csat": True,
                    "csat_template_sid": CSAT_FLOW_TEMPLATE_SID,
                },
            ):
                with patch.object(whatsapp_app, "send_whatsapp_typing_indicator"):
                    with patch.object(
                        whatsapp_app,
                        "send_whatsapp_content_template",
                        side_effect=RuntimeError("boom"),
                    ):
                        with patch.object(
                            whatsapp_app,
                            "load_memory",
                            return_value={"csat_sent": False},
                        ):
                            response = client.post(
                                "/whatsapp",
                                data={
                                    "Body": "chau",
                                    "From": "whatsapp:+5491112345678",
                                    "To": "whatsapp:+14155238886",
                                    "MessageSid": "SM123",
                                },
                            )

        self.assertEqual(response.status_code, 200)
        response_text = response.data.decode("utf-8")
        self.assertIn(GOODBYE_ANSWER, response_text)
