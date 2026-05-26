from __future__ import annotations

import importlib
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


def _reset_app_registry(whatsapp_app) -> None:
    with whatsapp_app._inbound_registry_lock:
        whatsapp_app._inbound_registry.clear()


class WhatsAppAsyncTests(unittest.TestCase):
    def test_benefits_request_with_saved_location_returns_ack_and_schedules_async_job(self) -> None:
        whatsapp_app = _load_app_module()
        _reset_app_registry(whatsapp_app)
        client = whatsapp_app.app.test_client()

        memory = {
            "pending_route": "",
            "pending_query": "",
            "missing_fields": [],
            "last_route": "benefits",
            "last_user_question": "",
            "last_assistant_answer": "",
            "last_topic": "beneficios",
            "updated_at": "",
            "user_location": {"latitude": "-34.45", "longitude": "-58.55"},
            "credit_card_statement": {},
        }

        with patch.object(
            whatsapp_app,
            "get_or_create_conversation_session",
            return_value="whatsapp-5491112345678-20260525-190502-a8f3",
        ):
            with patch.object(whatsapp_app, "load_memory", return_value=memory):
                with patch.object(whatsapp_app, "_submit_async_whatsapp_job") as submit_mock:
                    with patch.object(whatsapp_app, "run_bot_query") as run_bot_query_mock:
                        response = client.post(
                            "/whatsapp",
                            data={
                                "Body": "supermercados cerca",
                                "From": "whatsapp:+5491112345678",
                                "To": "whatsapp:+14155238886",
                                "MessageSid": "SM123",
                            },
                        )

        self.assertEqual(response.status_code, 200)
        response_text = response.data.decode("utf-8")
        self.assertIn(whatsapp_app._ack_message(), response_text)
        submit_mock.assert_called_once()
        submit_kwargs = submit_mock.call_args.kwargs
        self.assertEqual(submit_kwargs["user_id"], "whatsapp-5491112345678")
        self.assertEqual(
            submit_kwargs["session_id"],
            "whatsapp-5491112345678-20260525-190502-a8f3",
        )
        run_bot_query_mock.assert_not_called()

    def test_duplicate_async_message_sid_returns_empty_twiml(self) -> None:
        whatsapp_app = _load_app_module()
        _reset_app_registry(whatsapp_app)
        client = whatsapp_app.app.test_client()

        memory = {
            "pending_route": "",
            "pending_query": "",
            "missing_fields": [],
            "last_route": "benefits",
            "last_user_question": "",
            "last_assistant_answer": "",
            "last_topic": "beneficios",
            "updated_at": "",
            "user_location": {"latitude": "-34.45", "longitude": "-58.55"},
            "credit_card_statement": {},
        }

        with patch.object(
            whatsapp_app,
            "get_or_create_conversation_session",
            return_value="whatsapp-5491112345678-20260525-190502-a8f3",
        ):
            with patch.object(whatsapp_app, "load_memory", return_value=memory):
                with patch.object(whatsapp_app, "_submit_async_whatsapp_job") as submit_mock:
                    first_response = client.post(
                        "/whatsapp",
                        data={
                            "Body": "supermercados cerca",
                            "From": "whatsapp:+5491112345678",
                            "To": "whatsapp:+14155238886",
                            "MessageSid": "SM999",
                        },
                    )
                    second_response = client.post(
                        "/whatsapp",
                        data={
                            "Body": "supermercados cerca",
                            "From": "whatsapp:+5491112345678",
                            "To": "whatsapp:+14155238886",
                            "MessageSid": "SM999",
                        },
                    )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertIn(
            whatsapp_app._ack_message(),
            first_response.data.decode("utf-8"),
        )
        self.assertNotIn(
            whatsapp_app._ack_message(),
            second_response.data.decode("utf-8"),
        )
        self.assertEqual(submit_mock.call_count, 1)

    def test_background_worker_sends_final_whatsapp_message(self) -> None:
        whatsapp_app = _load_app_module()
        _reset_app_registry(whatsapp_app)

        with patch.object(whatsapp_app, "get_runtime", return_value=object()):
            with patch.object(
                whatsapp_app,
                "run_bot_query",
                return_value={
                    "final_answer": "*DIA* — Av. Perón 2201, San Fernando",
                    "topic": "beneficios",
                    "route": "benefits",
                },
            ):
                with patch.object(whatsapp_app, "send_whatsapp_message") as send_mock:
                    whatsapp_app._process_whatsapp_request_async(
                        body="supermercados cerca",
                        sender="whatsapp:+5491112345678",
                        user_id="whatsapp-5491112345678",
                        session_id="whatsapp-5491112345678-20260525-190502-a8f3",
                        message_sid="SM555",
                        user_location={"latitude": "-34.45", "longitude": "-58.55"},
                        media={},
                    )

        send_mock.assert_called_once()
        kwargs = send_mock.call_args.kwargs
        self.assertEqual(kwargs["to_number"], "whatsapp:+5491112345678")
        self.assertIn("*DIA*", kwargs["body"])

    def test_sync_request_uses_conversation_session_and_stable_langfuse_user_id(self) -> None:
        whatsapp_app = _load_app_module()
        _reset_app_registry(whatsapp_app)
        client = whatsapp_app.app.test_client()

        with patch.object(
            whatsapp_app,
            "get_or_create_conversation_session",
            return_value="whatsapp-5491112345678-20260525-190502-a8f3",
        ):
            with patch.object(whatsapp_app, "load_memory", return_value={}):
                with patch.object(whatsapp_app, "send_whatsapp_typing_indicator"):
                    with patch.object(
                        whatsapp_app,
                        "run_bot_query",
                        return_value={
                            "final_answer": "Hola",
                            "topic": "general",
                            "route": "fallback",
                        },
                    ) as run_bot_query_mock:
                        response = client.post(
                            "/whatsapp",
                            data={
                                "Body": "hola",
                                "From": "whatsapp:+5491112345678",
                                "To": "whatsapp:+14155238886",
                                "MessageSid": "SM777",
                            },
                        )

        self.assertEqual(response.status_code, 200)
        run_kwargs = run_bot_query_mock.call_args.kwargs
        self.assertEqual(
            run_kwargs["session_id"],
            "whatsapp-5491112345678-20260525-190502-a8f3",
        )
        self.assertEqual(run_kwargs["langfuse_user_id"], "whatsapp-5491112345678")

    def test_request_still_works_when_session_store_fails(self) -> None:
        whatsapp_app = _load_app_module()
        _reset_app_registry(whatsapp_app)
        client = whatsapp_app.app.test_client()

        with patch.object(
            whatsapp_app,
            "get_or_create_conversation_session",
            side_effect=RuntimeError("store down"),
        ):
            with patch.object(whatsapp_app, "load_memory", return_value={}):
                with patch.object(whatsapp_app, "send_whatsapp_typing_indicator"):
                    with patch.object(
                        whatsapp_app,
                        "run_bot_query",
                        return_value={
                            "final_answer": "Hola",
                            "topic": "general",
                            "route": "fallback",
                        },
                    ) as run_bot_query_mock:
                        response = client.post(
                            "/whatsapp",
                            data={
                                "Body": "hola",
                                "From": "whatsapp:+5491112345678",
                                "To": "whatsapp:+14155238886",
                                "MessageSid": "SM778",
                            },
                        )

        self.assertEqual(response.status_code, 200)
        run_kwargs = run_bot_query_mock.call_args.kwargs
        self.assertEqual(run_kwargs["langfuse_user_id"], "whatsapp-5491112345678")
        self.assertTrue(
            run_kwargs["session_id"].startswith("whatsapp-5491112345678-"),
        )


if __name__ == "__main__":
    unittest.main()
