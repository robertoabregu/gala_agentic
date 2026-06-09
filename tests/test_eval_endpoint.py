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


class EvaluateEndpointTests(unittest.TestCase):
    def test_evaluate_returns_200_with_clean_json(self) -> None:
        evaluation_app = _load_app_module()
        client = evaluation_app.app.test_client()
        captured_kwargs: dict[str, object] = {}

        def fake_run_bot_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            trace_context = kwargs.get("trace_context")
            if isinstance(trace_context, dict):
                trace_context["trace_id"] = "trace-123"
                trace_context["total_latency_ms"] = 123
                trace_context["initial_trace_metadata"] = {
                    "app_version": "app-test",
                    "graph_version": "graph-test",
                }
                trace_context["quality_payload"] = {
                    "metadata": {
                        "quality_eval_enabled": True,
                        "needs_human_review": 0,
                        "dataset_candidate": 1,
                    }
                }

            return {
                "route": "benefits",
                "topic": "beneficios",
                "tool_name": "benefits_location_api",
                "answer": "Hay beneficios cerca tuyo.",
                "final_answer": "Hay beneficios cerca tuyo.",
                "documents": [],
                "context": "",
                "needs_clarification": False,
                "memory": {},
            }

        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "dev",
                "EVALUATION_ENDPOINT_ENABLED": "true",
            },
            clear=False,
        ):
            with patch.object(evaluation_app, "get_runtime", return_value=object()):
                with patch.object(evaluation_app, "run_bot_query", side_effect=fake_run_bot_query):
                    response = client.post(
                        "/evaluate",
                        json={"question": "Que beneficios tengo cerca?"},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        body = response.get_json()
        self.assertEqual(body["answer"], "Hay beneficios cerca tuyo.")
        self.assertEqual(body["route"], "benefits")
        self.assertEqual(body["topic"], "beneficios")
        self.assertTrue(body["used_tool"])
        self.assertFalse(body["fallback"])
        self.assertFalse(body["needs_clarification"])
        self.assertEqual(body["documents_count"], 0)
        self.assertFalse(body["needs_human_review"])
        self.assertTrue(body["dataset_candidate"])
        self.assertEqual(body["trace_id"], "trace-123")
        self.assertEqual(body["latency_ms"], 123)
        self.assertEqual(body["app_version"], "app-test")
        self.assertEqual(body["graph_version"], "graph-test")
        self.assertTrue(str(body["session_id"]).startswith("eval-"))
        self.assertNotIn("<Response>", response.data.decode("utf-8"))
        self.assertEqual(captured_kwargs["langfuse_user_id"], "eval-user")
        self.assertEqual(captured_kwargs["observation_name"], "gala-evaluation-request")
        self.assertIn("evaluation", captured_kwargs["langfuse_tags"])
        self.assertIn("dataset_run", captured_kwargs["langfuse_tags"])

    def test_evaluate_rejects_missing_token_when_configured(self) -> None:
        evaluation_app = _load_app_module()
        client = evaluation_app.app.test_client()

        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "prod",
                "EVALUATION_ENDPOINT_ENABLED": "true",
                "EVALUATION_ENDPOINT_TOKEN": "secret-token",
            },
            clear=False,
        ):
            response = client.post(
                "/evaluate",
                json={"question": "hola"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"], "evaluation_unauthorized")

    def test_evaluate_prefixes_provided_non_eval_session_ids(self) -> None:
        evaluation_app = _load_app_module()
        client = evaluation_app.app.test_client()
        captured_kwargs: dict[str, object] = {}

        def fake_run_bot_query(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "route": "chitchat",
                "topic": "conversacion",
                "answer": "Hola",
                "final_answer": "Hola",
                "documents": [],
                "context": "",
                "needs_clarification": False,
                "memory": {},
            }

        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "dev",
                "EVALUATION_ENDPOINT_ENABLED": "true",
            },
            clear=False,
        ):
            with patch.object(evaluation_app, "get_runtime", return_value=object()):
                with patch.object(evaluation_app, "run_bot_query", side_effect=fake_run_bot_query):
                    response = client.post(
                        "/evaluate",
                        json={
                            "question": "hola",
                            "session_id": "manual-session",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["session_id"], "eval-manual_session")
        self.assertEqual(response.get_json()["session_id"], "eval-manual_session")


if __name__ == "__main__":
    unittest.main()
