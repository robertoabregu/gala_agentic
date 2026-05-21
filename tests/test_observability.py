from __future__ import annotations

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.constants import FALLBACK_ANSWER
from observability.metrics import (
    build_basic_scores,
    build_final_trace_metadata,
    build_initial_trace_metadata,
)


def _load_bot_runner_module():
    fake_faiss = types.ModuleType("faiss")
    with patch.dict(sys.modules, {"faiss": fake_faiss}):
        sys.modules.pop("core.bot_runner", None)
        return importlib.import_module("core.bot_runner")


class FakeGraph:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, object], dict[str, object] | None]] = []

    def invoke(
        self,
        state: dict[str, object],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append((state, config))
        return dict(self.result)


class BrokenLangfuseClient:
    def start_as_current_observation(self, **_kwargs):
        raise RuntimeError("langfuse unavailable")


class FakeSpan:
    def __init__(self) -> None:
        self.trace_id = "trace-123"
        self.updates: list[dict[str, object]] = []
        self.trace_scores: list[dict[str, object]] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def score_trace(self, **kwargs) -> None:
        self.trace_scores.append(kwargs)


class FakeObservationContextManager:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.span = FakeSpan()
        self.flush_called = False
        self.start_kwargs: dict[str, object] | None = None

    def start_as_current_observation(self, **kwargs):
        self.start_kwargs = kwargs
        return FakeObservationContextManager(self.span)

    def flush(self) -> None:
        self.flush_called = True


class ObservabilityMetricsTests(unittest.TestCase):
    def test_build_initial_trace_metadata_infers_whatsapp_and_pdf(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "prod",
                "APP_VERSION": "abc123",
                "GALA_GRAPH_VERSION": "gala_graph_v2",
            },
            clear=False,
        ):
            metadata = build_initial_trace_metadata(
                question="hola",
                user_location={"latitude": "-34.5", "longitude": "-58.4"},
                media={"content_type": "application/pdf", "filename": "statement.pdf"},
                langfuse_tags=["gala", "whatsapp"],
                observation_name="gala-whatsapp-request",
            )

        self.assertEqual(metadata["channel"], "whatsapp")
        self.assertEqual(metadata["entrypoint"], "webhook")
        self.assertEqual(metadata["environment"], "prod")
        self.assertEqual(metadata["app_version"], "abc123")
        self.assertEqual(metadata["graph_version"], "gala_graph_v2")
        self.assertTrue(metadata["has_location"])
        self.assertTrue(metadata["has_media"])
        self.assertEqual(metadata["media_type"], "pdf")
        self.assertEqual(metadata["input_chars"], 4)

    def test_build_final_metadata_and_scores_infer_flags_safely(self) -> None:
        state = {
            "route": "branch_locator",
            "answer": "draft answer",
            "final_answer": FALLBACK_ANSWER,
            "documents": [{"id": 1}, {"id": 2}],
            "context": "context",
            "search_query": "",
            "needs_clarification": True,
            "memory": {"last_topic": "sucursales_cercanas"},
        }

        metadata = build_final_trace_metadata(state, total_latency_ms=321)
        scores = build_basic_scores(state, total_latency_ms=321)

        self.assertEqual(metadata["final_route"], "branch_locator")
        self.assertEqual(metadata["final_topic"], "sucursales_cercanas")
        self.assertTrue(metadata["used_tool"])
        self.assertEqual(metadata["tool_name"], "branch_locator")
        self.assertTrue(metadata["fallback"])
        self.assertTrue(metadata["guardrail_blocked"])
        self.assertTrue(metadata["needs_clarification"])
        self.assertEqual(metadata["documents_count"], 2)
        self.assertEqual(metadata["total_latency_ms"], 321)

        self.assertEqual(scores["total_latency_ms"], 321)
        self.assertEqual(scores["answer_length"], len(FALLBACK_ANSWER))
        self.assertEqual(scores["retrieval_docs_count"], 2)
        self.assertEqual(scores["has_context"], 1)
        self.assertEqual(scores["fallback_used"], 1)
        self.assertEqual(scores["guardrail_blocked"], 1)
        self.assertEqual(scores["used_tool"], 1)
        self.assertEqual(scores["used_rag"], 1)
        self.assertEqual(scores["needs_clarification"], 1)


class BotRunnerObservabilityTests(unittest.TestCase):
    def test_run_bot_query_continues_when_langfuse_span_fails(self) -> None:
        bot_runner = _load_bot_runner_module()
        graph = FakeGraph(
            {
                "route": "chitchat",
                "answer": "ok",
                "final_answer": "ok",
                "documents": [],
                "context": "",
                "memory": {},
            }
        )
        runtime = SimpleNamespace(
            graph=graph,
            langfuse_handler=None,
            langfuse_client=BrokenLangfuseClient(),
            settings=SimpleNamespace(),
        )

        with patch.object(bot_runner, "load_memory", return_value={}):
            result = bot_runner.run_bot_query(
                runtime=runtime,
                question="hola",
                session_id="demo",
            )

        self.assertEqual(result["final_answer"], "ok")
        self.assertEqual(len(graph.calls), 1)

    def test_run_bot_query_updates_metadata_and_scores_when_langfuse_is_available(self) -> None:
        bot_runner = _load_bot_runner_module()
        graph = FakeGraph(
            {
                "route": "benefits",
                "topic": "beneficios",
                "tool_name": "benefits_location_api",
                "answer": "respuesta final",
                "final_answer": "respuesta final",
                "documents": [],
                "context": "",
                "search_query": "",
                "needs_clarification": False,
                "memory": {"last_topic": "beneficios"},
            }
        )
        langfuse_client = FakeLangfuseClient()
        runtime = SimpleNamespace(
            graph=graph,
            langfuse_handler="handler",
            langfuse_client=langfuse_client,
            settings=SimpleNamespace(),
        )

        with patch.object(bot_runner, "load_memory", return_value={}):
            result = bot_runner.run_bot_query(
                runtime=runtime,
                question="beneficios cerca",
                session_id="whatsapp-1",
                langfuse_tags=["gala", "whatsapp"],
                observation_name="gala-whatsapp-request",
                user_location={"latitude": "-34.5", "longitude": "-58.4"},
                media={"content_type": "application/pdf", "filename": "statement.pdf"},
            )

        self.assertEqual(result["final_answer"], "respuesta final")
        self.assertTrue(langfuse_client.flush_called)
        self.assertEqual(langfuse_client.start_kwargs["name"], "gala-whatsapp-request")
        self.assertEqual(graph.calls[0][1]["callbacks"], ["handler"])

        first_metadata = langfuse_client.span.updates[0]["metadata"]
        final_metadata = langfuse_client.span.updates[-1]["metadata"]
        score_names = {score["name"] for score in langfuse_client.span.trace_scores}

        self.assertEqual(first_metadata["channel"], "whatsapp")
        self.assertEqual(first_metadata["entrypoint"], "webhook")
        self.assertEqual(first_metadata["media_type"], "pdf")
        self.assertEqual(final_metadata["final_route"], "benefits")
        self.assertEqual(final_metadata["tool_name"], "benefits_location_api")
        self.assertIn("retrieval_docs_count", score_names)
        self.assertIn("has_context", score_names)
        self.assertIn("fallback_used", score_names)
        self.assertIn("answer_length", score_names)
        self.assertIn("total_latency_ms", score_names)
        self.assertIn("used_tool", score_names)


if __name__ == "__main__":
    unittest.main()
