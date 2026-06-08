from __future__ import annotations

import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from core.constants import FALLBACK_ANSWER
from observability.evaluators import (
    build_quality_metadata,
    build_quality_scores,
    evaluate_basic_quality_signals,
    evaluate_whatsapp_format,
    run_quality_evaluation,
)
from observability.langfuse_config import safe_update_current_trace
from observability.metrics import (
    build_categorical_scores,
    build_basic_scores,
    build_conversation_outcome,
    build_final_trace_metadata,
    build_initial_trace_metadata,
    build_node_end_metadata,
    build_node_scores,
    build_node_start_metadata,
    build_quality_status,
)
from observability.session_metrics import (
    build_session_categorical_scores,
    build_session_debug_payload,
    build_session_numeric_scores,
    reset_session_metrics,
    update_session_metrics,
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
        self.observation_scores: list[dict[str, object]] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def score_trace(self, **kwargs) -> None:
        self.trace_scores.append(kwargs)

    def score(self, **kwargs) -> None:
        self.observation_scores.append(kwargs)


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
        self.start_calls: list[dict[str, object]] = []

    def start_as_current_observation(self, **kwargs):
        self.start_kwargs = kwargs
        self.start_calls.append(kwargs)
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

    def test_build_node_metadata_and_scores_for_benefits(self) -> None:
        state = {
            "route": "benefits",
            "topic": "beneficios",
            "tool_name": "benefits_location_api",
            "tool_output": {
                "results_count": 3,
            },
            "needs_clarification": True,
            "missing_fields": ["user_location"],
            "answer": "respuesta",
            "final_answer": "respuesta",
        }

        start_metadata = build_node_start_metadata("benefits", state=state)
        end_metadata = build_node_end_metadata(
            "benefits",
            state=state,
            latency_ms=456,
        )
        scores = build_node_scores("benefits", state=state, latency_ms=456)

        self.assertEqual(start_metadata["node_name"], "benefits")
        self.assertEqual(start_metadata["results_count"], 3)
        self.assertTrue(start_metadata["needs_location"])
        self.assertEqual(end_metadata["node_status"], "success")
        self.assertEqual(end_metadata["node_latency_ms"], 456)
        self.assertEqual(end_metadata["tool_name"], "benefits_location_api")
        self.assertEqual(scores["benefits_latency_ms"], 456)
        self.assertEqual(scores["benefits_success"], 1)
        self.assertEqual(scores["benefits_error"], 0)
        self.assertEqual(scores["benefits_results_count"], 3)
        self.assertEqual(scores["benefits_used_tool"], 1)
        self.assertEqual(scores["benefits_needs_clarification"], 1)

    def test_build_categorical_scores_normalizes_route_topic_and_outcome(self) -> None:
        state = {
            "route": "bcra_credit_status",
            "topic": "situacion_crediticia_bcra",
            "tool_name": "bcra_credit_status",
            "tool_output": {},
            "needs_clarification": True,
            "missing_fields": ["identificacion"],
            "answer": "Necesito tu CUIT o CUIL.",
            "final_answer": "Necesito tu CUIT o CUIL.",
        }

        categorical_scores = build_categorical_scores(
            state,
            initial_metadata={"channel": "whatsapp", "environment": "prod"},
            final_metadata={
                "final_route": "bcra_credit_status",
                "final_topic": "situacion_crediticia_bcra",
                "used_tool": 1,
                "used_rag": 0,
                "needs_clarification": 1,
                "fallback_used": 0,
                "guardrail_blocked": 0,
            },
        )

        self.assertEqual(categorical_scores["conversation_route"], "bcra_agent")
        self.assertEqual(categorical_scores["conversation_topic"], "situacion_crediticia")
        self.assertEqual(categorical_scores["conversation_channel"], "whatsapp")
        self.assertEqual(categorical_scores["conversation_environment"], "prod")
        self.assertEqual(categorical_scores["conversation_tool_status"], "tool_needs_clarification")
        self.assertEqual(categorical_scores["conversation_quality_status"], "healthy")
        self.assertEqual(categorical_scores["conversation_outcome"], "bcra_agent_needs_clarification")

    def test_build_categorical_scores_detects_rag_without_context(self) -> None:
        state = {
            "route": "loans_rag",
            "topic": "prestamos",
            "documents": [],
            "context": "",
            "answer": "Respuesta sin contexto.",
            "final_answer": "Respuesta sin contexto.",
        }

        categorical_scores = build_categorical_scores(
            state,
            initial_metadata={"channel": "api", "environment": "staging"},
            final_metadata={
                "final_route": "loans_rag",
                "final_topic": "prestamos",
                "used_rag": 1,
                "has_context": 0,
                "documents_count": 0,
                "fallback_used": 0,
                "guardrail_blocked": 0,
            },
        )

        self.assertEqual(categorical_scores["conversation_route"], "rag")
        self.assertEqual(categorical_scores["conversation_rag_status"], "rag_without_context")
        self.assertEqual(categorical_scores["conversation_outcome"], "rag_no_context")
        self.assertEqual(
            build_conversation_outcome(state, final_metadata={"used_rag": 1, "documents_count": 0}),
            "rag_no_context",
        )
        self.assertEqual(
            build_quality_status(state, final_metadata={"fallback_used": 1}),
            "fallback",
        )


class LangfuseConfigTests(unittest.TestCase):
    def test_safe_update_current_trace_uses_propagate_attributes_with_sdk_compatible_args(self) -> None:
        recorded_calls: list[dict[str, object]] = []

        @contextmanager
        def fake_propagate_attributes(**kwargs):
            recorded_calls.append(kwargs)
            yield

        with patch("observability.langfuse_config.propagate_attributes", fake_propagate_attributes):
            result = safe_update_current_trace(
                object(),
                name="gala-whatsapp-request",
                user_id="user-123",
                session_id="session-456",
                tags=["gala", "", "whatsapp", "gala"],
                metadata={
                    "channel": "whatsapp",
                    "input_chars": 4,
                    "has_location": True,
                    "ignored": None,
                },
            )

        self.assertTrue(result)
        self.assertEqual(len(recorded_calls), 1)
        self.assertEqual(
            recorded_calls[0],
            {
                "trace_name": "gala-whatsapp-request",
                "user_id": "user-123",
                "session_id": "session-456",
                "tags": ["gala", "whatsapp"],
                "metadata": {
                    "channel": "whatsapp",
                    "input_chars": "4",
                    "has_location": "true",
                },
            },
        )

    def test_safe_update_current_trace_returns_false_when_propagation_fails(self) -> None:
        @contextmanager
        def failing_propagate_attributes(**_kwargs):
            raise RuntimeError("boom")
            yield

        with patch("observability.langfuse_config.propagate_attributes", failing_propagate_attributes):
            result = safe_update_current_trace(
                object(),
                name="gala-whatsapp-request",
                user_id="user-123",
            )

        self.assertFalse(result)


class QualityEvaluatorTests(unittest.TestCase):
    def test_evaluate_whatsapp_format_detects_common_issues(self) -> None:
        result = evaluate_whatsapp_format(
            "**Hola**\n\n\n"
            "*Titulo\n"
            "```python\nprint('x')\n```"
        )

        self.assertEqual(result["whatsapp_format_ok"], 0)
        self.assertEqual(result["whatsapp_has_double_asterisk"], 1)
        self.assertEqual(result["whatsapp_has_broken_bold"], 1)
        self.assertEqual(result["whatsapp_has_code_block"], 1)
        self.assertGreaterEqual(result["whatsapp_format_issues_count"], 3)

    def test_build_quality_scores_uses_programmatic_signals_when_judge_is_disabled(self) -> None:
        state = {
            "question": "Quiero saber sobre prestamos personales",
            "route": "fallback",
            "answer": FALLBACK_ANSWER,
            "final_answer": FALLBACK_ANSWER,
            "documents": [],
            "context": "",
            "needs_clarification": False,
        }

        basic_signals = evaluate_basic_quality_signals(state)
        scores = build_quality_scores(state, basic_signals=basic_signals)
        metadata = build_quality_metadata(state, basic_signals=basic_signals, quality_scores=scores)

        self.assertEqual(scores["likely_low_value_answer"], 1)
        self.assertEqual(scores["needs_human_review"], 1)
        self.assertEqual(scores["dataset_candidate"], 1)
        self.assertNotIn("routing_correctness", scores)
        self.assertEqual(metadata["needs_human_review"], 1)
        self.assertEqual(metadata["dataset_candidate"], 1)

    def test_run_quality_evaluation_skips_llm_judge_by_default(self) -> None:
        state = {
            "question": "Hola",
            "route": "chitchat",
            "answer": "Hola, soy Gala.",
            "final_answer": "Hola, soy Gala.",
            "documents": [],
            "context": "",
        }

        with patch.dict(
            "os.environ",
            {
                "LANGFUSE_QUALITY_EVAL_ENABLED": "true",
                "LLM_JUDGE_ENABLED": "false",
            },
            clear=False,
        ):
            result = run_quality_evaluation(state, client=None)

        self.assertIn("whatsapp_format_ok", result["scores"])
        self.assertEqual(result["metadata"]["quality_eval_enabled"], True)
        self.assertEqual(result["metadata"]["llm_judge_enabled"], False)
        self.assertFalse(result["llm_judge_ran"])


class SessionMetricsTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session_metrics("session-demo")

    def test_update_session_metrics_accumulates_safe_aggregates(self) -> None:
        reset_session_metrics("session-demo")

        update_session_metrics(
            "session-demo",
            {
                "route": "chitchat",
                "topic": "saludo",
                "answer": "Hola",
                "final_answer": "Hola",
            },
            scores={
                "total_latency_ms": 100,
                "answer_length": 4,
                "retrieval_docs_count": 0,
                "used_rag": 0,
                "used_tool": 0,
                "fallback_used": 0,
            },
            metadata={"channel": "whatsapp", "environment": "prod"},
        )
        update_session_metrics(
            "session-demo",
            {
                "route": "benefits",
                "topic": "beneficios",
                "tool_name": "benefits_location_api",
                "answer": "Necesito tu ubicacion.",
                "final_answer": "Necesito tu ubicacion.",
            },
            scores={
                "total_latency_ms": 200,
                "answer_length": 22,
                "retrieval_docs_count": 0,
                "used_rag": 0,
                "used_tool": 1,
                "needs_clarification": 1,
                "fallback_used": 0,
            },
            metadata={"tool_name": "benefits_location_api"},
        )
        session_metrics = update_session_metrics(
            "session-demo",
            {
                "route": "loans_rag",
                "topic": "prestamos",
                "documents": [{"id": 1}, {"id": 2}],
                "context": "contexto",
                "answer": "Te cuento sobre prestamos.",
                "final_answer": "Te cuento sobre prestamos.",
            },
            scores={
                "total_latency_ms": 300,
                "answer_length": 26,
                "retrieval_docs_count": 2,
                "used_rag": 1,
                "has_context": 1,
                "used_tool": 0,
                "fallback_used": 0,
            },
            metadata={"documents_count": 2},
        )

        numeric_scores = build_session_numeric_scores(session_metrics)
        categorical_scores = build_session_categorical_scores(session_metrics)
        debug_payload = build_session_debug_payload(session_metrics)

        self.assertEqual(numeric_scores["session_turn_count"], 3)
        self.assertEqual(numeric_scores["session_total_latency_ms"], 600)
        self.assertEqual(numeric_scores["session_avg_latency_ms"], 200.0)
        self.assertEqual(numeric_scores["session_max_latency_ms"], 300)
        self.assertEqual(numeric_scores["session_rag_turns_count"], 1)
        self.assertEqual(numeric_scores["session_tool_turns_count"], 1)
        self.assertEqual(numeric_scores["session_needs_clarification_count"], 1)
        self.assertEqual(numeric_scores["session_total_retrieval_docs"], 2)
        self.assertEqual(numeric_scores["session_distinct_routes_count"], 3)
        self.assertEqual(numeric_scores["session_distinct_topics_count"], 3)

        self.assertEqual(categorical_scores["session_status"], "mixed")
        self.assertEqual(categorical_scores["session_primary_route"], "mixed")
        self.assertEqual(categorical_scores["session_primary_topic"], "mixed")
        self.assertEqual(categorical_scores["session_has_rag"], "true")
        self.assertEqual(categorical_scores["session_has_tool"], "true")
        self.assertEqual(categorical_scores["session_has_fallback"], "false")
        self.assertEqual(categorical_scores["session_complexity"], "mixed")

        self.assertEqual(debug_payload["turn_count"], 3)
        self.assertEqual(debug_payload["fallback_count"], 0)
        self.assertEqual(debug_payload["rag_count"], 1)
        self.assertEqual(debug_payload["tool_count"], 1)
        self.assertEqual(debug_payload["status"], "mixed")
        self.assertEqual(sorted(debug_payload["routes_seen"]), ["benefits", "chitchat", "rag"])
        self.assertEqual(
            sorted(debug_payload["topics_seen"]),
            ["beneficios", "prestamos", "saludo"],
        )


class BotRunnerObservabilityTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_session_metrics("demo")
        reset_session_metrics("whatsapp-1")

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
            client=None,
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
        self.assertNotIn("callbacks", graph.calls[0][1])

    def test_run_bot_query_updates_metadata_and_scores_when_langfuse_is_available(self) -> None:
        bot_runner = _load_bot_runner_module()
        reset_session_metrics("whatsapp-1")
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
            client=None,
            langfuse_client=langfuse_client,
            settings=SimpleNamespace(),
        )

        with patch.dict(
            "os.environ",
            {
                "LANGFUSE_QUALITY_EVAL_ENABLED": "true",
                "LLM_JUDGE_ENABLED": "false",
            },
            clear=False,
        ):
            with patch.object(bot_runner, "safe_update_current_trace", return_value=True) as safe_update_current_trace_mock:
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
        self.assertEqual(len(langfuse_client.start_calls), 1)
        self.assertEqual(langfuse_client.start_kwargs["name"], "gala-whatsapp-request")
        self.assertNotIn("callbacks", graph.calls[0][1])
        self.assertEqual(safe_update_current_trace_mock.call_count, 2)

        initial_trace_call = safe_update_current_trace_mock.call_args_list[0]
        self.assertIs(initial_trace_call.args[0], langfuse_client)
        self.assertEqual(initial_trace_call.kwargs["name"], "gala-whatsapp-request")
        self.assertEqual(initial_trace_call.kwargs["user_id"], "whatsapp-1")
        self.assertEqual(initial_trace_call.kwargs["session_id"], "whatsapp-1")
        self.assertEqual(initial_trace_call.kwargs["tags"], ["gala", "whatsapp"])
        self.assertEqual(initial_trace_call.kwargs["metadata"]["channel"], "whatsapp")

        final_trace_call = safe_update_current_trace_mock.call_args_list[-1]
        self.assertEqual(final_trace_call.kwargs["metadata"]["final_route"], "benefits")
        self.assertEqual(final_trace_call.kwargs["metadata"]["quality_eval_enabled"], True)

        first_metadata = langfuse_client.span.updates[0]["metadata"]
        final_metadata = langfuse_client.span.updates[-1]["metadata"]
        score_names = {score["name"] for score in langfuse_client.span.trace_scores}

        self.assertEqual(first_metadata["channel"], "whatsapp")
        self.assertEqual(first_metadata["entrypoint"], "webhook")
        self.assertEqual(first_metadata["media_type"], "pdf")
        self.assertEqual(final_metadata["final_route"], "benefits")
        self.assertEqual(final_metadata["tool_name"], "benefits_location_api")
        self.assertEqual(final_metadata["quality_eval_enabled"], True)
        self.assertEqual(final_metadata["llm_judge_enabled"], False)
        self.assertIn("retrieval_docs_count", score_names)
        self.assertIn("has_context", score_names)
        self.assertIn("fallback_used", score_names)
        self.assertIn("answer_length", score_names)
        self.assertIn("total_latency_ms", score_names)
        self.assertIn("used_tool", score_names)
        self.assertIn("whatsapp_format_ok", score_names)
        self.assertIn("likely_low_value_answer", score_names)
        self.assertIn("needs_human_review", score_names)
        self.assertIn("dataset_candidate", score_names)
        self.assertIn("conversation_route", score_names)
        self.assertIn("conversation_topic", score_names)
        self.assertIn("conversation_outcome", score_names)
        self.assertIn("conversation_quality_status", score_names)
        self.assertIn("session_turn_count", score_names)
        self.assertIn("session_status", score_names)
        self.assertIn("session_primary_route", score_names)
        self.assertIn("session_complexity", score_names)

    def test_run_bot_query_continues_when_safe_update_current_trace_fails(self) -> None:
        bot_runner = _load_bot_runner_module()
        graph = FakeGraph(
            {
                "route": "chitchat",
                "answer": "hola",
                "final_answer": "hola",
                "documents": [],
                "context": "",
                "memory": {},
            }
        )
        langfuse_client = FakeLangfuseClient()
        runtime = SimpleNamespace(
            graph=graph,
            client=None,
            langfuse_client=langfuse_client,
            settings=SimpleNamespace(),
        )

        with patch.object(bot_runner, "safe_update_current_trace", return_value=False) as safe_update_current_trace_mock:
            with patch.object(bot_runner, "load_memory", return_value={}):
                result = bot_runner.run_bot_query(
                    runtime=runtime,
                    question="hola",
                    session_id="demo",
                    langfuse_tags=["gala", "whatsapp"],
                    observation_name="gala-whatsapp-request",
                )

        self.assertEqual(result["final_answer"], "hola")
        self.assertEqual(len(graph.calls), 1)
        self.assertNotIn("callbacks", graph.calls[0][1])
        self.assertTrue(langfuse_client.flush_called)
        self.assertEqual(safe_update_current_trace_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
