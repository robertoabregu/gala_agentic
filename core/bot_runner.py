from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ingestion.chunker import chunk_documents
from ingestion.embeddings import create_embeddings, load_documents
from memory.local_memory import load_memory
from observability.langfuse_config import safe_score, safe_update_observation
from observability.metrics import (
    build_basic_scores,
    build_final_trace_metadata,
    build_initial_trace_metadata,
    duration_ms,
    now_ms,
)
from observability.tracing import node_observability_context
from rag.retriever import LocalFaissRetriever


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SCORE_DEFINITIONS = {
    "total_latency_ms": {
        "data_type": "NUMERIC",
        "comment": "Latencia total de la ejecucion del bot en milisegundos.",
    },
    "answer_length": {
        "data_type": "NUMERIC",
        "comment": "Cantidad de caracteres de la respuesta final.",
    },
    "retrieval_docs_count": {
        "data_type": "NUMERIC",
        "comment": "Cantidad de documentos recuperados por el retriever.",
    },
    "has_context": {
        "data_type": "BOOLEAN",
        "comment": "Indica si el sistema construyo contexto RAG.",
    },
    "fallback_used": {
        "data_type": "BOOLEAN",
        "comment": "Indica si termino usando la respuesta fallback.",
    },
    "guardrail_blocked": {
        "data_type": "BOOLEAN",
        "comment": "Indica si guardrail modifico o bloqueo la respuesta final.",
    },
    "used_rag": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la ejecucion utilizo flujo RAG.",
    },
    "used_tool": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la ejecucion utilizo una tool o integracion.",
    },
    "needs_clarification": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la respuesta final requiere aclaracion o datos faltantes.",
    },
    "execution_error": {
        "data_type": "BOOLEAN",
        "comment": "Indica si la ejecucion del grafo termino con error.",
    },
}


@dataclass
class RuntimeSettings:
    api_key: str
    embedding_model: str
    chat_model: str
    min_chunk_size: int
    max_chunk_size: int
    score_threshold: float


@dataclass
class BotRuntime:
    client: OpenAI
    retriever: LocalFaissRetriever
    graph: Any | None
    langfuse_handler: Any | None
    langfuse_client: Any | None
    settings: RuntimeSettings
    top_k: int


def build_graph(*args, **kwargs):
    from graph.gala_graph import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def get_langfuse_handler():
    from observability.langfuse_config import get_langfuse_handler as _get_langfuse_handler

    return _get_langfuse_handler()


def get_langfuse_client():
    from langfuse import get_client

    return get_client()


def load_runtime_settings() -> RuntimeSettings:
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Copia .env.example a .env y completa la clave."
        )

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "500"))
    max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "800"))
    score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

    if min_chunk_size <= 0 or max_chunk_size <= 0 or min_chunk_size > max_chunk_size:
        raise ValueError(
            "MIN_CHUNK_SIZE y MAX_CHUNK_SIZE tienen una configuracion invalida."
        )

    return RuntimeSettings(
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        score_threshold=score_threshold,
    )


def build_vectorstore(
    client: OpenAI,
    documents_path: Path,
    vectorstore_dir: Path,
    embedding_model: str,
    min_chunk_size: int,
    max_chunk_size: int,
) -> LocalFaissRetriever:
    documents = load_documents(documents_path)
    chunks = chunk_documents(
        documents,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    if not chunks:
        raise ValueError(
            "No se encontraron documentos validos para indexar en data/documents.json."
        )

    embeddings = create_embeddings(
        client,
        [chunk["content"] for chunk in chunks],
        model=embedding_model,
    )

    retriever = LocalFaissRetriever(vectorstore_dir, embedding_model=embedding_model)
    retriever.build(chunks, embeddings)

    print(f"Indice generado con {len(documents)} documentos y {len(chunks)} chunks.")
    return retriever


def ensure_vectorstore(
    client: OpenAI,
    rebuild: bool,
    embedding_model: str,
    min_chunk_size: int,
    max_chunk_size: int,
) -> LocalFaissRetriever:
    retriever = LocalFaissRetriever(VECTORSTORE_DIR, embedding_model=embedding_model)

    if rebuild or not retriever.exists():
        return build_vectorstore(
            client=client,
            documents_path=DOCUMENTS_PATH,
            vectorstore_dir=VECTORSTORE_DIR,
            embedding_model=embedding_model,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    retriever.load()
    return retriever


def prepare_runtime(
    top_k: int = 4,
    rebuild: bool = False,
    include_graph: bool = True,
    include_langfuse: bool = True,
) -> BotRuntime:
    settings = load_runtime_settings()
    client = OpenAI(api_key=settings.api_key)
    retriever = ensure_vectorstore(
        client=client,
        rebuild=rebuild,
        embedding_model=settings.embedding_model,
        min_chunk_size=settings.min_chunk_size,
        max_chunk_size=settings.max_chunk_size,
    )

    graph = None
    if include_graph:
        graph = build_graph(
            client=client,
            retriever=retriever,
            top_k=top_k,
            score_threshold=settings.score_threshold,
            chat_model=settings.chat_model,
        )

    langfuse_handler = None
    langfuse_client = None
    if include_langfuse:
        langfuse_handler = get_langfuse_handler()
        if langfuse_handler:
            langfuse_client = get_langfuse_client()

    return BotRuntime(
        client=client,
        retriever=retriever,
        graph=graph,
        langfuse_handler=langfuse_handler,
        langfuse_client=langfuse_client,
        settings=settings,
        top_k=top_k,
    )


def build_initial_state(
    question: str,
    session_id: str,
    user_location: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = load_memory(session_id)
    persisted_location = memory.get("user_location", {})
    effective_user_location = (
        user_location
        if isinstance(user_location, dict) and user_location
        else persisted_location if isinstance(persisted_location, dict) else {}
    )

    return {
        "session_id": session_id,
        "memory": memory,
        "pending_route": memory.get("pending_route", ""),
        "question": question,
        "original_question": question,
        "standalone_question": question,
        "is_followup": False,
        "route": "",
        "search_query": "",
        "documents": [],
        "context": "",
        "answer": "",
        "final_answer": "",
        "error": None,
        "tool_name": "",
        "tool_input": {},
        "tool_output": {},
        "needs_clarification": False,
        "missing_fields": memory.get("missing_fields", []),
        "user_location": effective_user_location,
        "media": media or {},
        "credit_card_statement": memory.get("credit_card_statement", {}),
    }


def _log_observability(message: str) -> None:
    print(f"[observability] {message}")


def _score_result(span: Any, result: dict[str, Any], total_latency_ms: int | None = None) -> None:
    scores = build_basic_scores(result, total_latency_ms=total_latency_ms)
    trace_id = getattr(span, "trace_id", None)
    scores_sent = 0

    for name, value in scores.items():
        score_definition = SCORE_DEFINITIONS.get(name, {})
        if safe_score(
            span,
            trace_id,
            name,
            value,
            data_type=score_definition.get("data_type"),
            comment=score_definition.get("comment"),
        ):
            scores_sent += 1

    _log_observability(f"scores sent={scores_sent}")


def run_bot_query(
    runtime: BotRuntime,
    question: str,
    session_id: str,
    *,
    langfuse_user_id: str | None = None,
    langfuse_tags: list[str] | None = None,
    observation_name: str = "gala-rag-request",
    user_location: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime.graph is None:
        raise RuntimeError("El runtime no fue preparado con un grafo ejecutable.")

    start_ms = now_ms()
    resolved_langfuse_tags = langfuse_tags or ["gala", "langgraph", "rag", "local-prototype"]
    initial_state = build_initial_state(
        question=question,
        session_id=session_id,
        user_location=user_location,
        media=media,
    )
    initial_trace_metadata = build_initial_trace_metadata(
        question=question,
        user_location=initial_state.get("user_location"),
        media=media,
        langfuse_tags=resolved_langfuse_tags,
        observation_name=observation_name,
    )
    app_version = str(initial_trace_metadata.get("app_version") or "").strip() or None

    config: dict[str, Any] = {
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": langfuse_user_id or session_id,
            "langfuse_tags": resolved_langfuse_tags,
        }
    }

    if runtime.langfuse_handler:
        config["callbacks"] = [runtime.langfuse_handler]

    if runtime.langfuse_client:
        result: dict[str, Any] | None = None
        graph_error: Exception | None = None

        try:
            with runtime.langfuse_client.start_as_current_observation(
                as_type="span",
                name=observation_name,
            ) as span:
                safe_update_observation(
                    span,
                    metadata=initial_trace_metadata,
                    version=app_version,
                )

                try:
                    with node_observability_context(runtime.langfuse_client):
                        result = runtime.graph.invoke(initial_state, config=config)
                except Exception as exc:
                    graph_error = exc
                    total_latency_ms = duration_ms(start_ms)
                    final_trace_metadata = build_final_trace_metadata(
                        initial_state,
                        total_latency_ms=total_latency_ms,
                        error=exc,
                    )
                    safe_update_observation(
                        span,
                        metadata={**initial_trace_metadata, **final_trace_metadata},
                        level="ERROR",
                        status_message=type(exc).__name__,
                        version=app_version,
                    )
                    safe_score(
                        span,
                        getattr(span, "trace_id", None),
                        "execution_error",
                        1,
                        data_type=SCORE_DEFINITIONS["execution_error"]["data_type"],
                        comment=SCORE_DEFINITIONS["execution_error"]["comment"],
                    )
                    safe_score(
                        span,
                        getattr(span, "trace_id", None),
                        "total_latency_ms",
                        total_latency_ms,
                        data_type=SCORE_DEFINITIONS["total_latency_ms"]["data_type"],
                        comment=SCORE_DEFINITIONS["total_latency_ms"]["comment"],
                    )
                    _log_observability(f"total_latency_ms={total_latency_ms}")
                    raise

                total_latency_ms = duration_ms(start_ms)
                final_trace_metadata = build_final_trace_metadata(
                    result,
                    total_latency_ms=total_latency_ms,
                )
                safe_update_observation(
                    span,
                    metadata={**initial_trace_metadata, **final_trace_metadata},
                    version=app_version,
                )
                _score_result(span, result, total_latency_ms=total_latency_ms)
                _log_observability(f"total_latency_ms={total_latency_ms}")

            try:
                runtime.langfuse_client.flush()
            except Exception:
                _log_observability("Langfuse flush skipped")

            if result is not None:
                return result
        except Exception:
            if graph_error is not None:
                raise

            if result is not None:
                _log_observability("Langfuse post-processing failed, returning result")
                return result

            _log_observability("Langfuse span unavailable, continuing without advanced observability")

    try:
        fallback_config = config
        if runtime.langfuse_client:
            fallback_config = dict(config)
            fallback_config.pop("callbacks", None)

        result = runtime.graph.invoke(initial_state, config=fallback_config)
        total_latency_ms = duration_ms(start_ms)
        _log_observability(f"total_latency_ms={total_latency_ms}")
        return result
    except Exception:
        total_latency_ms = duration_ms(start_ms)
        _log_observability(f"total_latency_ms={total_latency_ms}")
        raise
