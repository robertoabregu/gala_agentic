from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from core.constants import FALLBACK_ANSWER
from ingestion.chunker import chunk_documents
from ingestion.embeddings import create_embeddings, load_documents
from memory.local_memory import load_memory
from rag.retriever import LocalFaissRetriever


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


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


def _score_result(span: Any, result: dict[str, Any]) -> None:
    docs_count = len(result.get("documents", []))
    has_context = 1 if result.get("context") else 0
    fallback_used = 1 if result.get("final_answer") == FALLBACK_ANSWER else 0
    answer_length = len(result.get("final_answer", ""))

    span.score_trace(
        name="retrieval_docs_count",
        value=docs_count,
        data_type="NUMERIC",
        comment="Cantidad de documentos recuperados por el retriever.",
    )

    span.score_trace(
        name="has_context",
        value=has_context,
        data_type="BOOLEAN",
        comment="Indica si el sistema construyo contexto RAG.",
    )

    span.score_trace(
        name="fallback_used",
        value=fallback_used,
        data_type="BOOLEAN",
        comment="Indica si termino usando la respuesta fallback.",
    )

    span.score_trace(
        name="answer_length",
        value=answer_length,
        data_type="NUMERIC",
        comment="Cantidad de caracteres de la respuesta final.",
    )


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

    initial_state = build_initial_state(
        question=question,
        session_id=session_id,
        user_location=user_location,
        media=media,
    )

    config: dict[str, Any] = {
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": langfuse_user_id or session_id,
            "langfuse_tags": langfuse_tags or ["gala", "langgraph", "rag", "local-prototype"],
        }
    }

    if runtime.langfuse_handler:
        config["callbacks"] = [runtime.langfuse_handler]

    if runtime.langfuse_client:
        with runtime.langfuse_client.start_as_current_observation(
            as_type="span",
            name=observation_name,
        ) as span:
            result = runtime.graph.invoke(initial_state, config=config)
            _score_result(span, result)

        runtime.langfuse_client.flush()
        return result

    return runtime.graph.invoke(initial_state, config=config)
