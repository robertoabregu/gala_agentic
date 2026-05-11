from dotenv import load_dotenv

load_dotenv()

import argparse

from core.bot_runner import (
    build_graph,
    ensure_vectorstore,
    prepare_runtime,
    run_bot_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG local simple para consultas sobre ayuda publica de Banco Galicia."
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Pregunta para responder. Si no se envia, se pide por consola.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Cantidad de chunks a recuperar.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruye el indice FAISS desde data/documents.json.",
    )
    parser.add_argument(
        "--session-id",
        default="demo-local",
        help="Identificador de sesion para memoria conversacional local.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    question = args.question
    include_graph = not (args.rebuild and not question)
    include_langfuse = include_graph

    runtime = prepare_runtime(
        top_k=args.top_k,
        rebuild=args.rebuild,
        include_graph=include_graph,
        include_langfuse=include_langfuse,
    )

    if args.rebuild and not question:
        print("Indice reconstruido.")
        return

    if not question:
        question = input("Pregunta: ").strip()

    if not question:
        raise ValueError("Debes ingresar una pregunta.")

    result = run_bot_query(
        runtime=runtime,
        question=question,
        session_id=args.session_id,
        langfuse_user_id="robert-local",
        langfuse_tags=["gala", "langgraph", "rag", "local-prototype"],
    )

    print("\nRespuesta:\n")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
