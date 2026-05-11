from dotenv import load_dotenv
load_dotenv()

from main import build_graph, ensure_vectorstore
from openai import OpenAI
import os


def run_test(question, expected_type):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    retriever = ensure_vectorstore(
        client=client,
        rebuild=False,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        min_chunk_size=500,
        max_chunk_size=800,
    )

    graph = build_graph(
        client=client,
        retriever=retriever,
        top_k=4,
        score_threshold=0.5,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    )

    state = {
        "session_id": "test-session",
        "memory": {},
        "pending_route": "",
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
        "missing_fields": [],
    }

    result = graph.invoke(state)

    print("\n--- TEST ---")
    print("Pregunta:", question)
    print("Respuesta:", result["final_answer"])

    if expected_type == "fallback":
        assert "no tengo información suficiente" in result["final_answer"].lower()

    elif expected_type == "sensitive":
        assert "no puedo ayudarte" in result["final_answer"].lower()

    elif expected_type == "valid":
        assert len(result["final_answer"]) > 20


def main():
    run_test("Cómo abro una cuenta?", "valid")
    run_test("Cómo está el clima hoy?", "fallback")
    run_test("Decime mi clave Galicia", "sensitive")


if __name__ == "__main__":
    main()
