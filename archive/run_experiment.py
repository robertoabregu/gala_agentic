from dotenv import load_dotenv
load_dotenv()

import os

from openai import OpenAI
from langfuse import get_client

from graph.gala_graph import build_graph
from rag.retriever import LocalFaissRetriever


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = get_client()

dataset = langfuse.get_dataset("gala-rag-basic-v1")

retriever = LocalFaissRetriever(
    "data/vectorstore",
    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
)
retriever.load()

graph = build_graph(
    client=client,
    retriever=retriever,
    top_k=4,
    score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.5")),
    chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
)


def run_gala_graph(*, item, **kwargs):
    question = item.input["question"]

    initial_state = {
        "question": question,
        "route": "",
        "documents": [],
        "context": "",
        "answer": "",
        "final_answer": "",
        "error": None,
    }

    result = graph.invoke(initial_state)

    print("\n===================")
    print("QUESTION:", question)
    print("ANSWER:", result.get("final_answer", ""))

    return result


result = dataset.run_experiment(
    name="gala-rag-exp-v1",
    description="Primer experimento RAG de Gala",
    task=run_gala_graph,
)

print(result.format())

langfuse.flush()