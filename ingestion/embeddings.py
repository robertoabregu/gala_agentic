import json
from pathlib import Path

from openai import OpenAI


def load_documents(documents_path: Path) -> list[dict]:
    raw_data = json.loads(documents_path.read_text(encoding="utf-8"))
    documents = raw_data.get("documents", []) if isinstance(raw_data, dict) else raw_data

    if not isinstance(documents, list):
        raise ValueError("data/documents.json debe contener una lista o un objeto con la clave 'documents'.")

    parsed_documents: list[dict] = []

    for index, item in enumerate(documents, start=1):
        if not isinstance(item, dict):
            continue

        content = (item.get("content") or item.get("text") or item.get("body") or "").strip()
        if not content:
            continue

        parsed_documents.append({
            **item,
            "id": str(item.get("id") or f"doc-{index}"),
            "title": item.get("title") or f"Documento {index}",
            "url": item.get("url") or "",
            "content": content,
        })

    return parsed_documents


def create_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for start_index in range(0, len(texts), batch_size):
        batch = texts[start_index : start_index + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings
