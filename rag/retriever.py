import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from ingestion.embeddings import create_embeddings


class LocalFaissRetriever:
    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "chunks.json"

    def __init__(self, vectorstore_dir: Path, embedding_model: str) -> None:
        self.vectorstore_dir = Path(vectorstore_dir)
        self.index_path = self.vectorstore_dir / self.INDEX_FILENAME
        self.metadata_path = self.vectorstore_dir / self.METADATA_FILENAME
        self.embedding_model = embedding_model
        self.index = None
        self.chunks: list[dict] = []

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def build(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks or not embeddings:
            raise ValueError("No hay datos suficientes para construir el indice.")

        if len(chunks) != len(embeddings):
            raise ValueError("La cantidad de chunks no coincide con la cantidad de embeddings.")

        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self.index = index
        self.chunks = chunks
        self.save()

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No hay indice cargado para guardar.")

        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "embedding_model": self.embedding_model,
            "chunks": self.chunks,
        }
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError("No existe un indice FAISS local para cargar.")

        self.index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.embedding_model = payload.get("embedding_model", self.embedding_model)
        self.chunks = payload.get("chunks", [])

    def search(self, client: OpenAI, query: str, top_k: int = 4) -> list[dict]:
        if self.index is None:
            self.load()

        query_embedding = create_embeddings(
            client,
            [query],
            model=self.embedding_model,
        )[0]

        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results: list[dict] = []
        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0:
                continue

            chunk = dict(self.chunks[chunk_index])
            chunk["score"] = float(score)
            results.append(chunk)

        return results
