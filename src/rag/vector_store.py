from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.schema import DocumentChunk


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


class LocalVectorStore:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector store with empty chunks")
        self.chunks = chunks
        corpus = [c.content for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> list[RetrievedChunk]:
        if self.matrix is None:
            raise RuntimeError("Vector store is not built")

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().reshape(-1)

        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=score))
        return results

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)

        with (index_dir / "vectorizer.pkl").open("wb") as f:
            pickle.dump(self.vectorizer, f)

        with (index_dir / "matrix.pkl").open("wb") as f:
            pickle.dump(self.matrix, f)

        with (index_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> "LocalVectorStore":
        store = cls()

        with (index_dir / "vectorizer.pkl").open("rb") as f:
            store.vectorizer = pickle.load(f)

        with (index_dir / "matrix.pkl").open("rb") as f:
            store.matrix = pickle.load(f)

        with (index_dir / "chunks.json").open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        store.chunks = [DocumentChunk(**item) for item in raw_chunks]
        return store
