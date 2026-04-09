from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.rag.chunker import split_text_into_chunks
from src.rag.loaders import load_documents
from src.rag.schema import DocumentChunk
from src.rag.vector_store import LocalVectorStore, RetrievedChunk


@dataclass
class RAGConfig:
    chunk_size: int = 350
    chunk_overlap: int = 80
    top_k: int = 4
    min_score: float = 0.05


class RAGPipeline:
    def __init__(self, config: RAGConfig, index_dir: Path) -> None:
        self.config = config
        self.index_dir = index_dir
        self.store: LocalVectorStore | None = None

    def build_index(self, data_dir: Path) -> int:
        docs = load_documents(data_dir)
        chunks: list[DocumentChunk] = []
        for source, content in docs:
            chunks.extend(
                split_text_into_chunks(
                    source=source,
                    text=content,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )

        store = LocalVectorStore()
        store.build(chunks)
        store.save(self.index_dir)
        self.store = store
        return len(chunks)

    def load_index(self) -> None:
        self.store = LocalVectorStore.load(self.index_dir)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self.store is None:
            self.load_index()
        assert self.store is not None
        return self.store.search(
            query=query,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )
