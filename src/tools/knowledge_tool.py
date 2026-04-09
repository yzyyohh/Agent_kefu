from __future__ import annotations

from dataclasses import dataclass

from src.rag.pipeline import RAGPipeline


@dataclass
class KnowledgeTool:
    rag: RAGPipeline

    name: str = "retrieve_knowledge"
    description: str = "从知识库检索与问题相关的内容"

    def run(self, query: str) -> str:
        hits = self.rag.retrieve(query)
        if not hits:
            return "未检索到可靠证据。"

        lines: list[str] = []
        for i, item in enumerate(hits, start=1):
            preview = item.chunk.content.replace("\n", " ")[:180]
            lines.append(
                f"[{i}] score={item.score:.3f} source={item.chunk.source} content={preview}"
            )
        return "\n".join(lines)
