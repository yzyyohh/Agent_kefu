from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agents.react_agent import ReActAgent
from src.core.config import load_settings
from src.core.llm import build_llm
from src.rag.pipeline import RAGConfig, RAGPipeline
from src.tools.knowledge_tool import KnowledgeTool
from src.tools.registry import ToolRegistry
from src.tools.robot_tool import RobotActionTool


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str | None = Field(default=None, description="会话ID")


class ChatResponse(BaseModel):
    answer: str
    traces: list[str]


@lru_cache(maxsize=1)
def build_agent() -> ReActAgent:
    settings = load_settings()
    rag_cfg = RAGConfig(**settings.rag)
    rag = RAGPipeline(config=rag_cfg, index_dir=settings.index_dir)
    rag.load_index()

    knowledge_tool = KnowledgeTool(rag=rag)
    action_tool = RobotActionTool()

    registry = ToolRegistry()
    registry.register(knowledge_tool.name, knowledge_tool.description, knowledge_tool.run)
    registry.register(action_tool.name, action_tool.description, action_tool.run)

    llm = build_llm()
    return ReActAgent(
        llm=llm,
        tool_registry=registry,
        system_prompt_path=Path("src/prompts/system_prompt.txt"),
        max_steps=settings.agent["max_steps"],
        temperature=settings.agent["temperature"],
    )


app = FastAPI(title="RAG ReAct Robot Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = build_agent()
    result = agent.run(req.query)
    return ChatResponse(answer=result.answer, traces=result.traces)
