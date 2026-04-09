from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.react_agent import ReActAgent
from src.core.config import load_settings
from src.core.llm import build_llm
from src.rag.pipeline import RAGConfig, RAGPipeline
from src.tools.knowledge_tool import KnowledgeTool
from src.tools.registry import ToolRegistry
from src.tools.robot_tool import RobotActionTool


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

    return ReActAgent(
        llm=build_llm(allow_mock_fallback=True),
        tool_registry=registry,
        system_prompt_path=Path("src/prompts/system_prompt.txt"),
        max_steps=settings.agent["max_steps"],
        temperature=settings.agent["temperature"],
    )


def main() -> None:
    agent = build_agent()
    print("RAG + ReAct Robot Agent CLI 已启动，输入 q 退出。")
    print(f"LLM backend={getattr(agent.llm, 'backend_name', 'unknown')} | model={getattr(agent.llm, 'model', 'unknown')}")

    while True:
        query = input("\n你: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break

        result = agent.run(query)
        print(f"\nAgent: {result.answer}")
        used_backend = getattr(agent.llm, "last_backend_used", getattr(agent.llm, "backend_name", "unknown"))
        last_error = getattr(agent.llm, "last_error", "")
        print(f"[LLM_USED] {used_backend}")
        if last_error:
            print(f"[LLM_FALLBACK_REASON] {last_error}")


if __name__ == "__main__":
    main()
