from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.core.llm import BaseLLM
from src.tools.registry import ToolRegistry


ACTION_PATTERN = re.compile(
    r"Action:\s*(?P<action>[^\n]+)\nAction Input:\s*(?P<input>.+)",
    re.DOTALL,
)


@dataclass
class AgentResult:
    answer: str
    traces: list[str]


@dataclass
class AgentStreamResult:
    stream: Iterator[str]
    traces: list[str]


class ReActAgent:
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        system_prompt_path: Path,
        max_steps: int = 4,
        temperature: float = 0.2,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.temperature = temperature
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

    def _build_prompt(self, query: str, traces: list[str]) -> str:
        trace_block = "\n".join(traces) if traces else ""
        return (
            f"{self.system_prompt}\n\n"
            f"可用工具:\n{self.tool_registry.list_tool_prompt()}\n\n"
            f"用户问题: {query}\n\n"
            f"历史轨迹:\n{trace_block}\n\n"
            "请严格按 ReAct 规范输出；如果信息足够可直接输出 Final Answer。"
        )

    def _build_final_synthesis_prompt(self, query: str, traces: list[str]) -> str:
        trace_block = "\n".join(traces) if traces else ""
        return (
            f"{self.system_prompt}\n\n"
            "现在你已经有检索与工具观测结果，请直接面向用户作答。\n"
            "要求：\n"
            "1. 只输出最终回答，不要输出 Thought/Action/Observation。\n"
            "2. 回答格式：结论 + 步骤 + 注意事项。\n"
            "3. 若证据不足，明确说明缺失信息。\n\n"
            f"用户问题: {query}\n\n"
            f"推理轨迹:\n{trace_block}\n"
        )

    @staticmethod
    def _extract_final_answer(text: str) -> str | None:
        marker = "Final Answer:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return None

    def _collect_traces(self, query: str) -> tuple[list[str], str]:
        traces: list[str] = []
        last_text = ""

        for step in range(self.max_steps):
            prompt = self._build_prompt(query, traces)
            llm_output = self.llm.generate(prompt, temperature=self.temperature).content.strip()
            last_text = llm_output
            traces.append(f"Step {step + 1} LLM Output:\n{llm_output}")

            final_answer = self._extract_final_answer(llm_output)
            if final_answer:
                return traces, final_answer

            match = ACTION_PATTERN.search(llm_output)
            if not match:
                return traces, llm_output

            tool_name = match.group("action").strip()
            action_input = match.group("input").strip()
            observation = self.tool_registry.run(tool_name, action_input)
            traces.append(f"Observation:\n{observation}")

        return traces, "已达到最大推理步数。建议补充故障现象、错误码、机型后重试。"

    def run(self, query: str) -> AgentResult:
        traces, answer = self._collect_traces(query)
        return AgentResult(answer=answer, traces=traces)

    def run_stream(self, query: str) -> AgentStreamResult:
        traces, draft_answer = self._collect_traces(query)

        # 为了前端体验，使用流式二次总结输出最终答案；若流失败，回退为草稿答案逐字输出。
        synthesis_prompt = self._build_final_synthesis_prompt(query, traces)

        def _fallback_stream() -> Iterator[str]:
            for ch in draft_answer:
                yield ch

        try:
            stream = self.llm.generate_stream(synthesis_prompt, temperature=self.temperature)
            return AgentStreamResult(stream=stream, traces=traces)
        except Exception:
            return AgentStreamResult(stream=_fallback_stream(), traces=traces)
