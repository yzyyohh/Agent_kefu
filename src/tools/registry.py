from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    runner: Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, runner: Callable[..., str]) -> None:
        self.tools[name] = ToolSpec(name=name, description=description, runner=runner)

    def list_tool_prompt(self) -> str:
        lines = []
        for spec in self.tools.values():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines)

    def run(self, tool_name: str, action_input: str) -> str:
        if tool_name not in self.tools:
            return f"工具不存在: {tool_name}"

        parsed: dict = {}
        if action_input.strip():
            try:
                parsed = json.loads(action_input)
            except json.JSONDecodeError:
                parsed = {"query": action_input}

        try:
            return self.tools[tool_name].runner(**parsed)
        except TypeError as e:
            return f"工具参数错误: {e}"
