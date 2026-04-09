from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator
from urllib import error, request


@dataclass
class LLMResponse:
    content: str


class BaseLLM:
    backend_name: str = "base"
    model: str = "unknown"

    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse:
        raise NotImplementedError

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        full_text = self.generate(prompt, temperature=temperature).content
        for ch in full_text:
            yield ch


class DashScopeLLM(BaseLLM):
    backend_name = "dashscope-compatible"

    def __init__(self) -> None:
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = (
            os.getenv("DASHSCOPE_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = (
            os.getenv("CHAT_MODEL_NAME", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or "qwen3-max"
        )

        if not self.api_key:
            raise RuntimeError("No API key found. Please set DASHSCOPE_API_KEY.")

    def _post_json(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"DashScope HTTPError {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"DashScope request failed: {e}") from e

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from DashScope: {raw[:300]}") from e

    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        data = self._post_json(payload)

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected DashScope response shape: {str(data)[:500]}") from e

        return LLMResponse(content=content or "")

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
        }
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = ""
                    try:
                        delta = data["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        delta = ""

                    if delta:
                        yield delta
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"DashScope HTTPError {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"DashScope stream failed: {e}") from e


class MockLLM(BaseLLM):
    backend_name = "mock"
    model = "mock-react"

    def generate(self, prompt: str, temperature: float = 0.2) -> LLMResponse:
        lowered = prompt.lower()
        if "Action:" in prompt and "Observation:" in prompt:
            return LLMResponse(content="Final Answer: 基于检索证据，建议先检查滚刷和滤网，再执行深度清洁模式。")

        if "故障" in prompt or "报错" in prompt or "不工作" in prompt:
            return LLMResponse(
                content=(
                    "Thought: 我需要检索故障排除知识。\n"
                    "Action: retrieve_knowledge\n"
                    "Action Input: {\"query\": \"扫地机器人故障排除\"}"
                )
            )

        if "拖布" in prompt or "异味" in prompt:
            return LLMResponse(
                content=(
                    "Thought: 先检索维护保养内容。\n"
                    "Action: retrieve_knowledge\n"
                    "Action Input: {\"query\": \"拖布 异味 维护\"}"
                )
            )

        if "怎么" in lowered or "如何" in prompt:
            return LLMResponse(
                content=(
                    "Thought: 检索通用知识并给出步骤。\n"
                    "Action: retrieve_knowledge\n"
                    "Action Input: {\"query\": \"扫地机器人 使用 建议\"}"
                )
            )

        return LLMResponse(content="Final Answer: 我可以帮你排查扫地机器人问题，请告诉我更具体的现象。")


def build_llm(allow_mock_fallback: bool = True) -> BaseLLM:
    has_key = bool(os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip())

    if has_key:
        llm = DashScopeLLM()
        return llm

    if allow_mock_fallback:
        return MockLLM()

    raise RuntimeError("No API key found and mock fallback disabled.")
