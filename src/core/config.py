from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    raw: dict[str, Any]

    @property
    def project_name(self) -> str:
        return self.raw["project"]["name"]

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["paths"]["data_dir"])

    @property
    def index_dir(self) -> Path:
        return Path(self.raw["paths"]["index_dir"])

    @property
    def log_dir(self) -> Path:
        return Path(self.raw["paths"]["log_dir"])

    @property
    def models(self) -> dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def rag(self) -> dict[str, Any]:
        return self.raw["rag"]

    @property
    def agent(self) -> dict[str, Any]:
        return self.raw["agent"]


def load_settings(config_path: str = "configs/settings.yaml") -> Settings:
    load_dotenv(override=False)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return Settings(raw=content)
