from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DPConfig:
    """Simple configuration wrapper for local DP settings."""

    enabled: bool = True
    target_epsilon: float = 2.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    accountant: str = "prv"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPConfig":
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"DP config file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}

        privacy = payload.get("privacy", {})
        return cls(
            enabled=bool(privacy.get("enabled", True)),
            target_epsilon=float(privacy.get("target_epsilon", 2.0)),
            target_delta=float(privacy.get("target_delta", 1e-5)),
            max_grad_norm=float(privacy.get("max_grad_norm", 1.0)),
            accountant=str(privacy.get("accountant", "prv")),
        )
