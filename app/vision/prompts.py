"""Prompt templates.

Templates live in `configs/prompts/*.txt` and use `string.Template` placeholders
(`$policy`, `$observations`, `$retrieved_context`) so JSON braces in the
templates stay literal. The classification policy is injected into every prompt
and kept separate from the factual knowledge documents.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from app.config import AppConfig, get_config

POLICY_FILE = "policy.txt"


class PromptError(RuntimeError):
    pass


@lru_cache(maxsize=64)
def _read(path: str) -> str:
    file = Path(path)
    if not file.exists():
        raise PromptError(f"Prompt template not found: {file}")
    return file.read_text(encoding="utf-8")


def load_policy(cfg: AppConfig | None = None) -> str:
    cfg = cfg or get_config()
    return _read(str(cfg.prompts_dir / POLICY_FILE)).strip()


def render(cfg: AppConfig | None, name: str, **fields: str) -> str:
    """Render `configs/prompts/{name}.txt` with the policy and extra fields."""
    cfg = cfg or get_config()
    raw = _read(str(cfg.prompts_dir / f"{name}.txt"))
    template = Template(raw)
    values = {"policy": load_policy(cfg), **fields}
    return template.substitute(values)
