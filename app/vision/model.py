"""Reusable Vision Model interface backed by MLX (Apple Silicon).

One model instance serves both inference passes with identical decoding
settings. Models are loaded by stable model ID and managed per ID, so one
model cannot silently replace another.

Load policy: the local checkpoint directory is preferred; otherwise the
HuggingFace repo id is resolved from the local cache. No cloud inference
is performed and no model is downloaded automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from app.config import MODEL_REGISTRY as MODEL_SPECS
from app.config import AppConfig, VisionConfig, get_config
from app.vision.schemas import VisionUsage

logger = logging.getLogger(__name__)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
BUDGET_NOTE = "\n\n(Reasoning budget reached; answering now.)\n"
ANSWER_MAX_TOKENS = 256  # room for the JSON answer after a forced end of reasoning
# Indented JSON allowed: mlx-vlm's own helper forces compact JSON, and lfm2-vl-450m, which
# writes indented JSON, then rambled past the token limit in 34 of 200 First Passes (8 free).
# Bounded, so a model that wants to end early cannot fill the token limit with whitespace.
SCHEMA_OPTIONS = {"whitespace_pattern": r"[ \t\n\r]{0,16}"}


class VisionModelError(RuntimeError):
    """Raised when the vision model cannot be loaded or run."""


def resolve_cached_snapshot(repo: str) -> Path | None:
    """Return the complete local HF snapshot for ``repo``, never downloading.

    ``snapshot_download(local_files_only=True)`` only inspects the local cache;
    it raises when any required file is missing instead of contacting the Hub.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - mlx-vlm installs it
        return None
    try:
        return Path(snapshot_download(repo_id=repo, local_files_only=True))
    except Exception:  # noqa: BLE001 - missing/partial cache must not download
        return None


def resolve_model_source(vision: VisionConfig) -> Path | None:
    """The checkpoint the loader uses: the local directory, else a complete cached snapshot.

    Never calls the Hub: a missing cache is an explicit error telling the user to
    download the model themselves.
    """
    local = vision.resolve_path()
    if local.exists():
        return local
    return resolve_cached_snapshot(vision.model_repo) if vision.model_repo else None


_LLG_TOKENIZERS: dict[int, object] = {}  # walking the vocabulary takes about a second


def _schema_logits_processor(tokenizer, schema: type[BaseModel]):
    """An llguidance logits processor that only allows JSON matching `schema`."""
    import llguidance
    import llguidance.hf
    from mlx_vlm.structured import LLGuidanceLogitsProcessor

    if id(tokenizer) not in _LLG_TOKENIZERS:
        _LLG_TOKENIZERS[id(tokenizer)] = llguidance.hf.from_tokenizer(tokenizer)
    grammar = llguidance.LLMatcher.grammar_from_json_schema(
        schema.model_json_schema(), defaults=SCHEMA_OPTIONS
    )
    return LLGuidanceLogitsProcessor(grammar, _LLG_TOKENIZERS[id(tokenizer)])


class VisionModel:
    """Thin wrapper over mlx_vlm with a stable generate() contract."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        vision = self.config.vision
        source = resolve_model_source(vision)
        if source is None:
            repo_hint = (
                f" or a complete Hugging Face cache entry for {vision.model_repo!r}"
                if vision.model_repo
                else ""
            )
            raise VisionModelError(
                f"Vision model {vision.model_id!r} is not available locally. "
                f"Expected a checkpoint at {vision.resolve_path()}{repo_hint}. "
                "Run `make download` and retry. "
                "DOGGO RAG never downloads models automatically during inference."
            )
        target = str(source)
        try:
            from mlx_vlm import load
        except ImportError as exc:  # pragma: no cover - environment guard
            raise VisionModelError("mlx_vlm is not installed") from exc
        logger.info("loading vision model from %s", target)
        try:
            self._model, self._processor = load(target)
        except Exception as exc:
            raise VisionModelError(
                f"Failed to load vision model {target!r}: {exc}. "
                "Run `make download` to fetch the checkpoint."
            ) from exc

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        thinking: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> str:
        """Run one completion over `image` with the configured decoding settings.

        `thinking` lets the model reason before answering where its chat template supports it.
        `schema` constrains decoding to that JSON Schema (llguidance via mlx-vlm), so the
        model cannot drop a key or quote a boolean; its text still goes through validation.
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image

        self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [
                    # Image first: the ordering the evaluated checkpoints
                    # (LFM2/LFM3/Qwen3-VL) are trained on. With text first,
                    # LFM2-VL-450M ignores the image and answers "dog".
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = apply_chat_template(
            self._processor,
            self._model.config,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        image = load_image(image_path)
        kwargs: dict = {"verbose": False}
        if self.config.vision.temperature > 0:
            kwargs["temperature"] = self.config.vision.temperature
        if self.config.vision.seed is not None:
            kwargs["seed"] = self.config.vision.seed

        def complete(prefix: str, limit: int, constrained: bool) -> str:
            if schema is not None and constrained:
                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                # Stateful: a new processor per call.
                kwargs["logits_processors"] = [_schema_logits_processor(tokenizer, schema)]
            else:
                kwargs.pop("logits_processors", None)
            result = generate(
                self._model, self._processor, prefix, image=image, max_tokens=limit, **kwargs
            )
            text = getattr(result, "text", None)
            return text if isinstance(text, str) else str(result)

        reasoning = rendered.rstrip().endswith(THINK_OPEN)
        # shortcut: free reasoning text cannot match the schema, so a reasoning call is only
        # constrained for a forced answer; wrap in ThinkingAwareLogitsProcessor if it matters.
        text = complete(rendered, max_tokens, constrained=not reasoning)
        if not reasoning:
            return text
        if THINK_CLOSE not in text:
            # Budget forcing: close the reasoning at the token limit so the model answers
            # from what it has reasoned so far; the note marks the cut in the raw output.
            text += f"{BUDGET_NOTE}{THINK_CLOSE}"
            text += complete(rendered + text, ANSWER_MAX_TOKENS, constrained=True)
        # The template opens the reasoning block itself; keep the tag so the text shows it.
        return f"{THINK_OPEN}{text}"

    def usage(self, prompt_name: str, latency_s: float) -> VisionUsage:
        vision = self.config.vision
        return VisionUsage(
            model_id=vision.model_id,
            backend=vision.backend,
            temperature=vision.temperature,
            seed=vision.seed,
            prompt_name=prompt_name,
            latency_s=round(latency_s, 3),
        )


_MODEL_CACHE: dict[str, VisionModel] = {}


def get_vision_model(model_id: str | None = None) -> VisionModel:
    """Return the VisionModel for a supported stable ID; never substitute another.

    Only one model is kept in memory: selecting a different ID drops the
    previous one (AGENTS.md §4), so `make eval` never holds all three.
    """
    cfg = get_config()
    requested = model_id or cfg.vision.model_id
    if requested not in MODEL_SPECS:
        raise VisionModelError(
            f"Unknown vision model ID: {requested!r}. Known IDs: {sorted(MODEL_SPECS)}"
        )
    if requested not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[requested] = VisionModel(cfg.for_model(requested))
    return _MODEL_CACHE[requested]


def loaded_model_id() -> str | None:
    """The model in memory; unlike get_vision_model, never creates or evicts one."""
    return next((model_id for model_id, model in _MODEL_CACHE.items() if model.is_loaded), None)


def reset_model_cache() -> None:
    """Drop the cached model instance (used by tests)."""
    _MODEL_CACHE.clear()
