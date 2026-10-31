"""Configuration loading: configs/default.yaml with environment overrides.

All relative paths resolve against the repository root. The retrieval query is
the First Pass description, so it is intentionally not a configuration value.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Stable model ID -> (local checkpoint, Hugging Face repo). The only place the
# three supported Vision Models are defined (spec §7).
MODEL_REGISTRY = {
    "lfm3-vl-2b": (
        "models/vision/LiquidAI/LFM3-VL-2B-MLX-4bit",
        "LiquidAI/LFM3-VL-2B-MLX-4bit",
    ),
    "lfm2-vl-450m": (
        "models/vision/mlx-community/LFM2-VL-450M-4bit",
        "mlx-community/LFM2-VL-450M-4bit",
    ),
    "qwen3-vl-2b": (
        "models/vision/mlx-community/Qwen3-VL-2B-Instruct-4bit",
        "mlx-community/Qwen3-VL-2B-Instruct-4bit",
    ),
}
DEFAULT_MODEL_ID = "lfm3-vl-2b"

# The fixed knowledge base: exactly these 11 species (also the dataset species).
SPECIES_DOCUMENT_IDS = [
    "dog",
    "wolf",
    "fox",
    "coyote",
    "dingo",
    "cat",
    "koala",
    "red_panda",
    "raccoon",
    "bear",
    "hyena",
]

# Evaluation dataset size (spec §8) and the version tag of the result files.
DATASET_SIZE = 200
NON_DOG_COUNT = 100
RESULT_SCHEMA_VERSION = "paired"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


class ConfigError(RuntimeError):
    """The configuration file is missing or malformed."""


class VisionConfig(BaseModel):
    model_path: str
    model_repo: str | None = None  # Hugging Face repo used when the local path is absent
    model_id: str
    backend: str = "mlx-vlm"
    temperature: float = 0.0
    seed: int = 42
    max_tokens_first_pass: int = 512
    max_tokens_second_pass: int = 600

    def resolve_path(self) -> Path:
        return _resolve(self.model_path)


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    oversample: int = 3  # chunks fetched per wanted document before keeping the best per document
    index_dir: str = "vector_store"

    def resolve_index_dir(self) -> Path:
        return _resolve(self.index_dir)


class KnowledgeConfig(BaseModel):
    dir: str = "knowledge"
    chunk_max_chars: int = 1200

    def resolve_dir(self) -> Path:
        return _resolve(self.dir)


class DatasetConfig(BaseModel):
    manifest: str = "doggo-dataset/metadata/manifest.csv"
    images_dir: str = "doggo-dataset/images"

    def resolve_manifest(self) -> Path:
        return _resolve(self.manifest)

    def resolve_images_dir(self) -> Path:
        return _resolve(self.images_dir)


class ResultsConfig(BaseModel):
    dir: str = "results"

    def resolve_dir(self) -> Path:
        return _resolve(self.dir)


class EvalConfig(BaseModel):
    sample_seed: int = 42  # picks the dogs of a fast-eval run; independent of the decoding seed


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8888
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    api_key: str = ""  # from DOGGO_API_KEY only; never put it in the YAML
    max_upload_mb: int = 20
    upload_max_age_hours: int = 24


class AppConfig(BaseModel):
    vision: VisionConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    config_path: str = str(DEFAULT_CONFIG)

    @property
    def prompts_dir(self) -> Path:
        return Path(self.config_path).parent / "prompts"

    def for_model(self, model_id: str) -> AppConfig:
        """Copy of this config with the vision model switched to a supported ID.

        Decoding settings are kept, so every model runs with the same settings.
        """
        if model_id not in MODEL_REGISTRY:
            raise ConfigError(f"Unknown vision model ID: {model_id!r}")
        cfg = self.model_copy(deep=True)
        cfg.vision.model_id = model_id
        cfg.vision.model_path, cfg.vision.model_repo = MODEL_REGISTRY[model_id]
        return cfg


def _env_overrides(raw: dict) -> dict:
    """Apply DOGGO_* environment overrides in place on the raw YAML dict."""
    mapping = {
        "DOGGO_EMBEDDING_MODEL": ("embedding", "model_name"),
        "DOGGO_TOP_K": ("retrieval", "top_k"),
        "DOGGO_KNOWLEDGE_DIR": ("knowledge", "dir"),
        "DOGGO_INDEX_DIR": ("retrieval", "index_dir"),
        "DOGGO_DATASET_MANIFEST": ("dataset", "manifest"),
        "DOGGO_API_KEY": ("server", "api_key"),
    }
    for env, (section, key) in mapping.items():
        value = os.environ.get(env)
        if not value:
            continue
        if key == "top_k":
            try:
                value = int(value)
            except ValueError as exc:
                raise ConfigError(f"{env} must be an integer, got {value!r}") from exc
        raw.setdefault(section, {})[key] = value
    return raw


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = _resolve(str(path)) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {config_path} must contain a mapping")
    raw = _env_overrides(raw)
    raw["config_path"] = str(config_path)
    try:
        return AppConfig(**raw)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {config_path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()
