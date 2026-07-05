from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


TTS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "tts-models/vits-zh-aishell3.tar.bz2"
)


@dataclass
class PathsConfig:
    base_dir: str = "artifacts"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None
    console: bool = True


@dataclass
class TtsConfig:
    host: str = "127.0.0.1"
    ws_host: str | None = None
    ws_port: int = 8787
    ws_path: str = "/v1/tts/ws"
    backend: str = "genshin_vits_onnx"
    sample_rate: int = 22050
    speaker_id: int = 115
    speed: float = 0.85
    noise_scale: float = 0.6
    noise_scale_w: float = 0.668
    length_scale: float = 1.2
    num_threads: int = 4
    stream_chunk_ms: int = 120
    max_concurrent_requests: int = 1
    queue_timeout_sec: float = 5.0
    model_url: str = TTS_MODEL_URL
    model_sha256: str | None = None
    genshin_model_path: str = "artifacts/models/vits-models-genshin-bh3/keqing/keqing_tunable.onnx"
    genshin_source_dir: str = "artifacts/sources/vits-models-genshin-bh3"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    @property
    def fs(self) -> "Paths":
        return Paths(self.paths)


class Paths:
    def __init__(self, cfg: PathsConfig):
        self.base = Path(cfg.base_dir)

    @property
    def tts_dir(self) -> Path:
        return self.base / "tts" / "vits-zh-aishell3"

    @property
    def tts_archive(self) -> Path:
        return self.base / "tts" / "vits-zh-aishell3.tar.bz2"

    @property
    def output_dir(self) -> Path:
        return self.base / "output"

    @property
    def speakers_dir(self) -> Path:
        return self.base / "speakers"

    def ensure_dirs(self) -> None:
        self.tts_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.speakers_dir.mkdir(parents=True, exist_ok=True)


def _merge(dc: Any, overrides: dict) -> None:
    valid = {f.name for f in fields(dc)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def validate_config(cfg: Config) -> None:
    level = cfg.logging.level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("logging.level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    if not 1 <= cfg.tts.ws_port <= 65535:
        raise ValueError(f"tts.ws_port must be 1..65535, got {cfg.tts.ws_port}")
    if not cfg.tts.ws_path.startswith("/"):
        raise ValueError("tts.ws_path must start with /")
    if cfg.tts.backend not in {"sherpa_onnx_vits", "genshin_vits_onnx"}:
        raise ValueError("tts.backend must be one of: sherpa_onnx_vits, genshin_vits_onnx")
    _require_positive("tts.sample_rate", cfg.tts.sample_rate)
    if cfg.tts.speaker_id < 0:
        raise ValueError("tts.speaker_id must be >= 0")
    _require_positive("tts.speed", cfg.tts.speed)
    _require_non_negative("tts.noise_scale", cfg.tts.noise_scale)
    _require_non_negative("tts.noise_scale_w", cfg.tts.noise_scale_w)
    _require_positive("tts.length_scale", cfg.tts.length_scale)
    _require_positive("tts.num_threads", cfg.tts.num_threads)
    _require_positive("tts.stream_chunk_ms", cfg.tts.stream_chunk_ms)
    _require_positive("tts.max_concurrent_requests", cfg.tts.max_concurrent_requests)
    _require_non_negative("tts.queue_timeout_sec", cfg.tts.queue_timeout_sec)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    validate_config(cfg)
    return cfg
