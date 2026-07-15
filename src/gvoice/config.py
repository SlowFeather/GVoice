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
class Cosyvoice3Config:
    # sidecar WebSocket 地址（autostart 时 host/port 也从这里解析）
    url: str = "ws://127.0.0.1:8788/v1/cosyvoice3/ws"
    # 连不上 sidecar 时自动拉起（uv run cosyvoice3-sidecar serve）
    autostart: bool = True
    sidecar_dir: str = "sidecars/cosyvoice3"
    uv_executable: str = "uv"
    # 模型：0.5b（默认）/ 1.5b / 完整模型 id；留空用 sidecar 自己的配置。
    # autostart 时作为 --model 传给 sidecar；已运行的 sidecar 模型不一致时自动重启切换
    model: str = "0.5b"
    # 参考音频与其转写（决定音色）；留空用 sidecar 自己的配置
    prompt_wav: str = ""
    prompt_text: str | None = None
    # 语速；None 用 sidecar 默认（注意与 tts.speed 无关，那是 VITS 后端的）
    speed: float | None = None
    connect_timeout_sec: float = 10.0
    autostart_wait_sec: float = 60.0
    # 等首个音频帧的超时；首次运行包含模型下载+加载，故很长
    start_timeout_sec: float = 1800.0
    chunk_timeout_sec: float = 60.0
    restart_on_model_mismatch: bool = True
    log_file: str = "artifacts/logs/cosyvoice3.sidecar.log"


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
    max_pending_text_requests: int = 8
    pcm_queue_chunks: int = 16
    cancel_wait_sec: float = 3.0
    max_text_chars: int = 2000
    model_url: str = TTS_MODEL_URL
    model_sha256: str | None = None
    genshin_model_path: str = "artifacts/models/vits-models-genshin-bh3/keqing/keqing_tunable.onnx"
    genshin_source_dir: str = "artifacts/sources/vits-models-genshin-bh3"
    cosyvoice3: Cosyvoice3Config = field(default_factory=Cosyvoice3Config)


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
    if cfg.tts.backend not in {"sherpa_onnx_vits", "genshin_vits_onnx", "cosyvoice3"}:
        raise ValueError("tts.backend must be one of: sherpa_onnx_vits, genshin_vits_onnx, cosyvoice3")
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
    _require_positive("tts.max_pending_text_requests", cfg.tts.max_pending_text_requests)
    _require_positive("tts.pcm_queue_chunks", cfg.tts.pcm_queue_chunks)
    _require_positive("tts.cancel_wait_sec", cfg.tts.cancel_wait_sec)
    _require_positive("tts.max_text_chars", cfg.tts.max_text_chars)
    c3 = cfg.tts.cosyvoice3
    if not c3.url.startswith(("ws://", "wss://")):
        raise ValueError("tts.cosyvoice3.url must start with ws:// or wss://")
    _require_positive("tts.cosyvoice3.connect_timeout_sec", c3.connect_timeout_sec)
    _require_positive("tts.cosyvoice3.autostart_wait_sec", c3.autostart_wait_sec)
    _require_positive("tts.cosyvoice3.start_timeout_sec", c3.start_timeout_sec)
    _require_positive("tts.cosyvoice3.chunk_timeout_sec", c3.chunk_timeout_sec)
    if c3.speed is not None:
        _require_positive("tts.cosyvoice3.speed", c3.speed)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    validate_config(cfg)
    return cfg
