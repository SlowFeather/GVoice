from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# 已知模型别名 -> ModelScope/HuggingFace 模型 id。
# 1.5B 官方尚未放出开源权重，放出后如 id 不同可直接在 model.model_id 填完整 id。
MODEL_ALIASES: dict[str, str] = {
    "0.5b": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "cosyvoice3-0.5b": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "1.5b": "FunAudioLLM/Fun-CosyVoice3-1.5B",
    "cosyvoice3-1.5b": "FunAudioLLM/Fun-CosyVoice3-1.5B",
}


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8788
    path: str = "/v1/cosyvoice3/ws"


@dataclass
class ModelConfig:
    # 别名（0.5b / 1.5b）或完整模型 id（含 "/"）
    name: str = "cosyvoice3-0.5b"
    # 直接指定模型 id 时优先于 name
    model_id: str = ""
    # 已下载好的模型目录；留空时自动下载到 models_dir 下
    model_dir: str = ""
    models_dir: str = "models"
    download_source: str = "modelscope"  # modelscope | huggingface
    fp16: bool = True
    load_trt: bool = False
    load_vllm: bool = False
    # 加载完成后合成一小段文本预热 CUDA kernel，降低首句延迟
    warmup: bool = True
    warmup_text: str = "你好。"


@dataclass
class PromptConfig:
    # 参考音频（决定音色）
    wav: str = ""
    # 参考音频的逐字转写。填了走 zero_shot 模式（音色相似度更高）；
    # 留空走 cross_lingual 模式（不需要转写文本）。
    text: str = ""
    # instruct 指令（可选，如“请用四川话表达”），走 instruct2 模式
    instruct: str = ""
    # CosyVoice3 的系统前缀（自动拼 <|endofprompt|>），v2 及更早模型忽略
    system: str = "You are a helpful assistant."


@dataclass
class TtsConfig:
    speed: float = 1.0
    # 走 CosyVoice 自带文本正规化（数字、单位等）；异常时可关掉排查
    text_frontend: bool = True
    # 首个流式块的 token 数（25 token = 1 秒音频）。上游在流式过程中会把 hop 翻倍以提升
    # 质量但不跨请求复位（bug），sidecar 每次合成前都会按此值复位。调小可进一步压首包
    # 延迟，但块间衔接质量可能略降；0 表示用模型默认值
    first_chunk_tokens: int = 25


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = ""
    console: bool = True


@dataclass
class Config:
    # CosyVoice 官方仓库位置（git clone --recursive），代码经 sys.path 引入
    cosyvoice_repo: str = "vendor/CosyVoice"
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_model_id(self) -> str:
        if self.model.model_id:
            return self.model.model_id
        name = self.model.name.strip()
        if "/" in name:
            return name
        alias = MODEL_ALIASES.get(name.lower())
        if alias is None:
            known = ", ".join(sorted(set(MODEL_ALIASES)))
            raise ValueError(f"unknown model name {name!r}; use one of: {known} or a full model id")
        return alias

    def resolve_model_dir(self) -> Path:
        if self.model.model_dir:
            return Path(self.model.model_dir)
        model_id = self.resolve_model_id()
        return Path(self.model.models_dir) / model_id.split("/")[-1]


def _merge(dc: Any, overrides: dict) -> None:
    valid = {f.name for f in fields(dc)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif value is not None:
            setattr(dc, key, value)


def validate_config(cfg: Config) -> None:
    if not 1 <= cfg.server.port <= 65535:
        raise ValueError(f"server.port must be 1..65535, got {cfg.server.port}")
    if not cfg.server.path.startswith("/"):
        raise ValueError("server.path must start with /")
    if cfg.model.download_source not in {"modelscope", "huggingface"}:
        raise ValueError("model.download_source must be modelscope or huggingface")
    if cfg.tts.speed <= 0:
        raise ValueError(f"tts.speed must be positive, got {cfg.tts.speed}")
    cfg.resolve_model_id()


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    validate_config(cfg)
    return cfg
