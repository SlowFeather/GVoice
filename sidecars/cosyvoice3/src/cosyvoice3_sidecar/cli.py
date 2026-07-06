from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .config import load_config

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(cfg) -> None:
    handlers: list[logging.Handler] = []
    if cfg.logging.console:
        handlers.append(logging.StreamHandler())
    if cfg.logging.file:
        path = Path(cfg.logging.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)


def _apply_overrides(cfg, args):
    if getattr(args, "log_level", None):
        cfg.logging.level = args.log_level
    if getattr(args, "log_file", None):
        cfg.logging.file = args.log_file
    if getattr(args, "host", None):
        cfg.server.host = args.host
    if getattr(args, "port", None):
        cfg.server.port = args.port
    if getattr(args, "model", None):
        cfg.model.name = args.model
        cfg.model.model_id = ""
        cfg.model.model_dir = ""
    if getattr(args, "model_dir", None):
        cfg.model.model_dir = args.model_dir
    if getattr(args, "prompt_wav", None):
        cfg.prompt.wav = args.prompt_wav
    if getattr(args, "prompt_text", None) is not None:
        cfg.prompt.text = args.prompt_text
    if getattr(args, "no_fp16", False):
        cfg.model.fp16 = False
    return cfg


def _add_common_args(parser):
    parser.add_argument("--model", default=None, help="0.5b / 1.5b 或完整模型 id")
    parser.add_argument("--model-dir", default=None, help="已下载好的模型目录")
    parser.add_argument("--prompt-wav", default=None, help="参考音频路径（决定音色）")
    parser.add_argument("--prompt-text", default=None, help="参考音频转写文本；留空走 cross_lingual")
    parser.add_argument("--no-fp16", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosyvoice3-sidecar")
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=None)
    parser.add_argument("--log-file", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="start the CosyVoice3 WebSocket sidecar")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    _add_common_args(sp)

    sp = sub.add_parser("tts", help="synthesize text to a WAV file (debug)")
    sp.add_argument("text")
    sp.add_argument("--out", default="output.wav")
    sp.add_argument("--speed", type=float, default=None)
    _add_common_args(sp)

    sp = sub.add_parser("download", help="download the configured model")
    _add_common_args(sp)

    return parser


def cmd_serve(args) -> int:
    from .server import run_service

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    try:
        run_service(cfg)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("CosyVoice3 sidecar stopped by keyboard interrupt")
    return 0


def cmd_tts(args) -> int:
    import io
    import wave

    from .engine import CosyVoice3Engine

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    engine = CosyVoice3Engine(cfg)
    engine.load()
    pcm = bytearray()
    for chunk in engine.stream(args.text, speed=args.speed, _skip_ready_check=True):
        pcm.extend(chunk)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(engine.sample_rate)
        wav.writeframes(bytes(pcm))
    out.write_bytes(buf.getvalue())
    print(out)
    return 0


def cmd_download(args) -> int:
    from .engine import CosyVoice3Engine

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    engine = CosyVoice3Engine(cfg)
    model_dir = engine._ensure_model()
    print(model_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "tts":
        return cmd_tts(args)
    if args.command == "download":
        return cmd_download(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
