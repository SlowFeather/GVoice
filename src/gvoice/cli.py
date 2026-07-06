from __future__ import annotations

import argparse
import logging
import logging.handlers
from pathlib import Path
import sys

from .config import load_config


# 与 ChatCaht 全家统一的格式：2026-07-06 10:09:29,554 INFO gvoice.server: 消息
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _apply_overrides(cfg, args):
    if getattr(args, "log_level", None):
        cfg.logging.level = args.log_level
    if getattr(args, "log_file", None):
        cfg.logging.file = args.log_file
    if getattr(args, "host", None):
        cfg.tts.host = args.host
        cfg.tts.ws_host = args.host
    if getattr(args, "port", None):
        cfg.tts.ws_port = args.port
    if getattr(args, "backend", None):
        cfg.tts.backend = args.backend
    if getattr(args, "speaker_id", None) is not None:
        cfg.tts.speaker_id = args.speaker_id
    if getattr(args, "speed", None) is not None:
        cfg.tts.speed = args.speed
    if getattr(args, "noise_scale", None) is not None:
        cfg.tts.noise_scale = args.noise_scale
    if getattr(args, "noise_scale_w", None) is not None:
        cfg.tts.noise_scale_w = args.noise_scale_w
    if getattr(args, "length_scale", None) is not None:
        cfg.tts.length_scale = args.length_scale
    return cfg


def configure_logging(cfg) -> None:
    # stdout/stderr 切到 UTF-8：Windows 默认 GBK 会让中文乱码甚至抛
    # UnicodeEncodeError；stdout 还会被 ChatCaht 捕获成 service log（按 UTF-8 读）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    handlers: list[logging.Handler] = []
    if cfg.logging.console:
        handlers.append(logging.StreamHandler())
    if cfg.logging.file:
        path = Path(cfg.logging.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
        )
    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    logging.getLogger(__name__).debug(
        "Logging configured level=%s file=%s console=%s",
        cfg.logging.level.upper(),
        cfg.logging.file,
        cfg.logging.console,
    )


def _add_tts_tuning_args(parser):
    parser.add_argument("--backend", choices=["genshin_vits_onnx", "sherpa_onnx_vits", "cosyvoice3"], default=None)
    parser.add_argument("--speaker-id", type=int, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--noise-scale-w", type=float, default=None)
    parser.add_argument("--length-scale", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gvoice")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=None)
    parser.add_argument("--log-file", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="start streaming TTS WebSocket service")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    _add_tts_tuning_args(sp)

    sp = sub.add_parser("tts", help="synthesize text to WAV")
    sp.add_argument("text")
    sp.add_argument("--out", default=None)
    _add_tts_tuning_args(sp)
    sp.add_argument("--speaker", default=None, help="speaker profile name")

    sp = sub.add_parser("speakers", help="manage speaker profiles")
    speaker_sub = sp.add_subparsers(dest="speaker_command", required=True)
    speaker_sub.add_parser("list", help="list speaker profiles")

    sp_show = speaker_sub.add_parser("show", help="show one speaker profile")
    sp_show.add_argument("name")

    sp_init = speaker_sub.add_parser("init", help="create a sherpa speaker-id profile")
    sp_init.add_argument("name")
    sp_init.add_argument("--speaker-id", type=int, required=True)
    sp_init.add_argument("--speed", type=float, default=None)
    sp_init.add_argument("--notes", default="")

    sp_clone = speaker_sub.add_parser("clone", help="create a clone profile from reference audio")
    sp_clone.add_argument("name")
    sp_clone.add_argument("--backend", choices=["cosyvoice", "gpt_sovits", "openvoice"], default="cosyvoice")
    sp_clone.add_argument("--audio", action="append", required=True, help="reference audio; repeat for multiple files")
    sp_clone.add_argument("--reference-text", default=None)
    sp_clone.add_argument("--language", default="zh")
    sp_clone.add_argument("--notes", default="")

    return parser


def cmd_serve(args) -> int:
    from .server import run_service

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    logger = logging.getLogger(__name__)
    try:
        run_service(cfg)
    except KeyboardInterrupt:
        logger.info("GVoice stopped by keyboard interrupt")
    return 0


def cmd_tts(args) -> int:
    from .engine import TtsEngine, TtsRequest
    from .speakers import apply_profile

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    logger = logging.getLogger(__name__)
    speaker_id = args.speaker_id
    speed = args.speed
    if args.speaker:
        speaker_id, speed = apply_profile(cfg, args.speaker, speaker_id=speaker_id, speed=speed)
    out = Path(args.out) if args.out else cfg.fs.output_dir / "speech.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = TtsEngine(cfg).synthesize_wav(
        TtsRequest(
            args.text,
            speaker_id=speaker_id,
            speed=speed,
            speaker=args.speaker,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )
    )
    out.write_bytes(data)
    logger.info("Wrote synthesized WAV path=%s bytes=%d", out, len(data))
    print(out)
    return 0


def cmd_speakers(args) -> int:
    import json

    from .speakers import create_clone_profile, create_sherpa_profile, list_profiles, load_profile

    cfg = _apply_overrides(load_config(args.config), args)
    configure_logging(cfg)
    logger = logging.getLogger(__name__)
    cfg.fs.ensure_dirs()
    if args.speaker_command == "list":
        for name in list_profiles(cfg):
            print(name)
        return 0
    if args.speaker_command == "show":
        print(json.dumps(load_profile(cfg, args.name).__dict__, ensure_ascii=False, indent=2))
        return 0
    if args.speaker_command == "init":
        profile = create_sherpa_profile(
            cfg,
            args.name,
            speaker_id=args.speaker_id,
            speed=args.speed,
            notes=args.notes,
        )
        logger.info("Created speaker profile name=%s backend=%s", profile.name, profile.backend)
        print(json.dumps(profile.__dict__, ensure_ascii=False, indent=2))
        return 0
    if args.speaker_command == "clone":
        profile = create_clone_profile(
            cfg,
            args.name,
            backend=args.backend,
            audio_paths=args.audio,
            reference_text=args.reference_text,
            language=args.language,
            notes=args.notes,
        )
        logger.info("Created clone profile name=%s backend=%s references=%d", profile.name, profile.backend, len(profile.reference_audio))
        print(json.dumps(profile.__dict__, ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"unknown speakers command: {args.speaker_command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "tts":
        return cmd_tts(args)
    if args.command == "speakers":
        return cmd_speakers(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
