from __future__ import annotations

import argparse
import wave
from pathlib import Path

from gvoice.config import load_config
from gvoice.engine import TtsEngine, TtsRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one local GVoice WAV and print basic audio metadata.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--text", default="\u4f60\u597d\uff0c\u6211\u662f\u672c\u5730\u8bed\u97f3\u52a9\u624b\u3002")
    parser.add_argument("--out", default="artifacts/output/smoke_tts.wav")
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wav_data = TtsEngine(cfg).synthesize_wav(TtsRequest(args.text, speed=args.speed))
    out.write_bytes(wav_data)

    with wave.open(str(out), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
    duration = frames / sample_rate if sample_rate else 0
    print(
        f"{out} rate={sample_rate}Hz channels={channels} "
        f"width={sample_width} frames={frames} duration={duration:.2f}s bytes={out.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
