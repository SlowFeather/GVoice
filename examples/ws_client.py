from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets


async def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="text segments to synthesize")
    parser.add_argument("--url", default="ws://127.0.0.1:8787/v1/tts/ws")
    parser.add_argument("--out", default="stream.wav")
    args = parser.parse_args()

    sample_rate = 16000
    with wave.open(args.out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        async with websockets.connect(args.url) as ws:
            for text in args.text:
                await ws.send(json.dumps({"type": "text", "text": text}, ensure_ascii=False))
            await ws.send(json.dumps({"type": "flush"}))

            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    wav.writeframes(msg)
                    continue
                event = json.loads(msg)
                if event.get("type") == "start":
                    sample_rate = int(event["sample_rate"])
                    wav.setframerate(sample_rate)
                elif event.get("type") == "flushed":
                    break
                elif event.get("type") == "error":
                    raise RuntimeError(event["error"])

    print(args.out)
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
