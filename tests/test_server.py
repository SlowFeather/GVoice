import asyncio
import json
import logging
import threading
import time

import websockets

from gvoice.cli import configure_logging
from gvoice.config import load_config
from gvoice.engine import AudioChunk
from gvoice.server import TtsWebSocketService, _request_from_dict
from gvoice.speakers import create_sherpa_profile


class FakeEngine:
    sample_rate = 16000

    def load(self):
        return None

    def stream_pcm(self, req):
        assert req.text
        yield AudioChunk(b"\x01\x00\x02\x00", sample_rate=16000)
        yield AudioChunk(b"\x03\x00\x04\x00", sample_rate=16000)

    def synthesize_wav(self, req):
        assert req.text
        return b"RIFFxxxxWAVEfmt "


async def run_ws_server(engine, cfg=None):
    cfg = cfg or load_config()
    cfg.tts.ws_port = 0
    service = TtsWebSocketService(cfg)
    service.engine = engine
    return await run_ws_service(service)


async def run_ws_service(service):
    server = await websockets.serve(service.handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}{service.cfg.tts.ws_path}"


async def _test_websocket_ping_returns_pong():
    server, url = await run_ws_server(FakeEngine())
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            msg = json.loads(await ws.recv())
            assert msg["type"] == "pong"
            assert msg["backend"]
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_ping_returns_pong():
    asyncio.run(_test_websocket_ping_returns_pong())


async def _test_websocket_text_returns_pcm_chunks_and_end():
    server, url = await run_ws_server(FakeEngine())
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d"}))
            queued = json.loads(await ws.recv())
            started = json.loads(await ws.recv())
            chunk1 = await ws.recv()
            chunk2 = await ws.recv()
            ended = json.loads(await ws.recv())

            assert queued["type"] == "queued"
            assert started["type"] == "start"
            assert started["sample_rate"] == 16000
            assert chunk1 == b"\x01\x00\x02\x00"
            assert chunk2 == b"\x03\x00\x04\x00"
            assert ended["type"] == "end"
            assert ended["chunks"] == 2
            assert ended["bytes"] == 8
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_text_returns_pcm_chunks_and_end():
    asyncio.run(_test_websocket_text_returns_pcm_chunks_and_end())


async def _test_websocket_accepts_next_text_while_audio_is_blocked():
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def stream_pcm(self, req):
            yield AudioChunk(req.text.encode("utf-8"), sample_rate=16000)
            release.wait(timeout=2)

    cfg = load_config()
    cfg.tts.max_concurrent_requests = 1
    server, url = await run_ws_server(BlockingEngine(), cfg)
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "A"}))
            await ws.send(json.dumps({"type": "text", "text": "B"}))
            msg1 = json.loads(await ws.recv())
            msg2 = json.loads(await ws.recv())
            start = json.loads(await ws.recv())
            chunk = await ws.recv()
            assert msg1["type"] == "queued"
            assert msg2["type"] == "queued"
            assert start["type"] == "start"
            assert chunk == b"A"
            release.set()
            end1 = json.loads(await ws.recv())
            start2 = json.loads(await ws.recv())
            chunk2 = await ws.recv()
            assert end1["type"] == "end"
            assert start2["type"] == "start"
            assert chunk2 == b"B"
    finally:
        release.set()
        server.close()
        await server.wait_closed()


def test_websocket_accepts_next_text_while_audio_is_blocked():
    asyncio.run(_test_websocket_accepts_next_text_while_audio_is_blocked())


async def _test_websocket_synthesis_error_returns_json_error():
    class FailingEngine(FakeEngine):
        def stream_pcm(self, req):
            raise RuntimeError("boom")

    server, url = await run_ws_server(FailingEngine())
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d"}))
            assert json.loads(await ws.recv())["type"] == "queued"
            assert json.loads(await ws.recv())["type"] == "start"
            err = json.loads(await ws.recv())
            assert err == {"type": "error", "error": "synthesis failed"}
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_synthesis_error_returns_json_error():
    asyncio.run(_test_websocket_synthesis_error_returns_json_error())


async def _test_websocket_concurrency_limit_returns_busy():
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def stream_pcm(self, req):
            started.set()
            release.wait(timeout=2)
            yield AudioChunk(b"done", sample_rate=16000)

    cfg = load_config()
    cfg.tts.max_concurrent_requests = 1
    cfg.tts.queue_timeout_sec = 0.05
    busy_service = TtsWebSocketService(cfg)
    busy_service.engine = BlockingEngine()

    acquired = busy_service._request_slots.acquire(blocking=False)
    assert acquired is True
    server, url = await run_ws_service(busy_service)
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u5fd9\u788c"}))
            assert json.loads(await ws.recv())["type"] == "queued"
            err = json.loads(await ws.recv())
            assert err["type"] == "error"
            assert err["error"] == "service busy"
    finally:
        busy_service._request_slots.release()
        release.set()
        server.close()
        await server.wait_closed()


def test_websocket_concurrency_limit_returns_busy():
    asyncio.run(_test_websocket_concurrency_limit_returns_busy())


async def _test_websocket_stream_resolves_speaker_profile(tmp_path):
    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    create_sherpa_profile(cfg, "demo", speaker_id=5, speed=1.3)
    seen = {}

    class ProfileEngine(FakeEngine):
        def stream_pcm(self, req):
            seen["speaker_id"] = req.speaker_id
            seen["speed"] = req.speed
            yield AudioChunk(b"\x01\x00", sample_rate=16000)

    server, url = await run_ws_server(ProfileEngine(), cfg)
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d", "speaker": "demo"}))
            assert json.loads(await ws.recv())["type"] == "queued"
            assert json.loads(await ws.recv())["type"] == "start"
            assert await ws.recv() == b"\x01\x00"
            assert json.loads(await ws.recv())["type"] == "end"
            assert seen == {"speaker_id": 5, "speed": 1.3}
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_stream_resolves_speaker_profile(tmp_path):
    asyncio.run(_test_websocket_stream_resolves_speaker_profile(tmp_path))


def test_request_parses_genshin_tuning_fields():
    req = _request_from_dict({
        "text": "\u4f60\u597d",
        "speaker_id": "115",
        "speed": "1.1",
        "noise_scale": "0.7",
        "noise_scale_w": "0.75",
        "length_scale": "1.15",
    })
    assert req.speaker_id == 115
    assert req.speed == 1.1
    assert req.noise_scale == 0.7
    assert req.noise_scale_w == 0.75
    assert req.length_scale == 1.15


def test_configure_logging_creates_log_file_parent(tmp_path):
    cfg = load_config()
    cfg.logging.console = False
    cfg.logging.file = str(tmp_path / "logs" / "gvoice.log")

    configure_logging(cfg)
    logging.getLogger("gvoice.test").info("hello")
    logging.shutdown()

    log_file = tmp_path / "logs" / "gvoice.log"
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")
