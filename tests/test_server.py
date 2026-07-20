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

    def stream_pcm(self, req, cancel=None):
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
    service._model_loaded = True
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
            assert msg["state"] == "READY"
            assert {"ready", "state", "model_loaded", "audio_open", "last_error"} <= msg.keys()
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_ping_returns_pong():
    asyncio.run(_test_websocket_ping_returns_pong())


def test_status_is_starting_before_model_load():
    service = TtsWebSocketService(load_config())
    assert service.status()["state"] == "STARTING"


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


async def _test_websocket_client_close_after_flush_is_clean(caplog):
    server, url = await run_ws_server(FakeEngine())
    try:
        with caplog.at_level(logging.ERROR):
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d"}))
                await ws.send(json.dumps({"type": "flush"}))
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        continue
                    event = json.loads(msg)
                    if event["type"] == "flushed":
                        break
            await asyncio.sleep(0.05)
        failures = [
            record for record in caplog.records
            if record.name == "websockets.server" and "connection handler failed" in record.getMessage()
        ]
        assert not failures
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_client_close_after_flush_is_clean(caplog):
    asyncio.run(_test_websocket_client_close_after_flush_is_clean(caplog))


async def _test_websocket_start_uses_actual_chunk_sample_rate():
    class ActualRateEngine(FakeEngine):
        sample_rate = 16000

        def stream_pcm(self, req, cancel=None):
            assert req.text
            yield AudioChunk(b"\x01\x00\x02\x00", sample_rate=8000)

    server, url = await run_ws_server(ActualRateEngine())
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d"}))
            assert json.loads(await ws.recv())["type"] == "queued"
            started = json.loads(await ws.recv())
            assert started["type"] == "start"
            assert started["sample_rate"] == 8000
    finally:
        server.close()
        await server.wait_closed()


def test_websocket_start_uses_actual_chunk_sample_rate():
    asyncio.run(_test_websocket_start_uses_actual_chunk_sample_rate())


async def _test_websocket_accepts_next_text_while_audio_is_blocked():
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def stream_pcm(self, req, cancel=None):
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


async def _test_disconnect_cancels_worker_before_releasing_slot():
    canceled = threading.Event()
    finished = threading.Event()

    class CancelAwareEngine(FakeEngine):
        def stream_pcm(self, req, cancel=None):
            yield AudioChunk(b"first", sample_rate=16000)
            while cancel is not None and not cancel.is_set():
                time.sleep(0.01)
            canceled.set()
            time.sleep(0.1)
            finished.set()

    cfg = load_config()
    cfg.tts.max_concurrent_requests = 1
    cfg.tts.cancel_wait_sec = 0.02
    service = TtsWebSocketService(cfg)
    service.engine = CancelAwareEngine()
    server, url = await run_ws_service(service)
    try:
        ws = await websockets.connect(url)
        await ws.send(json.dumps({"type": "text", "text": "cancel me"}))
        assert json.loads(await ws.recv())["type"] == "queued"
        assert json.loads(await ws.recv())["type"] == "start"
        assert await ws.recv() == b"first"
        await ws.close()
        assert await asyncio.to_thread(canceled.wait, 1.0)
        assert service._request_slots.acquire(blocking=False) is False
        assert await asyncio.to_thread(finished.wait, 1.0)
        assert service._request_slots.acquire(blocking=False) is True
        service._request_slots.release()
    finally:
        server.close()
        await server.wait_closed()


def test_disconnect_cancels_worker_before_releasing_slot():
    asyncio.run(_test_disconnect_cancels_worker_before_releasing_slot())


async def _test_websocket_synthesis_error_returns_json_error():
    class FailingEngine(FakeEngine):
        def stream_pcm(self, req, cancel=None):
            raise RuntimeError("boom")

    server, url = await run_ws_server(FailingEngine())
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "text", "text": "\u4f60\u597d"}))
            assert json.loads(await ws.recv())["type"] == "queued"
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
        def stream_pcm(self, req, cancel=None):
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
        def stream_pcm(self, req, cancel=None):
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
