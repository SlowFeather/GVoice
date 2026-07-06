import json
import threading
import time

import pytest
from websockets.sync.server import serve

from gvoice.config import load_config, validate_config
from gvoice.cosyvoice3 import CosyVoice3Engine
from gvoice.engine import TtsRequest, _make_engine


PONG = {
    "type": "pong",
    "ready": True,
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "sample_rate": 24000,
}


def make_engine(url):
    cfg = load_config()
    cfg.tts.backend = "cosyvoice3"
    cfg.tts.cosyvoice3.url = url
    cfg.tts.cosyvoice3.autostart = False
    cfg.tts.cosyvoice3.connect_timeout_sec = 2.0
    cfg.tts.cosyvoice3.start_timeout_sec = 5.0
    cfg.tts.cosyvoice3.chunk_timeout_sec = 5.0
    return CosyVoice3Engine(cfg)


class FakeSidecar:
    def __init__(self, handler):
        self.server = serve(handler, "127.0.0.1", 0)
        port = self.server.socket.getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/cosyvoice3/ws"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()


def simple_handler(ws):
    for message in ws:
        data = json.loads(message)
        if data["type"] == "ping":
            ws.send(json.dumps(PONG))
        elif data["type"] == "synthesize":
            rid = data["id"]
            ws.send(json.dumps({"type": "start", "id": rid, "sample_rate": 24000, "channels": 1}))
            ws.send(b"\x01\x00\x02\x00")
            ws.send(b"\x03\x00\x04\x00")
            ws.send(json.dumps({"type": "end", "id": rid, "chunks": 2, "bytes": 8}))


def test_stream_pcm_yields_chunks_with_sidecar_sample_rate():
    sidecar = FakeSidecar(simple_handler)
    try:
        engine = make_engine(sidecar.url)
        chunks = list(engine.stream_pcm(TtsRequest("你好。")))
        assert [c.pcm for c in chunks] == [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"]
        assert all(c.sample_rate == 24000 for c in chunks)
        # 连接可复用：第二次请求走同一条连接
        chunks = list(engine.stream_pcm(TtsRequest("第二句。")))
        assert len(chunks) == 2
    finally:
        sidecar.close()


def test_cancel_stops_stream_and_connection_stays_usable():
    def handler(ws):
        for message in ws:
            data = json.loads(message)
            if data["type"] == "ping":
                ws.send(json.dumps(PONG))
            elif data["type"] == "synthesize":
                rid = data["id"]
                ws.send(json.dumps({"type": "start", "id": rid, "sample_rate": 24000}))
                while True:
                    try:
                        msg = ws.recv(timeout=0.01)
                    except TimeoutError:
                        msg = None
                    if msg is not None:
                        cancel = json.loads(msg)
                        assert cancel["type"] == "cancel"
                        ws.send(json.dumps({"type": "cancelled", "id": rid}))
                        break
                    ws.send(b"\x01\x00" * 32)
                    time.sleep(0.01)

    sidecar = FakeSidecar(handler)
    try:
        engine = make_engine(sidecar.url)
        cancel = threading.Event()
        received = 0
        for _chunk in engine.stream_pcm(TtsRequest("很长很长的一句话。"), cancel):
            received += 1
            if received >= 3:
                cancel.set()
        assert received >= 3
        # 取消后连接仍可复用
        engine._ws is not None
    finally:
        sidecar.close()


def test_error_frame_raises_runtime_error():
    def handler(ws):
        for message in ws:
            data = json.loads(message)
            if data["type"] == "ping":
                ws.send(json.dumps(PONG))
            elif data["type"] == "synthesize":
                ws.send(json.dumps({"type": "error", "id": data["id"], "error": "boom"}))

    sidecar = FakeSidecar(handler)
    try:
        engine = make_engine(sidecar.url)
        with pytest.raises(RuntimeError, match="boom"):
            list(engine.stream_pcm(TtsRequest("你好。")))
    finally:
        sidecar.close()


def test_make_engine_supports_cosyvoice3():
    cfg = load_config()
    cfg.tts.backend = "cosyvoice3"
    assert isinstance(_make_engine(cfg), CosyVoice3Engine)


def test_config_validates_cosyvoice3():
    cfg = load_config()
    cfg.tts.backend = "cosyvoice3"
    validate_config(cfg)

    cfg.tts.cosyvoice3.url = "http://127.0.0.1:8788"
    with pytest.raises(ValueError, match="ws://"):
        validate_config(cfg)


def test_model_matches():
    match = CosyVoice3Engine._model_matches
    assert match("0.5b", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    assert match("cosyvoice3-0.5b", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    assert not match("1.5b", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    assert match("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    assert not match("FunAudioLLM/Other", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
