from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .config import Config
from .engine import AudioChunk, TtsRequest, _BaseEngine

logger = logging.getLogger(__name__)

_CANCEL_DRAIN_SEC = 5.0


class CosyVoice3Engine(_BaseEngine):
    """经 sidecar 子进程（sidecars/cosyvoice3）合成的流式后端。

    - 与 sidecar 保持一条持久 WebSocket 连接，PCM 帧直通转发（sidecar 流式产出即转发，
      不做整句缓冲），首包延迟取决于 sidecar 的 GPU 推理速度
    - cancel 事件触发时向 sidecar 发 cancel 并排空残留帧；连接异常时直接断开，
      sidecar 侧检测到断开会自动停止合成
    - autostart 打开时自动拉起 sidecar；sidecar 常驻，GVoice 重启无需重新加载模型
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._ws = None
        self._lock = threading.Lock()
        self._rid = 0
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def _c3(self):
        return self.cfg.tts.cosyvoice3

    # ---- 连接管理 ----

    def load(self) -> None:
        with self._lock:
            ws = self._ensure_ws()
            self._wait_ready(ws, self._c3.start_timeout_sec)

    def _ensure_ws(self):
        if self._ws is not None:
            return self._ws
        ws = self._try_connect(self._c3.connect_timeout_sec)
        if ws is None and self._c3.autostart:
            self._spawn_sidecar()
            ws = self._wait_connect(self._c3.autostart_wait_sec)
        if ws is None:
            raise RuntimeError(
                f"cannot reach cosyvoice3 sidecar at {self._c3.url}; "
                "start it manually or enable tts.cosyvoice3.autostart"
            )
        ws = self._check_model(ws)
        self._ws = ws
        return ws

    def _try_connect(self, timeout: float):
        try:
            return connect(self._c3.url, open_timeout=timeout, max_size=None)
        except Exception:
            return None

    def _wait_connect(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ws = self._try_connect(2.0)
            if ws is not None:
                return ws
            time.sleep(0.5)
        return None

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _ping(self, ws) -> dict[str, Any]:
        ws.send(json.dumps({"type": "ping"}))
        deadline = time.monotonic() + self._c3.connect_timeout_sec
        while time.monotonic() < deadline:
            msg = ws.recv(timeout=max(0.1, deadline - time.monotonic()))
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            if data.get("type") == "pong":
                return data
        raise RuntimeError("cosyvoice3 sidecar did not answer ping")

    def _wait_ready(self, ws, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pong = self._ping(ws)
            self._sample_rate = int(pong.get("sample_rate") or self._sample_rate)
            if pong.get("ready"):
                return
            if str(pong.get("state") or "").upper() == "FAILED":
                raise RuntimeError(str(pong.get("last_error") or "cosyvoice3 model load failed"))
            time.sleep(0.5)
        raise RuntimeError(f"cosyvoice3 model warmup timed out after {timeout:.0f}s")

    def _check_model(self, ws):
        pong = self._ping(ws)
        self._sample_rate = int(pong.get("sample_rate") or self._sample_rate)
        wanted = (self._c3.model or "").strip()
        actual = str(pong.get("model") or "")
        if not wanted or self._model_matches(wanted, actual):
            return ws
        if not (self._c3.autostart and self._c3.restart_on_model_mismatch):
            logger.warning(
                "cosyvoice3 sidecar runs model %s but config wants %s; keeping running model",
                actual,
                wanted,
            )
            return ws
        logger.info("Restarting cosyvoice3 sidecar to switch model %s -> %s", actual, wanted)
        try:
            ws.send(json.dumps({"type": "shutdown"}))
            ws.close()
        except Exception:
            pass
        time.sleep(1.0)
        self._spawn_sidecar()
        ws = self._wait_connect(self._c3.autostart_wait_sec)
        if ws is None:
            raise RuntimeError("cosyvoice3 sidecar did not come back after model switch")
        pong = self._ping(ws)
        self._sample_rate = int(pong.get("sample_rate") or self._sample_rate)
        return ws

    @staticmethod
    def _model_matches(wanted: str, actual: str) -> bool:
        wanted_l = wanted.lower()
        actual_l = actual.lower()
        if "/" in wanted_l:
            return wanted_l == actual_l
        return wanted_l.replace("cosyvoice3-", "") in actual_l

    def _spawn_sidecar(self) -> None:
        c3 = self._c3
        sidecar_dir = Path(c3.sidecar_dir)
        if not sidecar_dir.exists():
            raise FileNotFoundError(f"cosyvoice3 sidecar dir not found: {sidecar_dir.resolve()}")
        parsed = urlparse(c3.url)
        args = [c3.uv_executable, "run", "cosyvoice3-sidecar", "serve"]
        if parsed.hostname:
            args += ["--host", parsed.hostname]
        if parsed.port:
            args += ["--port", str(parsed.port)]
        if c3.model:
            args += ["--model", c3.model]
        if c3.prompt_wav:
            args += ["--prompt-wav", c3.prompt_wav]
        if c3.prompt_text is not None:
            args += ["--prompt-text", c3.prompt_text]
        log_path = Path(c3.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("ab")
        logger.info("Starting cosyvoice3 sidecar cmd=%s cwd=%s log=%s", " ".join(args), sidecar_dir, log_path)
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        try:
            subprocess.Popen(
                args,
                cwd=str(sidecar_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        finally:
            log_file.close()

    # ---- 合成 ----

    def synthesize_sentence(self, text: str, req: TtsRequest) -> AudioChunk:
        pcm = bytearray()
        sample_rate = self._sample_rate
        request = TtsRequest(text=text, speaker_id=req.speaker_id, speed=req.speed, speaker=req.speaker)
        for chunk in self.stream_pcm(request):
            pcm.extend(chunk.pcm)
            sample_rate = chunk.sample_rate
        return AudioChunk(bytes(pcm), sample_rate=sample_rate)

    def stream_pcm(self, req: TtsRequest, cancel: threading.Event | None = None) -> Iterable[AudioChunk]:
        text = req.text.strip()
        if not text:
            return
        with self._lock:
            yielded = False
            for attempt in (1, 2):
                ws = self._ensure_ws()
                try:
                    for chunk in self._stream_once(ws, text, req, cancel):
                        yielded = True
                        yield chunk
                    return
                except GeneratorExit:
                    # 消费方中途放弃：直接断开连接，sidecar 检测到断开即停止合成
                    self._close_ws()
                    raise
                except (ConnectionClosed, OSError) as exc:
                    self._close_ws()
                    if yielded or attempt == 2:
                        raise
                    logger.warning("cosyvoice3 sidecar connection lost (%s); reconnecting", exc)

    def _stream_once(self, ws, text: str, req: TtsRequest, cancel: threading.Event | None):
        self._rid += 1
        rid = self._rid
        payload: dict[str, Any] = {"type": "synthesize", "id": rid, "text": text}
        speed = req.speed if req.speed is not None else self._c3.speed
        if speed is not None:
            payload["speed"] = float(speed)
        ws.send(json.dumps(payload, ensure_ascii=False))

        deadline = time.monotonic() + self._c3.start_timeout_sec
        while True:
            if cancel is not None and cancel.is_set():
                self._cancel_request(ws, rid)
                return
            try:
                msg = ws.recv(timeout=0.1)
            except TimeoutError:
                if time.monotonic() > deadline:
                    self._close_ws()
                    raise RuntimeError(f"cosyvoice3 sidecar timed out for request id={rid}")
                continue
            if isinstance(msg, bytes):
                deadline = time.monotonic() + self._c3.chunk_timeout_sec
                yield AudioChunk(msg, sample_rate=self._sample_rate)
                continue
            data = json.loads(msg)
            typ = data.get("type")
            if typ == "start":
                self._sample_rate = int(data.get("sample_rate") or self._sample_rate)
            elif typ in {"end", "cancelled"} and data.get("id") == rid:
                return
            elif typ == "error":
                raise RuntimeError(f"cosyvoice3 synthesis failed: {data.get('error')}")

    def _cancel_request(self, ws, rid: int) -> None:
        """通知 sidecar 停止当前合成并排空残留帧，保持连接可复用。"""
        try:
            ws.send(json.dumps({"type": "cancel", "id": rid}))
            deadline = time.monotonic() + _CANCEL_DRAIN_SEC
            while time.monotonic() < deadline:
                try:
                    msg = ws.recv(timeout=0.2)
                except TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                if data.get("type") in {"cancelled", "end", "error"} and data.get("id") == rid:
                    return
            logger.warning("cosyvoice3 cancel drain timed out; dropping connection")
            self._close_ws()
        except Exception:
            self._close_ws()
