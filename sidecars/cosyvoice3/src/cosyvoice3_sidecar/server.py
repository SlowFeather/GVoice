from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .config import Config
from .engine import CosyVoice3Engine, EngineNotReady

logger = logging.getLogger(__name__)

# 模型未就绪时，synthesize 请求最多等待模型加载的秒数（含首次下载）
MODEL_READY_TIMEOUT_SEC = 600.0

_CLOSE = object()


class _Job:
    def __init__(self, rid: Any, text: str, speed: float | None):
        self.rid = rid
        self.text = text
        self.speed = speed
        self.cancel = threading.Event()


class CosyVoice3Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = CosyVoice3Engine(cfg)
        self._stop = asyncio.Event()

    async def serve_forever(self) -> None:
        self.engine.start_loading()
        async with websockets.serve(self.handler, self.cfg.server.host, self.cfg.server.port, max_size=None):
            logger.info(
                "CosyVoice3 sidecar listening url=ws://%s:%s%s model=%s",
                self.cfg.server.host,
                self.cfg.server.port,
                self.cfg.server.path,
                self.cfg.resolve_model_id(),
            )
            await self._stop.wait()
            logger.info("CosyVoice3 sidecar shutting down")

    async def handler(self, websocket) -> None:
        path = self._path(websocket)
        client = self._client_name(websocket)
        if path != self.cfg.server.path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return
        logger.info("WebSocket connected client=%s", client)

        queue: asyncio.Queue[_Job | object] = asyncio.Queue()
        current: dict[str, _Job | None] = {"job": None}
        producer = asyncio.create_task(self._receive(websocket, queue, current))
        consumer = asyncio.create_task(self._synthesize_loop(websocket, queue, current))
        try:
            await consumer
        finally:
            job = current["job"]
            if job is not None:
                job.cancel.set()
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        logger.info("WebSocket disconnected client=%s", client)

    async def _receive(self, websocket, queue: asyncio.Queue, current: dict) -> None:
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._send_json(websocket, {"type": "error", "error": "binary input is not supported"})
                    continue
                try:
                    payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("message must be a JSON object")
                    msg_type = str(payload.get("type") or "")
                    if msg_type == "ping":
                        await self._send_json(
                            websocket,
                            {
                                "type": "pong",
                                "ready": self.engine.ready,
                                "state": self.engine.state,
                                "model_loaded": self.engine.ready,
                                "last_error": self.engine.last_error,
                                "model": self.cfg.resolve_model_id(),
                                "sample_rate": self.engine.sample_rate,
                            },
                        )
                    elif msg_type == "synthesize":
                        text = str(payload.get("text") or "").strip()
                        if not text:
                            raise ValueError("text is required")
                        speed = payload.get("speed")
                        job = _Job(payload.get("id"), text, None if speed in {None, ""} else float(speed))
                        await queue.put(job)
                    elif msg_type == "cancel":
                        rid = payload.get("id")
                        await self._cancel_jobs(websocket, queue, current, rid)
                    elif msg_type == "shutdown":
                        await self._send_json(websocket, {"type": "bye"})
                        job = current["job"]
                        if job is not None:
                            job.cancel.set()
                        self._stop.set()
                        return
                    elif msg_type == "close":
                        await queue.put(_CLOSE)
                        return
                    else:
                        raise ValueError(f"unsupported message type: {msg_type!r}")
                except Exception as exc:
                    logger.warning("Invalid message error=%s", exc)
                    await self._send_json(websocket, {"type": "error", "error": str(exc)})
        except ConnectionClosed:
            pass
        finally:
            await queue.put(_CLOSE)

    async def _cancel_jobs(self, websocket, queue: asyncio.Queue, current: dict, rid: Any) -> None:
        # 取消当前任务：cancelled 帧由 _run_job 在音频流真正停止后发出
        running = False
        job = current["job"]
        if job is not None and (rid is None or job.rid == rid):
            job.cancel.set()
            running = True
        # 排队中尚未开始的任务直接应答 cancelled
        pending: list[_Job | object] = []
        matched_queued = False
        while not queue.empty():
            item = queue.get_nowait()
            if isinstance(item, _Job) and (rid is None or item.rid == rid):
                item.cancel.set()
                matched_queued = True
                await self._send_json(websocket, {"type": "cancelled", "id": item.rid})
            else:
                pending.append(item)
        for item in pending:
            queue.put_nowait(item)
        if not running and not matched_queued:
            await self._send_json(websocket, {"type": "cancelled", "id": rid})

    async def _synthesize_loop(self, websocket, queue: asyncio.Queue, current: dict) -> None:
        while True:
            item = await queue.get()
            if item is _CLOSE:
                return
            job = item
            assert isinstance(job, _Job)
            if job.cancel.is_set():
                continue
            current["job"] = job
            try:
                await self._run_job(websocket, job)
            except ConnectionClosed:
                job.cancel.set()
                return
            except Exception as exc:
                logger.exception("Synthesis failed id=%s", job.rid)
                try:
                    await self._send_json(websocket, {"type": "error", "id": job.rid, "error": str(exc)})
                except ConnectionClosed:
                    return
            finally:
                current["job"] = None

    async def _run_job(self, websocket, job: _Job) -> None:
        try:
            await asyncio.to_thread(self.engine.wait_ready, MODEL_READY_TIMEOUT_SEC)
        except EngineNotReady as exc:
            await self._send_json(websocket, {"type": "error", "id": job.rid, "error": str(exc)})
            return

        started = time.perf_counter()
        first_chunk_ms: float | None = None
        chunks = 0
        total_bytes = 0
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

        def worker() -> None:
            try:
                for pcm in self.engine.stream(job.text, speed=job.speed, cancel=job.cancel):
                    loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
            except BaseException as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        thread = threading.Thread(target=worker, name="cosyvoice3-synth", daemon=True)
        thread.start()

        await self._send_json(
            websocket,
            {
                "type": "start",
                "id": job.rid,
                "sample_rate": self.engine.sample_rate,
                "format": "pcm_s16le",
                "channels": 1,
            },
        )
        try:
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                if first_chunk_ms is None:
                    first_chunk_ms = round((time.perf_counter() - started) * 1000, 1)
                await websocket.send(item)
                chunks += 1
                total_bytes += len(item)
        except BaseException:
            job.cancel.set()  # 发送失败/连接断开时停止合成线程
            raise

        cancelled = job.cancel.is_set()
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        if cancelled:
            await self._send_json(websocket, {"type": "cancelled", "id": job.rid})
        else:
            await self._send_json(
                websocket,
                {
                    "type": "end",
                    "id": job.rid,
                    "chunks": chunks,
                    "bytes": total_bytes,
                    "first_chunk_ms": first_chunk_ms,
                    "duration_ms": duration_ms,
                },
            )
        logger.info(
            "Synthesis %s id=%s text_chars=%d chunks=%d bytes=%d first_chunk_ms=%s duration_ms=%.1f",
            "cancelled" if cancelled else "complete",
            job.rid,
            len(job.text),
            chunks,
            total_bytes,
            first_chunk_ms,
            duration_ms,
        )

    async def _send_json(self, websocket, message: dict[str, Any]) -> None:
        await websocket.send(json.dumps(message, ensure_ascii=False))

    def _client_name(self, websocket) -> str:
        remote = websocket.remote_address
        if isinstance(remote, tuple) and len(remote) >= 2:
            return f"{remote[0]}:{remote[1]}"
        return str(remote)

    def _path(self, websocket) -> str | None:
        path = getattr(websocket, "path", None)
        if path is not None:
            return path
        request = getattr(websocket, "request", None)
        return getattr(request, "path", None)


def run_service(cfg: Config) -> None:
    asyncio.run(CosyVoice3Service(cfg).serve_forever())
