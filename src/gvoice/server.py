from __future__ import annotations

import asyncio
import json
import logging
import queue as sync_queue
import threading
import time
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .config import Config
from .engine import TtsEngine, TtsRequest
from .speakers import apply_profile


logger = logging.getLogger(__name__)

_FLUSH = object()
_CLOSE = object()


def _request_from_dict(data: dict[str, Any], default_text: str = "") -> TtsRequest:
    text = str(data.get("text") or default_text).strip()
    if not text:
        raise ValueError("text is required")
    speaker_id = data.get("speaker_id")
    speed = data.get("speed")
    speaker = data.get("speaker")
    noise_scale = data.get("noise_scale")
    noise_scale_w = data.get("noise_scale_w")
    length_scale = data.get("length_scale")
    return TtsRequest(
        text=text,
        speaker_id=None if speaker_id in {None, ""} else int(speaker_id),
        speed=None if speed in {None, ""} else float(speed),
        speaker=None if speaker in {None, ""} else str(speaker),
        noise_scale=None if noise_scale in {None, ""} else float(noise_scale),
        noise_scale_w=None if noise_scale_w in {None, ""} else float(noise_scale_w),
        length_scale=None if length_scale in {None, ""} else float(length_scale),
    )


class TtsWebSocketService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = TtsEngine(cfg)
        self._request_slots = threading.BoundedSemaphore(int(cfg.tts.max_concurrent_requests))
        self._jobs_lock = threading.Lock()
        self._active_jobs = 0
        self._model_loaded = False
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self._jobs_lock:
            active_jobs = self._active_jobs
        return {
            "type": "status",
            "ready": self._model_loaded,
            "state": "busy" if active_jobs else ("ready" if self._model_loaded else "starting"),
            "model_loaded": self._model_loaded,
            "audio_open": None,
            "last_error": self._last_error,
            "backend": self.cfg.tts.backend,
            "active_jobs": active_jobs,
        }

    async def handler(self, websocket) -> None:
        client = self._client_name(websocket)
        path = self._path(websocket)
        if path != self.cfg.tts.ws_path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return

        logger.info("WebSocket connected client=%s path=%s", client, path)
        queue: asyncio.Queue[TtsRequest | object] = asyncio.Queue(
            maxsize=int(self.cfg.tts.max_pending_text_requests)
        )
        producer = asyncio.create_task(self._receive_messages(websocket, queue))
        consumer = asyncio.create_task(self._send_audio(websocket, queue))
        try:
            done, _pending = await asyncio.wait({producer, consumer}, return_when=asyncio.FIRST_COMPLETED)
            if producer in done and not consumer.done():
                explicit_close = producer.result()
                if explicit_close:
                    await consumer
                else:
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
            elif consumer in done:
                await consumer
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        for task in (producer, consumer):
            if task.cancelled():
                continue
            exc = task.exception()
            if exc and not isinstance(exc, ConnectionClosed):
                logger.exception("WebSocket task failed client=%s", client, exc_info=exc)
        logger.info("WebSocket disconnected client=%s", client)

    async def _receive_messages(
        self,
        websocket,
        queue: asyncio.Queue[TtsRequest | object],
    ) -> bool:
        client = self._client_name(websocket)
        should_close = False
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._send_json(websocket, {"type": "error", "error": "binary input is not supported"})
                    continue
                try:
                    payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("message must be a JSON object")
                    msg_type = str(payload.get("type") or "text")
                    if msg_type == "ping":
                        status = self.status()
                        status["type"] = "pong"
                        await self._send_json(websocket, status)
                    elif msg_type == "status":
                        await self._send_json(websocket, self.status())
                    elif msg_type == "text":
                        req = _request_from_dict(payload)
                        if len(req.text) > self.cfg.tts.max_text_chars:
                            raise ValueError(f"text exceeds max_text_chars={self.cfg.tts.max_text_chars}")
                        try:
                            queue.put_nowait(req)
                        except asyncio.QueueFull:
                            await self._send_json(
                                websocket,
                                {"type": "error", "code": "QUEUE_FULL", "error": "text queue is full"},
                            )
                            continue
                        await self._send_json(websocket, {"type": "queued", "text_chars": len(req.text), "queue_size": queue.qsize()})
                    elif msg_type == "flush":
                        await queue.put(_FLUSH)
                    elif msg_type == "close":
                        await queue.put(_CLOSE)
                        should_close = True
                        return True
                    else:
                        raise ValueError(f"unsupported message type: {msg_type}")
                except Exception as exc:
                    logger.warning("Invalid WebSocket message client=%s error=%s", client, exc)
                    await self._send_json(websocket, {"type": "error", "error": str(exc)})
        except ConnectionClosed:
            return False
        finally:
            if not should_close:
                await queue.put(_CLOSE)
        return False

    async def _send_audio(
        self,
        websocket,
        queue: asyncio.Queue[TtsRequest | object],
    ) -> None:
        client = self._client_name(websocket)
        while True:
            item = await queue.get()
            if item is _CLOSE:
                try:
                    await self._send_json(websocket, {"type": "flushed"})
                except ConnectionClosed:
                    pass
                return
            if item is _FLUSH:
                try:
                    await self._send_json(websocket, {"type": "flushed"})
                except ConnectionClosed:
                    return
                continue
            req = item
            assert isinstance(req, TtsRequest)
            try:
                req = self._resolve_speaker(req)
            except Exception as exc:
                logger.warning("invalid speaker profile client=%s error=%s", client, exc)
                await self._send_json(websocket, {"type": "error", "code": "INVALID_SPEAKER", "error": str(exc)})
                continue
            started = time.perf_counter()
            acquired = await asyncio.to_thread(self._request_slots.acquire, True, float(self.cfg.tts.queue_timeout_sec))
            if not acquired:
                logger.warning(
                    "WebSocket request rejected queue_timeout client=%s text_chars=%d limit=%d",
                    client,
                    len(req.text),
                    self.cfg.tts.max_concurrent_requests,
                )
                await self._send_json(
                    websocket,
                    {
                        "type": "error",
                        "error": "service busy",
                        "retry_after_sec": self.cfg.tts.queue_timeout_sec,
                    },
                )
                continue
            chunks = 0
            total_bytes = 0
            try:
                started_audio = False
                async for chunk in self._stream_request_audio(req):
                    if not started_audio:
                        await self._send_json(
                            websocket,
                            {
                                "type": "start",
                                "text_chars": len(req.text),
                                "sample_rate": chunk.sample_rate,
                                "format": "pcm_s16le",
                                "channels": chunk.channels,
                            },
                        )
                        started_audio = True
                    await websocket.send(chunk.pcm)
                    chunks += 1
                    total_bytes += len(chunk.pcm)
                await self._send_json(
                    websocket,
                    {
                        "type": "end",
                        "chunks": chunks,
                        "bytes": total_bytes,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
                logger.info(
                    "WebSocket synthesis complete client=%s text_chars=%d chunks=%d bytes=%d duration_ms=%.1f",
                    client,
                    len(req.text),
                    chunks,
                    total_bytes,
                    (time.perf_counter() - started) * 1000,
                )
            except ConnectionClosed:
                return
            except Exception as exc:
                logger.exception("WebSocket synthesis failed client=%s error=%s", client, exc)
                await self._send_json(websocket, {"type": "error", "error": "synthesis failed"})

    async def serve_forever(self) -> None:
        logger.info(
            "Loading TTS engine backend=%s sample_rate=%s num_threads=%s",
            self.cfg.tts.backend,
            self.cfg.tts.sample_rate,
            self.cfg.tts.num_threads,
        )
        try:
            self.engine.load()
        except Exception as exc:
            self._last_error = str(exc)
            raise
        self._model_loaded = True
        self._last_error = None
        logger.info("TTS engine loaded backend=%s sample_rate=%s", self.cfg.tts.backend, self.engine.sample_rate)

        host = self.cfg.tts.ws_host or self.cfg.tts.host
        async with websockets.serve(self.handler, host, self.cfg.tts.ws_port):
            logger.info("GVoice WebSocket listening url=ws://%s:%s%s", host, self.cfg.tts.ws_port, self.cfg.tts.ws_path)
            await asyncio.Future()

    async def _stream_request_audio(self, req: TtsRequest):
        audio_queue: sync_queue.Queue[Any | Exception | None] = sync_queue.Queue(
            maxsize=int(self.cfg.tts.pcm_queue_chunks)
        )
        cancel = threading.Event()
        done = threading.Event()

        def put_output(item: Any | Exception | None) -> bool:
            while not cancel.is_set():
                try:
                    audio_queue.put(item, timeout=0.1)
                    return True
                except sync_queue.Full:
                    continue
            return False

        def worker() -> None:
            with self._jobs_lock:
                self._active_jobs += 1
            try:
                for chunk in self.engine.stream_pcm(req, cancel):
                    if not put_output(chunk):
                        break
            except Exception as exc:
                put_output(exc)
            finally:
                try:
                    audio_queue.put_nowait(None)
                except sync_queue.Full:
                    pass
                done.set()
                with self._jobs_lock:
                    self._active_jobs -= 1
                self._request_slots.release()

        thread = threading.Thread(target=worker, name="gvoice-ws-synth", daemon=True)
        try:
            thread.start()
        except Exception:
            self._request_slots.release()
            raise
        try:
            while True:
                item = await asyncio.to_thread(audio_queue.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # 客户端断开/打断时（async generator 被关闭）通知合成线程尽快停止，
            # 避免被放弃的长合成占住并发槽位
            cancel.set()
            finished = await asyncio.to_thread(done.wait, float(self.cfg.tts.cancel_wait_sec))
            if not finished:
                logger.warning(
                    "synthesis worker still stopping after %.1fs; concurrency slot remains held",
                    self.cfg.tts.cancel_wait_sec,
                )

    def _resolve_speaker(self, req: TtsRequest) -> TtsRequest:
        if not req.speaker:
            return req
        speaker_id, speed = apply_profile(self.cfg, req.speaker, speaker_id=req.speaker_id, speed=req.speed)
        return TtsRequest(
            text=req.text,
            speaker_id=speaker_id,
            speed=speed,
            speaker=req.speaker,
            noise_scale=req.noise_scale,
            noise_scale_w=req.noise_scale_w,
            length_scale=req.length_scale,
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
    asyncio.run(TtsWebSocketService(cfg).serve_forever())
