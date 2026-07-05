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

    async def handler(self, websocket) -> None:
        client = self._client_name(websocket)
        path = self._path(websocket)
        if path != self.cfg.tts.ws_path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return

        logger.info("WebSocket connected client=%s path=%s", client, path)
        queue: asyncio.Queue[TtsRequest | object] = asyncio.Queue()
        producer = asyncio.create_task(self._receive_messages(websocket, queue))
        consumer = asyncio.create_task(self._send_audio(websocket, queue))
        try:
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
    ) -> None:
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
                        await self._send_json(websocket, {"type": "pong", "backend": self.cfg.tts.backend})
                    elif msg_type == "text":
                        req = _request_from_dict(payload)
                        await queue.put(req)
                        await self._send_json(websocket, {"type": "queued", "text_chars": len(req.text), "queue_size": queue.qsize()})
                    elif msg_type == "flush":
                        await queue.put(_FLUSH)
                    elif msg_type == "close":
                        await queue.put(_CLOSE)
                        should_close = True
                        return
                    else:
                        raise ValueError(f"unsupported message type: {msg_type}")
                except Exception as exc:
                    logger.warning("Invalid WebSocket message client=%s error=%s", client, exc)
                    await self._send_json(websocket, {"type": "error", "error": str(exc)})
        except ConnectionClosed:
            return
        finally:
            if not should_close:
                await queue.put(_CLOSE)

    async def _send_audio(
        self,
        websocket,
        queue: asyncio.Queue[TtsRequest | object],
    ) -> None:
        client = self._client_name(websocket)
        while True:
            item = await queue.get()
            if item is _CLOSE:
                await self._send_json(websocket, {"type": "flushed"})
                return
            if item is _FLUSH:
                await self._send_json(websocket, {"type": "flushed"})
                continue
            req = item
            assert isinstance(req, TtsRequest)
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
                req = self._resolve_speaker(req)
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
            finally:
                self._request_slots.release()

    async def serve_forever(self) -> None:
        logger.info(
            "Loading TTS engine backend=%s sample_rate=%s num_threads=%s",
            self.cfg.tts.backend,
            self.cfg.tts.sample_rate,
            self.cfg.tts.num_threads,
        )
        self.engine.load()
        logger.info("TTS engine loaded backend=%s sample_rate=%s", self.cfg.tts.backend, self.engine.sample_rate)

        host = self.cfg.tts.ws_host or self.cfg.tts.host
        async with websockets.serve(self.handler, host, self.cfg.tts.ws_port):
            logger.info("GVoice WebSocket listening url=ws://%s:%s%s", host, self.cfg.tts.ws_port, self.cfg.tts.ws_path)
            await asyncio.Future()

    async def _stream_request_audio(self, req: TtsRequest):
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[Any | Exception | None] = asyncio.Queue()

        def worker() -> None:
            try:
                for chunk in self.engine.stream_pcm(req):
                    loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(audio_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        thread = threading.Thread(target=worker, name="gvoice-ws-synth", daemon=True)
        thread.start()
        while True:
            item = await audio_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

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
