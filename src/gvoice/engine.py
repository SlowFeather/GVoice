from __future__ import annotations

from dataclasses import dataclass
import importlib
import io
import json
import os
import re
import sys
import tarfile
import types
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config
from .download import download


_SENTENCE_RE = re.compile(r"[^\u3002\uff01\uff1f!?\uff1b;\n]+[\u3002\uff01\uff1f!?\uff1b;]?", re.U)


@dataclass(frozen=True)
class TtsRequest:
    text: str
    speaker_id: int | None = None
    speed: float | None = None
    speaker: str | None = None
    noise_scale: float | None = None
    noise_scale_w: float | None = None
    length_scale: float | None = None


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2


def safe_extract_tar(tar: tarfile.TarFile, target_dir: Path) -> None:
    target = Path(target_dir).resolve()
    for member in tar.getmembers():
        member_path = (target / member.name).resolve()
        if member_path != target and target not in member_path.parents:
            raise RuntimeError(f"refusing to extract unsafe tar member {member.name!r}")
    tar.extractall(target)


def split_text(text: str) -> list[str]:
    normalized = " ".join(text.replace("\r", "\n").split())
    if not normalized:
        return []
    pieces = [m.group(0).strip() for m in _SENTENCE_RE.finditer(normalized)]
    return [p for p in pieces if p]


def float_to_pcm16(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return b""
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def wav_bytes_from_pcm(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


class _BaseEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def sample_rate(self) -> int:
        return int(self.cfg.tts.sample_rate)

    def load(self) -> None:
        raise NotImplementedError

    def synthesize_sentence(self, text: str, req: TtsRequest) -> AudioChunk:
        raise NotImplementedError

    def stream_pcm(self, req: TtsRequest) -> Iterable[AudioChunk]:
        for sentence in split_text(req.text):
            audio = self.synthesize_sentence(sentence, req)
            chunk_samples = max(1, int(audio.sample_rate * self.cfg.tts.stream_chunk_ms / 1000))
            chunk_bytes = chunk_samples * audio.sample_width * audio.channels
            for offset in range(0, len(audio.pcm), chunk_bytes):
                part = audio.pcm[offset:offset + chunk_bytes]
                if part:
                    yield AudioChunk(part, sample_rate=audio.sample_rate)

    def synthesize_wav(self, req: TtsRequest) -> bytes:
        sample_rate = self.sample_rate
        pcm = bytearray()
        for chunk in self.stream_pcm(req):
            sample_rate = chunk.sample_rate
            pcm.extend(chunk.pcm)
        return wav_bytes_from_pcm(bytes(pcm), sample_rate=sample_rate)


class SherpaOnnxVitsEngine(_BaseEngine):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._tts = None

    @property
    def sample_rate(self) -> int:
        return 16000

    def load(self) -> None:
        if self._tts is not None:
            return
        _prefer_venv_onnxruntime_dll()
        import onnxruntime  # noqa: F401
        import sherpa_onnx

        fs = self.cfg.fs
        fs.ensure_dirs()
        onnx = fs.tts_dir / "vits-aishell3.onnx"
        if not onnx.exists():
            archive = download(
                self.cfg.tts.model_url,
                fs.tts_archive,
                desc="TTS model",
                sha256=self.cfg.tts.model_sha256,
            )
            with tarfile.open(archive, "r:bz2") as tar:
                safe_extract_tar(tar, fs.tts_dir.parent)

        lexicon = fs.tts_dir / "lexicon.txt"
        tokens = fs.tts_dir / "tokens.txt"
        missing = [p for p in (onnx, lexicon, tokens) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"TTS model files are missing: {missing}")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(onnx),
                    lexicon=str(lexicon),
                    tokens=str(tokens),
                ),
                provider="cpu",
                num_threads=int(self.cfg.tts.num_threads),
            )
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)

    def synthesize_sentence(self, text: str, req: TtsRequest) -> AudioChunk:
        self.load()
        assert self._tts is not None
        speaker_id = self.cfg.tts.speaker_id if req.speaker_id is None else req.speaker_id
        speed = self.cfg.tts.speed if req.speed is None else req.speed
        audio = self._tts.generate(text, sid=int(speaker_id), speed=float(speed))
        return AudioChunk(float_to_pcm16(audio.samples), sample_rate=int(audio.sample_rate))


class GenshinVitsOnnxEngine(_BaseEngine):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._session = None
        self._hps = None
        self._text_to_sequence = None

    def load(self) -> None:
        if self._session is not None:
            return
        _prefer_venv_onnxruntime_dll()
        import onnxruntime as ort

        model_path = Path(self.cfg.tts.genshin_model_path)
        source_dir = Path(self.cfg.tts.genshin_source_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Genshin VITS ONNX model not found: {model_path}")
        if not (source_dir / "config" / "config.json").exists():
            raise FileNotFoundError(f"Genshin VITS source/config not found: {source_dir}")

        self._load_text_adapter(source_dir)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(self.cfg.tts.num_threads)
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _load_text_adapter(self, source_dir: Path) -> None:
        _install_optional_text_dummies()
        resolved = str(source_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        text_module = importlib.import_module("text")
        self._text_to_sequence = text_module.text_to_sequence
        self._hps = _load_hparams(source_dir / "config" / "config.json")

    def synthesize_sentence(self, text: str, req: TtsRequest) -> AudioChunk:
        self.load()
        assert self._session is not None
        assert self._hps is not None
        assert self._text_to_sequence is not None

        seq, _clean_text = self._text_to_sequence(
            f"[ZH]{text}[ZH]",
            self._hps.symbols,
            self._hps.data.text_cleaners,
        )
        if self._hps.data.add_blank:
            seq = _intersperse(seq, 0)
        if not seq:
            return AudioChunk(b"", sample_rate=self.sample_rate)

        speaker_id = self.cfg.tts.speaker_id if req.speaker_id is None else req.speaker_id
        noise_scale = self.cfg.tts.noise_scale if req.noise_scale is None else req.noise_scale
        noise_scale_w = self.cfg.tts.noise_scale_w if req.noise_scale_w is None else req.noise_scale_w
        length_scale = self._resolve_length_scale(req)
        x = np.asarray(seq, dtype=np.int64)[None, :]
        feeds = {
            "x": x,
            "x_lengths": np.asarray([x.shape[1]], dtype=np.int64),
            "sid": np.asarray([int(speaker_id)], dtype=np.int64),
            "noise_scale": np.asarray([float(noise_scale)], dtype=np.float32),
            "noise_scale_w": np.asarray([float(noise_scale_w)], dtype=np.float32),
            "length_scale": np.asarray([float(length_scale)], dtype=np.float32),
        }
        audio = self._session.run(None, feeds)[0][0, 0]
        return AudioChunk(float_to_pcm16(audio), sample_rate=self.sample_rate)

    def _resolve_length_scale(self, req: TtsRequest) -> float:
        if req.length_scale is not None:
            return float(req.length_scale)
        if req.speed is not None:
            return max(0.05, float(self.cfg.tts.length_scale) / float(req.speed))
        return float(self.cfg.tts.length_scale)


class TtsEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._impl = _make_engine(cfg)

    @property
    def sample_rate(self) -> int:
        return self._impl.sample_rate

    def load(self) -> None:
        self._impl.load()

    def synthesize_sentence(self, text: str, *, speaker_id: int | None = None, speed: float | None = None) -> AudioChunk:
        return self._impl.synthesize_sentence(text, TtsRequest(text, speaker_id=speaker_id, speed=speed))

    def stream_pcm(self, req: TtsRequest) -> Iterable[AudioChunk]:
        return self._impl.stream_pcm(req)

    def synthesize_wav(self, req: TtsRequest) -> bytes:
        return self._impl.synthesize_wav(req)


def _make_engine(cfg: Config) -> _BaseEngine:
    if cfg.tts.backend == "sherpa_onnx_vits":
        return SherpaOnnxVitsEngine(cfg)
    if cfg.tts.backend == "genshin_vits_onnx":
        return GenshinVitsOnnxEngine(cfg)
    raise ValueError(f"unsupported TTS backend: {cfg.tts.backend}")


def _install_optional_text_dummies() -> None:
    if "pyopenjtalk" not in sys.modules:
        pyopenjtalk = types.ModuleType("pyopenjtalk")
        pyopenjtalk.g2p = lambda text, kana=False: text
        sys.modules["pyopenjtalk"] = pyopenjtalk
    if "jamo" not in sys.modules:
        jamo = types.ModuleType("jamo")
        jamo.h2j = lambda text: text
        jamo.j2hcj = lambda text: text
        sys.modules["jamo"] = jamo


def _load_hparams(config_path: Path) -> types.SimpleNamespace:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _to_namespace(data)


def _to_namespace(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _intersperse(items: list[int], item: int) -> list[int]:
    result = [item] * (len(items) * 2 + 1)
    result[1::2] = items
    return result


def _prefer_venv_onnxruntime_dll() -> None:
    import onnxruntime

    capi = Path(onnxruntime.__file__).resolve().parent / "capi"
    if not capi.exists():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(capi))
    os.environ["PATH"] = str(capi) + os.pathsep + os.environ.get("PATH", "")
