import io
import tarfile
import wave

import numpy as np
import pytest

from gvoice.config import load_config
from gvoice.engine import GenshinVitsOnnxEngine, TtsRequest, safe_extract_tar, split_text, wav_bytes_from_pcm


def test_split_text_keeps_chinese_sentences():
    text = "\u4f60\u597d\u3002\u73b0\u5728\u5f00\u59cb\u6d41\u5f0f\u8f93\u51fa\uff01OK"
    assert split_text(text) == [
        "\u4f60\u597d\u3002",
        "\u73b0\u5728\u5f00\u59cb\u6d41\u5f0f\u8f93\u51fa\uff01",
        "OK",
    ]


def test_wav_bytes_from_pcm_is_valid():
    data = wav_bytes_from_pcm((np.zeros(160, dtype="<i2")).tobytes(), sample_rate=16000)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"bad"
        info = tarfile.TarInfo("../bad.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, "r") as tar:
        with pytest.raises(RuntimeError):
            safe_extract_tar(tar, tmp_path / "out")


def test_config_defaults():
    cfg = load_config()
    assert cfg.tts.ws_port == 8787
    assert cfg.tts.ws_path == "/v1/tts/ws"
    assert cfg.tts.backend == "genshin_vits_onnx"
    assert cfg.tts.sample_rate == 22050
    assert cfg.tts.speaker_id == 115
    assert cfg.tts.speed == 0.85
    assert cfg.tts.noise_scale == 0.6
    assert cfg.tts.noise_scale_w == 0.668
    assert cfg.tts.length_scale == 1.2


def test_genshin_speed_maps_to_length_scale():
    cfg = load_config()
    engine = GenshinVitsOnnxEngine(cfg)
    assert engine._resolve_length_scale(TtsRequest("hi", speed=2.0)) == 0.6
    assert engine._resolve_length_scale(TtsRequest("hi", length_scale=1.05, speed=2.0)) == 1.05
