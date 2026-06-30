import numpy as np
import soundfile as sf
import pytest

from gvoice.config import load_config
from gvoice.speakers import (
    apply_profile,
    create_clone_profile,
    create_sherpa_profile,
    list_profiles,
    load_profile,
)


def test_sherpa_profile_roundtrip(tmp_path):
    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")

    create_sherpa_profile(cfg, "demo", speaker_id=3, speed=1.2)

    assert list_profiles(cfg) == ["demo"]
    profile = load_profile(cfg, "demo")
    assert profile.speaker_id == 3
    assert apply_profile(cfg, "demo", speaker_id=None, speed=None) == (3, 1.2)
    assert apply_profile(cfg, "demo", speaker_id=4, speed=None) == (4, 1.2)


def test_clone_profile_copies_reference_audio_and_refuses_default_synthesis(tmp_path):
    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(1600, dtype=np.float32), 16000)

    create_clone_profile(
        cfg,
        "alice",
        backend="cosyvoice",
        audio_paths=[str(ref)],
        reference_text="\u4f60\u597d",
    )

    profile = load_profile(cfg, "alice")
    assert profile.backend == "cosyvoice"
    assert profile.reference_audio == ["references/ref_01.wav"]
    assert (tmp_path / "artifacts" / "speakers" / "alice" / "references" / "ref_01.wav").exists()
    with pytest.raises(ValueError, match="clone backend"):
        apply_profile(cfg, "alice", speaker_id=None, speed=None)
