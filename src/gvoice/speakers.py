from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil

from .config import Config


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class SpeakerProfile:
    name: str
    backend: str = "genshin_vits_onnx"
    speaker_id: int | None = None
    speed: float | None = None
    reference_audio: list[str] = field(default_factory=list)
    reference_text: str | None = None
    language: str = "zh"
    notes: str = ""


def validate_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise ValueError("speaker name may contain only letters, numbers, dot, dash, and underscore")
    return name


def profile_dir(cfg: Config, name: str) -> Path:
    return cfg.fs.speakers_dir / validate_name(name)


def profile_path(cfg: Config, name: str) -> Path:
    return profile_dir(cfg, name) / "speaker.json"


def list_profiles(cfg: Config) -> list[str]:
    root = cfg.fs.speakers_dir
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "speaker.json").exists())


def load_profile(cfg: Config, name: str) -> SpeakerProfile:
    path = profile_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"speaker profile not found: {name}")
    return SpeakerProfile(**json.loads(path.read_text(encoding="utf-8")))


def save_profile(cfg: Config, profile: SpeakerProfile) -> Path:
    out_dir = profile_dir(cfg, profile.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "speaker.json"
    path.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def create_sherpa_profile(
    cfg: Config,
    name: str,
    *,
    speaker_id: int,
    speed: float | None = None,
    notes: str = "",
) -> SpeakerProfile:
    profile = SpeakerProfile(
        name=validate_name(name),
        backend=cfg.tts.backend,
        speaker_id=int(speaker_id),
        speed=speed,
        notes=notes,
    )
    save_profile(cfg, profile)
    return profile


def create_clone_profile(
    cfg: Config,
    name: str,
    *,
    backend: str,
    audio_paths: list[str],
    reference_text: str | None = None,
    language: str = "zh",
    notes: str = "",
) -> SpeakerProfile:
    if backend not in {"cosyvoice", "gpt_sovits", "openvoice"}:
        raise ValueError("clone backend must be one of: cosyvoice, gpt_sovits, openvoice")
    if not audio_paths:
        raise ValueError("at least one reference audio file is required")

    out_dir = profile_dir(cfg, name)
    refs_dir = out_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for idx, src_text in enumerate(audio_paths, start=1):
        src = Path(src_text)
        if not src.exists():
            raise FileNotFoundError(src)
        suffix = src.suffix.lower() or ".wav"
        dest = refs_dir / f"ref_{idx:02d}{suffix}"
        shutil.copy2(src, dest)
        copied.append(str(dest.relative_to(out_dir)).replace("\\", "/"))

    profile = SpeakerProfile(
        name=validate_name(name),
        backend=backend,
        reference_audio=copied,
        reference_text=reference_text,
        language=language,
        notes=notes,
    )
    save_profile(cfg, profile)
    return profile


def apply_profile(
    cfg: Config,
    name: str,
    *,
    speaker_id: int | None,
    speed: float | None,
) -> tuple[int | None, float | None]:
    profile = load_profile(cfg, name)
    if profile.backend not in {"sherpa_onnx_vits", "genshin_vits_onnx"}:
        raise ValueError(
            f"speaker {name!r} uses clone backend {profile.backend!r}; "
            "configure that backend before using it for synthesis"
        )
    return (
        speaker_id if speaker_id is not None else profile.speaker_id,
        speed if speed is not None else profile.speed,
    )
