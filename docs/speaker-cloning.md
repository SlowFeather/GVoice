# Speaker Profiles and Voice Cloning

GVoice currently ships with a CPU-friendly default backend:

- `genshin_vits_onnx`
- Chinese-capable
- multi-speaker through numeric `speaker_id`
- streaming through the GVoice WebSocket service
- no arbitrary voice cloning by itself

This means there are two different workflows:

1. Use an existing VITS checkpoint or ONNX file with a built-in speaker id. This is fully supported now.
2. Use one or more reference recordings to clone a person. GVoice can store the profile now, and the profile is designed for a CosyVoice/GPT-SoVITS/OpenVoice backend adapter.

For the current Keqing model, no reference audio is needed. The voice is already inside `keqing_tunable.onnx`; generation only needs text plus the numeric `speaker_id`.

## Built-In Speaker Profiles

Create a reusable profile:

```powershell
uv run gvoice speakers init keqing --speaker-id 115 --speed 1.0
```

List profiles:

```powershell
uv run gvoice speakers list
```

Generate with that profile:

```powershell
uv run gvoice tts "\u4f60\u597d\uff0c\u6211\u662f\u523b\u6674\u3002" --speaker keqing --out artifacts\output\keqing_profile.wav
```

Use the same profile through WebSocket:

```json
{
  "type": "text",
  "text": "\u4f60\u597d\uff0c\u6211\u662f\u523b\u6674\u3002",
  "speaker": "keqing",
  "noise_scale": 0.6,
  "noise_scale_w": 0.668,
  "length_scale": 1.2
}
```

## Reference Audio Profiles

Create a clone profile from one recording:

```powershell
uv run gvoice speakers clone alice --backend cosyvoice --audio D:\voices\alice_01.wav --reference-text "\u53c2\u8003\u97f3\u9891\u7684\u539f\u6587"
```

Create one from multiple recordings:

```powershell
uv run gvoice speakers clone alice --backend cosyvoice ^
  --audio D:\voices\alice_01.wav ^
  --audio D:\voices\alice_02.wav ^
  --audio D:\voices\alice_03.wav ^
  --reference-text "\u8fd9\u91cc\u5199\u53c2\u8003\u97f3\u9891\u7684\u6587\u672c"
```

The files are copied into:

```text
artifacts/speakers/<name>/references/
artifacts/speakers/<name>/speaker.json
```

GVoice intentionally refuses to synthesize clone profiles with the fixed VITS backends. The profile records enough metadata for a future adapter, but the actual voice-clone model must be installed separately.

## Recommended Clone Backends

Use CosyVoice first if you want a modern Chinese zero-shot voice-clone backend with streaming-oriented design. Use GPT-SoVITS if you want the broader community tooling and are comfortable with a heavier stack. Fish Speech is less suitable on this machine because local inference generally favors CUDA/high VRAM.

## Recording Advice

- Use WAV when possible.
- 16 kHz or 24 kHz mono is easiest to debug.
- Record 10 to 30 seconds of clean speech for a first try.
- For multi-reference profiles, use 3 to 10 clips from the same microphone and room.
- Include the exact transcript in `--reference-text` when the backend requires prompt text.
- Only clone voices you have permission to use.
