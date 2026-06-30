# TTS Backend Research

Hardware checked on this machine:

- CPU: AMD Ryzen AI 9 HX 370, 12 cores / 24 threads
- GPU: AMD Radeon 890M, no NVIDIA CUDA
- RAM: about 24 GB

Decision update: the service now defaults to the exported Keqing `genshin_vits_onnx` model because the user selected its voice sample. It supports Chinese, runs on CPU through ONNX Runtime, and is exposed as a service-level streaming API by splitting text into short units and returning PCM chunks as soon as each unit is synthesized.

The original `sherpa-onnx` Chinese VITS backend remains as the stable CPU fallback.

## Current Model Position

The current Keqing model is useful, lightweight, and easy to serve on this machine after ONNX export, but it should not be considered a leading modern TTS model. It is a fixed-speaker VITS checkpoint with runtime controls for noise and length. That makes it good for:

- a stable anime-style Chinese voice
- CPU-friendly local inference
- simple ONNX deployment
- service-level streaming by sentence chunking

It is weaker than newer LLM/codec/diffusion-style systems for:

- zero-shot voice cloning from a short prompt
- natural long-form prosody
- emotion control
- cross-lingual voice preservation
- native low-latency streaming

## Better Upgrade Paths

For this project, the best next backend depends on the target:

| Goal | Recommended backend | Why |
| --- | --- | --- |
| Better Chinese cloning with local developer tooling | GPT-SoVITS | Strong community tooling, zero-shot and few-shot workflows, Chinese supported |
| Best streaming-oriented Chinese backend to try next | CosyVoice / CosyVoice 2 or 3 | Designed around modern zero-shot multilingual TTS and streaming/non-streaming use |
| Highest-quality open-source-style emotional generation, if hardware allows | Fish Speech / Fish Audio S2 | Very strong quality and emotion control, but local inference can require high VRAM |
| Expressive Chinese/English emotional TTS | IndexTTS / IndexTTS2 | Strong emotional and controllable TTS direction, but training/reproduction story may be less simple |
| Lightweight fixed voice service | Current Genshin VITS ONNX | Small, fast enough, already integrated |

Practical recommendation for this AMD CPU/iGPU machine:

1. Keep `genshin_vits_onnx` as the default lightweight fixed voice.
2. Add CosyVoice as the next experimental backend if streaming and Chinese quality matter most.
3. Add GPT-SoVITS if the priority is cloning a specific speaker from one or more reference clips.
4. Avoid Fish Speech as the first local backend unless a suitable GPU or hosted inference path is available.

## Candidates

| Backend | Chinese | Streaming | Local fit |
| --- | --- | --- | --- |
| CosyVoice / CosyVoice2 | Yes | Strong native streaming and low latency support | Best quality upgrade path, but heavier dependencies and models |
| GPT-SoVITS | Yes | Mainly HTTP/inference-service oriented; can be adapted | Good cloning quality, heavier stack for this no-CUDA PC |
| Bert-VITS2 | Yes | Mostly non-streaming VITS-style inference | Older and training-oriented, less attractive for a new service |
| Fish Speech | Yes | Streaming supported in the ecosystem | Current local inference favors CUDA/high VRAM; not ideal on AMD iGPU |
| ChatTTS | Yes | Streaming support is less stable/central | Nice conversational style, but less suitable as the first reliable local service |
| Genshin VITS ONNX | Yes | Service-level streaming, not native acoustic streaming | Current default for the selected Keqing voice |
| sherpa-onnx VITS | Yes | Service-level streaming, not native acoustic streaming | Stable CPU fallback |

## Sources

- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice2: https://funaudiollm.github.io/cosyvoice2/
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- Bert-VITS2: https://github.com/fishaudio/Bert-VITS2
- Fish Speech inference docs: https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md
- ChatTTS: https://github.com/2noise/ChatTTS
- Genshin VITS model source: https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3
- IndexTTS: https://github.com/index-tts/index-tts
