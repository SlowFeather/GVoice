# CLI Debugging Guide

This guide records common GVoice commands used during local debugging.

## Install and Verify

```powershell
cd D:\Project\Python_Project\GVoice
uv sync --extra dev
uv run python -m compileall -q src tests tools
uv run pytest -q
```

Install export-only dependencies when you need to re-export `.pth` to ONNX:

```powershell
uv sync --extra dev --extra export
```

## Start the WebSocket Service

```powershell
uv run gvoice --config configs\config.yaml serve
```

If port `8787` is already occupied, start a temporary debug instance on another port:

```powershell
uv run gvoice --config configs\config.yaml serve --port 8795
```

Check the connection with the WebSocket sample client:

```powershell
uv run python examples\ws_client.py "你好，我是 GVoice。" --url ws://127.0.0.1:8795/v1/tts/ws --out artifacts\output\ws_debug.wav
```

Find the process listening on the default WebSocket port:

```powershell
Get-NetTCPConnection -LocalPort 8787 -State Listen |
  Select-Object LocalAddress,LocalPort,OwningProcess
```

Inspect the process command:

```powershell
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" |
  Select-Object ProcessId,CommandLine |
  Format-List
```

Stop a known debug process:

```powershell
Stop-Process -Id <PID>
```

## Generate WAV

Generate with default settings:

```powershell
uv run gvoice tts "你好，我是 GVoice。" --out artifacts\output\speech.wav
```

Try a more expressive tone:

```powershell
uv run gvoice tts "你好，我是刻晴。" ^
  --noise-scale 0.75 ^
  --noise-scale-w 0.75 ^
  --length-scale 1.15 ^
  --out artifacts\output\keqing_expressive.wav
```

## Stream Over WebSocket

```powershell
uv run python examples\ws_client.py "第一句。" "第二句。" --out artifacts\output\stream.wav
```

The client sends JSON text frames and writes binary PCM frames into a WAV file.

## Speaker Profiles

Create a numeric speaker profile for the current fixed VITS backend:

```powershell
uv run gvoice speakers init keqing --speaker-id 115 --speed 1.0
```

List profiles:

```powershell
uv run gvoice speakers list
```

Generate using a profile:

```powershell
uv run gvoice tts "你好，我是刻晴。" --speaker keqing --out artifacts\output\keqing_profile.wav
```

## Switch Backend

Use the sherpa-onnx AISHELL3 backend for comparison:

```powershell
uv run gvoice tts "你好，现在测试 sherpa 后端。" ^
  --backend sherpa_onnx_vits ^
  --speaker-id 0 ^
  --out artifacts\output\sherpa_compare.wav
```

Start the service with sherpa-onnx:

```powershell
uv run gvoice --config configs\config.yaml serve --backend sherpa_onnx_vits --port 8796
```

## Re-Export Keqing ONNX

```powershell
uv sync --extra dev --extra export

uv run python tools\export_genshin_vits_onnx.py ^
  --checkpoint artifacts\models\vits-models-genshin-bh3\keqing\keqing.pth ^
  --sid 115 ^
  --out artifacts\models\vits-models-genshin-bh3\keqing\keqing_tunable.onnx
```
