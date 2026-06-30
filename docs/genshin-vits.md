# Genshin VITS `.pth` Models

The `zomehwh/vits-models-genshin-bh3` Space includes PyTorch VITS checkpoints and the matching source code/config. Some Genshin entries are Chinese-capable:

| Key | Language | Speaker id | Notes |
| --- | --- | ---: | --- |
| `keqing` | Chinese | 115 | Downloaded and exported to ONNX locally |
| `eula` | Chinese | 124 | Download was interrupted; retry before use |

These files are not sherpa-onnx model packages. They are raw VITS checkpoints that need the original repository's text preprocessing code. GVoice now includes a `genshin_vits_onnx` backend that uses that source tree to turn Chinese text into token ids, then runs ONNX Runtime on CPU.

## Export Result

Keqing has been exported successfully:

```text
artifacts/models/vits-models-genshin-bh3/keqing/keqing.pth
artifacts/models/vits-models-genshin-bh3/keqing/keqing.onnx
artifacts/models/vits-models-genshin-bh3/keqing/keqing_tunable.onnx
```

The ONNX model was checked with `onnx.checker` and loaded with `onnxruntime` on CPU. A dummy token input produced an audio tensor with shape `[1, 1, 16384]`.
The tunable ONNX model exposes `noise_scale`, `noise_scale_w`, and `length_scale` as runtime inputs.

The default GVoice config uses:

```yaml
backend: "genshin_vits_onnx"
sample_rate: 22050
speaker_id: 115
noise_scale: 0.6
noise_scale_w: 0.668
length_scale: 1.2
genshin_model_path: "artifacts/models/vits-models-genshin-bh3/keqing/keqing_tunable.onnx"
genshin_source_dir: "artifacts/sources/vits-models-genshin-bh3"
```

These values match the selected `artifacts/output/keqing_tunable_default.wav` sample.

## Re-Export

Install export-only dependencies first:

```powershell
uv sync --extra dev --extra export
```

Export Keqing:

```powershell
uv run python tools\export_genshin_vits_onnx.py ^
  --checkpoint artifacts\models\vits-models-genshin-bh3\keqing\keqing.pth ^
  --sid 115 ^
  --out artifacts\models\vits-models-genshin-bh3\keqing\keqing_tunable.onnx
```

The export script emits an ONNX graph with these inputs:

- `x`: token ids, shape `[1, text_length]`
- `x_lengths`: token length, shape `[1]`
- `sid`: speaker id, shape `[1]`
- `noise_scale`: emotional/random variation control, shape `[1]`
- `noise_scale_w`: duration/prosody variation control, shape `[1]`
- `length_scale`: speech speed scale, shape `[1]`

Good starting values:

- Natural/default: `noise_scale=0.6`, `noise_scale_w=0.668`, `length_scale=1.2`
- More expressive: `noise_scale=0.75`, `noise_scale_w=0.75`, `length_scale=1.15`
- More stable/flat: `noise_scale=0.3`, `noise_scale_w=0.5`, `length_scale=1.2`

The GVoice backend imports the original `text_to_sequence` preprocessing from the downloaded source tree. On Windows, `pyopenjtalk` can be hard to build; for the Chinese-only path used here, GVoice installs tiny compatibility dummies for `pyopenjtalk` and `jamo` before importing the source text package.

## Service Usage

Generate with the default Keqing settings:

```powershell
uv run gvoice tts "\u4f60\u597d\uff0c\u6211\u662f\u523b\u6674\u3002\u4eca\u5929\u4e5f\u8981\u52aa\u529b\u5de5\u4f5c\u3002" --out artifacts\output\gvoice_keqing_default.wav
```

Try a more expressive version:

```powershell
uv run gvoice tts "\u4f60\u597d\uff0c\u6211\u662f\u523b\u6674\u3002" ^
  --noise-scale 0.75 ^
  --noise-scale-w 0.75 ^
  --length-scale 1.15 ^
  --out artifacts\output\gvoice_keqing_expressive.wav
```

Use WebSocket streaming:

```powershell
uv run python examples\ws_client.py "\u4f60\u597d\uff0c\u6211\u662f\u523b\u6674\u3002" --out artifacts\output\keqing_ws.wav
```

## Download and Replace a Character

Original Space:

```text
https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3
```

Keqing source checkpoint:

```text
https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3/resolve/main/pretrained_models/keqing/keqing.pth
```

Keqing cover:

```text
https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3/resolve/main/pretrained_models/keqing/cover.png
```

Character list and speaker ids:

```text
https://huggingface.co/spaces/zomehwh/vits-models-genshin-bh3/resolve/main/pretrained_models/info.json
```

To replace Keqing with another character:

1. Open the Space file browser and find `pretrained_models/<character>/`.
2. Download `<character>.pth` and optional `cover.png`.
3. Look up that character's `sid` in `pretrained_models/info.json`.
4. Export the checkpoint to tunable ONNX:

```powershell
uv sync --extra dev --extra export

uv run python tools\export_genshin_vits_onnx.py ^
  --checkpoint artifacts\models\vits-models-genshin-bh3\<character>\<character>.pth ^
  --sid <sid> ^
  --out artifacts\models\vits-models-genshin-bh3\<character>\<character>_tunable.onnx
```

5. Update `configs/config.yaml`:

```yaml
tts:
  backend: "genshin_vits_onnx"
  speaker_id: <sid>
  genshin_model_path: "artifacts/models/vits-models-genshin-bh3/<character>/<character>_tunable.onnx"
  genshin_source_dir: "artifacts/sources/vits-models-genshin-bh3"
```

The source tree under `artifacts/sources/vits-models-genshin-bh3` can be reused for other characters from the same Space.

## Safety

The `.pth` files are PyTorch pickle checkpoints. Only load them in an isolated virtual environment and only if you trust the source.
