# GVoice

GVoice 是一个面向本地部署的中文文本转语音服务，提供命令行合成和 WebSocket 流式全双工 TTS。项目当前重点是 CPU 友好、易调试、易扩展，适合在普通开发机上做中文 TTS 后端、说话人配置和后续语音克隆适配实验。

> 代码内置默认后端是 `genshin_vits_onnx`，需要本地准备对应 ONNX 模型和文本处理源码。仓库提供的 `configs/config.example.yaml` 使用 `sherpa_onnx_vits`，方便新贡献者先跑通公开 AISHELL3 VITS 模型。

## 功能

- 中文 TTS：支持 `cosyvoice3`（零样本音色克隆，GPU 流式）、`genshin_vits_onnx` 和 `sherpa_onnx_vits` 三个后端。
- WebSocket 全双工：客户端可以连续发送文本片段，服务端在同一连接上持续返回 PCM 二进制音频帧。
- WAV 合成：通过 CLI 直接生成 WAV。
- 说话人配置：支持保存固定 speaker id 配置，并预留参考音频克隆配置。
- 本地优先：主要使用 CPU ONNX Runtime，适合没有 NVIDIA CUDA 的开发环境。

## 项目结构

```text
GVoice/
  configs/              # 本地配置和示例配置
  docs/                 # 设计、调研和调试文档
  examples/             # 客户端和调用示例
  sidecars/cosyvoice3/  # CosyVoice3 流式合成 sidecar 子项目（独立 uv 环境）
  src/gvoice/           # Python 包源码
  tests/                # pytest 测试
  tools/                # 模型导出等开发工具
```

`artifacts/` 用于存放下载模型、导出文件、参考音频和生成结果，默认不会提交到版本库。

## 快速开始

推荐使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。

```powershell
cd D:\Project\Python_Project\GVoice
uv sync --extra dev
```

复制示例配置：

```powershell
Copy-Item configs\config.example.yaml configs\config.yaml
```

如果你已经准备好了 Genshin VITS ONNX 资产，也可以参考 `configs/config.genshin.example.yaml`。

启动 WebSocket 服务：

```powershell
uv run gvoice --config configs\config.yaml serve
```

排障时可以提高日志级别，或指定日志文件：

```powershell
uv run gvoice --config configs\config.yaml --log-level DEBUG --log-file artifacts\logs\gvoice.log serve
```

常驻服务默认一次只执行一个合成请求，其余请求最多等待 5 秒；超时会返回 WebSocket JSON 错误 `service busy`。可以在配置中调整：

```yaml
tts:
  max_concurrent_requests: 1
  queue_timeout_sec: 5.0
```

生成 WAV：

```powershell
uv run gvoice --config configs\config.yaml tts "你好，我是 GVoice。" --out artifacts\output\speech.wav
```

WebSocket 客户端示例：

```powershell
uv run python examples\ws_client.py "你好。" "第二句会继续流式合成。" --out artifacts\output\ws.wav
```

## WebSocket API

默认地址：

```text
ws://127.0.0.1:8787/v1/tts/ws
```

客户端发送 JSON 文本帧。最简单的文本消息：

```json
{"type":"text","text":"你好，我是 GVoice。"}
```

服务端返回两类帧：

- JSON 文本帧：状态、错误、开始和结束事件。
- Binary 帧：`pcm_s16le` 单声道音频 chunk。

典型流程：

```text
client -> {"type":"text","text":"第一句。"}
client -> {"type":"text","text":"第二句。"}
client -> {"type":"flush"}

server -> {"type":"queued",...}
server -> {"type":"queued",...}
server -> {"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,...}
server -> <binary pcm chunk>
server -> <binary pcm chunk>
server -> {"type":"end","chunks":2,"bytes":...}
server -> {"type":"start",...}
server -> <binary pcm chunk>
server -> {"type":"end",...}
server -> {"type":"flushed"}
```

支持的客户端消息：

| type | 说明 |
| --- | --- |
| `ping` | 返回 `pong`，用于连接检查。 |
| `text` | 入队一段文本并开始合成。 |
| `flush` | 等待已入队文本发送完毕后返回 `flushed`。 |
| `close` | 请求服务端关闭连接。 |

`text` 消息可附带调参字段：

```json
{
  "type": "text",
  "text": "你好。",
  "speaker": "demo",
  "speaker_id": 115,
  "speed": 0.85,
  "noise_scale": 0.6,
  "noise_scale_w": 0.668,
  "length_scale": 1.2
}
```

## 后端选择

### sherpa_onnx_vits

适合新贡献者快速启动。首次运行会下载 AISHELL3 VITS 模型到 `artifacts/tts/`。

```yaml
tts:
  backend: "sherpa_onnx_vits"
  sample_rate: 16000
```

### cosyvoice3

零样本音色克隆 + 低延迟流式合成，经独立 sidecar 子进程推理（需要 NVIDIA GPU）。
默认模型 `Fun-CosyVoice3-0.5B-2512`，可切换 1.5B。安装与说明见
[sidecars/cosyvoice3/README.md](sidecars/cosyvoice3/README.md) 和 [docs/cosyvoice3.md](docs/cosyvoice3.md)。

```yaml
tts:
  backend: "cosyvoice3"
  cosyvoice3:
    model: "0.5b"        # 或 1.5b
    prompt_wav: "D:/voices/reference.wav"   # 参考音频，决定音色
```

### genshin_vits_onnx

适合已经准备本地角色模型和导出 ONNX 的用户。需要以下路径可用：

- `artifacts/models/vits-models-genshin-bh3/keqing/keqing_tunable.onnx`
- `artifacts/sources/vits-models-genshin-bh3/config/config.json`
- `artifacts/sources/vits-models-genshin-bh3/text/`

导出说明见 [docs/genshin-vits.md](docs/genshin-vits.md)。

## 说话人配置

创建固定 speaker id 配置：

```powershell
uv run gvoice speakers init keqing --speaker-id 115 --speed 0.85
uv run gvoice tts "你好，我是刻晴。" --speaker keqing --out artifacts\output\keqing.wav
```

创建参考音频克隆配置：

```powershell
uv run gvoice speakers clone alice `
  --backend cosyvoice `
  --audio D:\voices\alice_01.wav `
  --reference-text "参考音频的原文"
```

克隆配置目前只保存资料，不会直接接入默认合成后端。路线说明见 [docs/speaker-cloning.md](docs/speaker-cloning.md)。

## 开发

安装开发依赖：

```powershell
uv sync --extra dev
```

运行测试：

```powershell
uv run pytest
```

打包检查：

```powershell
uv build
```

参与贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。如果你要报告 bug 或提交功能建议，请优先使用 `.github/ISSUE_TEMPLATE/` 中的模板。

## 文档

- [docs/research.md](docs/research.md)：中文 TTS 后端调研记录。
- [docs/genshin-vits.md](docs/genshin-vits.md)：Genshin VITS 模型和 ONNX 导出笔记。
- [docs/speaker-cloning.md](docs/speaker-cloning.md)：说话人配置和克隆后端路线。
- [docs/cli-debugging.md](docs/cli-debugging.md)：本地调试命令记录。

## 许可证和素材

项目源码使用 MIT 许可证，见 [LICENSE](LICENSE)。

请注意，下载或导出的角色语音模型、参考音频和其他第三方素材可能有独立授权限制。除非你确认拥有相应权利，否则请只将这些资产用于个人研究和实验，不要提交到仓库。
