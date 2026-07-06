# CosyVoice3 Sidecar

GVoice 的 CosyVoice3 流式合成 sidecar：独立进程、独立 uv 环境（CUDA torch + 官方 CosyVoice 代码），
通过 WebSocket 向 GVoice 提供低延迟流式 PCM。GVoice 的 `tts.backend: cosyvoice3` 会自动拉起本服务。

为什么独立成 sidecar：CosyVoice 依赖 CUDA 版 torch/transformers 等重型栈，与 GVoice 主项目的
CPU ONNX 栈完全隔离；模型常驻显存，GVoice 重启不需要重新加载模型。

## 安装

```powershell
cd sidecars\cosyvoice3
.\setup.ps1              # clone CosyVoice + uv sync + 下载 0.5b 模型
```

或手动：

```powershell
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor\CosyVoice
uv sync
uv run cosyvoice3-sidecar download
```

## 使用

```powershell
uv run cosyvoice3-sidecar serve
uv run cosyvoice3-sidecar serve --model 1.5b
uv run cosyvoice3-sidecar tts "你好，我是 CosyVoice3。" --out output.wav   # 调试
```

配置见 `configs/config.yaml`（首次从 `config.example.yaml` 复制）：

- `model.name`：`0.5b`（默认，`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`）或 `1.5b`；也可填完整模型 id
- `prompt.wav`：参考音频，决定输出音色
- `prompt.text`：参考音频逐字转写。填了走 zero_shot（相似度更高，音色特征启动时缓存）；留空走 cross_lingual
- `model.warmup`：启动预热，显著降低首句延迟
- `model.fp16`：GPU 半精度（无 CUDA 时自动回退）

> 注：1.5B 官方开源权重放出后若 id 与 `FunAudioLLM/Fun-CosyVoice3-1.5B` 不同，直接在 `model.model_id` 填实际 id 即可。

## WebSocket 协议

地址 `ws://127.0.0.1:8788/v1/cosyvoice3/ws`，客户端发 JSON 文本帧：

| type | 说明 |
| --- | --- |
| `ping` | 回 `pong`，含 `ready`（模型是否加载完）、`model`、`sample_rate` |
| `synthesize` | `{"type":"synthesize","id":1,"text":"...","speed":1.0}`，回 `start` + 若干 binary PCM 帧 + `end` |
| `cancel` | `{"type":"cancel","id":1}` 停止当前/排队合成，回 `cancelled`（用于打断） |
| `shutdown` | 优雅退出进程（GVoice 切换模型时用） |
| `close` | 关闭连接 |

音频为 `pcm_s16le` 单声道，采样率见 `start` 帧（CosyVoice3 为 24000）。
服务监听后立即可连接；模型在后台加载，未就绪时 `synthesize` 会等待加载完成。
