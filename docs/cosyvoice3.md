# CosyVoice3 后端

`tts.backend: cosyvoice3` 通过独立的 sidecar 子进程（[sidecars/cosyvoice3](../sidecars/cosyvoice3/README.md)）
提供 CosyVoice3 零样本音色克隆 + 低延迟流式合成，实时接入 GVoice 的全双工 WebSocket 协议。

## 架构

```text
ChatCaht ──ws 8787──▶ GVoice (cosyvoice3 backend) ──ws 8788──▶ cosyvoice3-sidecar (CUDA)
                        │ PCM 帧直通转发，不整句缓冲                │ AutoModel 流式推理
                        │ cancel/断连 → sidecar 立即停止合成        │ 模型常驻显存
```

拆成 sidecar 的原因：

- CosyVoice 依赖 CUDA torch/transformers 重型栈，与 GVoice 主项目的 CPU ONNX 栈隔离，互不影响升级
- 模型常驻 sidecar 显存，GVoice/ChatCaht 重启不触发模型重新加载
- GPU 合成崩溃不影响 GVoice 服务本体

## 快速开始

```powershell
cd sidecars\cosyvoice3
.\setup.ps1                 # clone CosyVoice + uv sync + 下载 0.5b 模型
```

GVoice 配置（`configs/config.yaml`）：

```yaml
tts:
  backend: "cosyvoice3"
  cosyvoice3:
    model: "0.5b"           # 或 1.5b；改完重启 GVoice 自动切换
    prompt_wav: "C:/Users/Administrator/Downloads/zh_f.wav"
    prompt_text:            # 可选：参考音频逐字转写，填了相似度更高
```

之后正常 `uv run gvoice --config configs\config.yaml serve` 即可；连不上 sidecar 时 GVoice 会自动拉起它
（日志见 `artifacts/logs/cosyvoice3.sidecar.log`）。

## 延迟（RTX 4090 + fp16 实测，2026-07）

- 模型加载 ~18s（sidecar 常驻后不再发生）；启动即监听端口，模型后台加载 + 自动预热
- 稳态首包 **~1.2-1.4s**/句（`first_chunk_tokens: 15`，首块约 0.6-1s 音频），后续块 RTF 0.4-0.7；
  ChatCaht 按句流水（合成下一句与播放上一句重叠），所以只有整轮回复的第一句感知到首包延迟
- 音色特征启动时用 `add_zero_shot_spk` 缓存（无论有没有 `prompt_text`），每句免重复提特征
- 打断（barge-in）→ GVoice 发 cancel → sidecar 在当前块边界停止；被打断句子残余的
  GPU 工作无法瞬间中止（上游 llm 线程不可中断），紧跟其后的下一句首包可能多等 1-2s

## 上游 bug 的兜底（sidecar 内处理，无需改 vendor 代码）

- `token_hop_len` 跨请求翻倍不复位 → 首包持续劣化：每次合成前按 `tts.first_chunk_tokens` 复位
- `model.tts` 无 try/finally → 取消后 llm 线程僵尸运行 + 缓存字典泄漏：每次合成收尾清扫
  残留条目（僵尸线程随之以 KeyError 终止，日志中会出现一条无害的线程异常堆栈）
- CosyVoice3 要求文本或 prompt 中含 `<|endofprompt|>`：sidecar 自动按模式拼接
  `prompt.system` 前缀（zero_shot 拼在 prompt_text，cross_lingual 拼在正文，instruct2 拼在指令）

## 已知说明

- 1.5B 官方开源权重放出后，若模型 id 与 `FunAudioLLM/Fun-CosyVoice3-1.5B` 不一致，
  在 sidecar 配置 `model.model_id` 里填实际 id 即可
- `tts.speed`/`speaker_id` 等 VITS 调参字段对本后端无效；语速用 `tts.cosyvoice3.speed`
- 输出为 24 kHz `pcm_s16le`，采样率随 `start` 帧下发，ChatCaht 播放端自适应
