# 贡献指南

感谢你愿意改进 GVoice。这个项目欢迎小而清晰的改动：bug 修复、测试补充、文档修正、后端适配和可复现的性能优化都很有价值。

## 开发环境

1. 安装 Python 3.10+。
2. 安装 uv。
3. 安装依赖：

```powershell
uv sync --extra dev
```

4. 复制示例配置：

```powershell
Copy-Item configs\config.example.yaml configs\config.yaml
```

5. 运行测试：

```powershell
uv run pytest
```

## 提交前检查

请尽量在提交前完成：

```powershell
uv run pytest
```

如果改动涉及打包配置，也请运行：

```powershell
uv build
```

## 分支和提交

- 分支名建议使用 `fix/...`、`feat/...`、`docs/...` 或 `chore/...`。
- 提交信息建议使用简短祈使句，例如 `docs: add setup guide`。
- 一个 PR 尽量只解决一个问题，避免把重构、格式化和功能改动混在一起。

## 代码约定

- 保持模块边界清晰：配置在 `config.py`，HTTP 服务在 `server.py`，合成引擎在 `engine.py`，说话人资料在 `speakers.py`。
- 新增后端时优先实现一个独立 engine，再通过 `_make_engine` 注册。
- 新增用户可见行为时，请补充测试或说明为什么暂时无法自动化测试。
- 不要提交 `artifacts/`、参考音频、下载模型、导出模型、`.venv/` 或缓存文件。

## 文档约定

- README 面向首次使用者，保留安装、运行、API 和后端选择。
- `docs/` 面向开发者，记录设计决策、调研、导出流程和调试命令。
- 如果文档包含第三方模型、素材或论文链接，请写清楚来源和授权注意事项。

## 提交 Issue

报告 bug 时请包含：

- 操作系统和 Python 版本。
- 使用的后端和配置片段。
- 复现命令或最小代码。
- 实际输出、期望输出和关键报错。

功能建议请说明使用场景、期望行为和可以接受的折中方案。
