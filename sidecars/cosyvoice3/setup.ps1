# CosyVoice3 sidecar 一键安装：vendor CosyVoice 仓库 + 同步依赖 + 下载模型
# 用法：  .\setup.ps1            （默认 0.5b）
#        .\setup.ps1 -Model 1.5b
param(
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "vendor\CosyVoice\cosyvoice")) {
    Write-Host "==> Cloning CosyVoice (with submodules)..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git vendor\CosyVoice
} elseif (-not (Test-Path "vendor\CosyVoice\third_party\Matcha-TTS\matcha")) {
    Write-Host "==> Fetching Matcha-TTS submodule..."
    git -C vendor\CosyVoice submodule update --init --recursive
}

if (-not (Test-Path "configs\config.yaml")) {
    Copy-Item configs\config.example.yaml configs\config.yaml
}

Write-Host "==> uv sync (torch cu121, first run may take a while)..."
uv sync

Write-Host "==> Downloading model..."
if ($Model) {
    uv run cosyvoice3-sidecar download --model $Model
} else {
    uv run cosyvoice3-sidecar download
}

Write-Host "==> Done. Start with: uv run cosyvoice3-sidecar serve"
