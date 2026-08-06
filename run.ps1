# 增量投放決策引擎 —— Windows 進入點
#   .\run.ps1 setup     建虛擬環境並裝套件
#   .\run.ps1 selftest  驗證評估程式（第 3 週要先過這關）
#   .\run.ps1 data      下載資料集
#   .\run.ps1 describe  印出資料實際 schema
#   .\run.ps1 balance   跑共變數平衡檢定
#   .\run.ps1 compare   跑完整模型比較
#   .\run.ps1 demo      開 Gradio 介面

param([Parameter(Position = 0)][string]$Command = "help")

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Require-Venv {
    if (-not (Test-Path $py)) {
        Write-Host "找不到虛擬環境。先跑： .\run.ps1 setup" -ForegroundColor Yellow
        exit 1
    }
}

switch ($Command) {
    "setup" {
        if (-not (Test-Path ".\.venv")) { python -m venv .venv }
        & $py -m pip install --upgrade pip
        & $py -m pip install -r requirements.txt
        if (-not (Test-Path ".env")) { Copy-Item .env.example .env }
        Write-Host "`n完成。下一步： .\run.ps1 selftest" -ForegroundColor Green
    }
    "selftest" {
        Require-Venv
        & $py -m src.evaluate
    }
    "data" {
        Require-Venv
        & $py scripts\download_data.py --dataset synthetic
        & $py scripts\download_data.py --dataset hillstrom
    }
    "describe" {
        Require-Venv
        & $py -m src.data --describe --dataset hillstrom
    }
    "balance" {
        Require-Venv
        & $py -m src.balance --dataset hillstrom
    }
    "compare" {
        Require-Venv
        & $py -m src.uplift --dataset hillstrom
    }
    "demo" {
        Require-Venv
        & $py app.py
    }
    default {
        Write-Host @"
增量投放決策引擎

  .\run.ps1 setup     建虛擬環境並裝套件
  .\run.ps1 selftest  驗證評估程式（不需要資料，先跑這個）
  .\run.ps1 data      下載 Kaggle 資料集
  .\run.ps1 describe  印出資料實際 schema
  .\run.ps1 balance   共變數平衡檢定
  .\run.ps1 compare   完整模型比較
  .\run.ps1 demo      開 Gradio 介面

建議順序: setup -> selftest -> data -> describe -> balance -> compare -> demo
"@
    }
}
