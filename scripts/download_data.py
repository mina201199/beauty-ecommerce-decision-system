"""用 Kaggle API 下載資料集。

前置作業（只要做一次）：
  1. 到 https://www.kaggle.com/settings -> API -> Create New Token
  2. 會下載 kaggle.json
  3. 放到 C:\\Users\\User\\.kaggle\\kaggle.json

用法：
  python scripts/download_data.py --dataset synthetic
  python scripts/download_data.py --all
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, RAW_DIR  # noqa: E402


def download(key: str) -> None:
    ds = DATASETS[key]
    dest = RAW_DIR / key
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.rglob("*.csv")):
        print(f"[skip] {key} 已經有資料了 -> {dest}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (ImportError, OSError) as e:
        print(f"[error] 無法載入 kaggle 套件：{e}")
        print("        pip install kaggle，並確認 ~/.kaggle/kaggle.json 存在")
        return

    api = KaggleApi()
    api.authenticate()
    print(f"[..] 下載 {ds.kaggle_slug} -> {dest}")
    api.dataset_download_files(ds.kaggle_slug, path=str(dest), quiet=False)

    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()

    files = sorted(p.name for p in dest.rglob("*") if p.is_file())
    print(f"[ok] {key}: {files}")
    if ds.note:
        print(f"     注意：{ds.note}")
    print(f"     下一步：python -m src.data --describe --dataset {key}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS))
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    if a.all:
        for k in DATASETS:
            download(k)
    elif a.dataset:
        download(a.dataset)
    else:
        print("可用的資料集：")
        for k, d in DATASETS.items():
            print(f"  {k:20s} {d.kaggle_slug}")
        print("\n建議從 --dataset synthetic 開始（估計器驗證用）")
