"""從 Kaggle 下載原始資料。

用法：
    python -m src.download

需要先設定 Kaggle API 金鑰，兩種方式擇一：
  (a) 把 kaggle.json 放到  C:\\Users\\User\\.kaggle\\kaggle.json
  (b) 複製 .env.example 成 .env，填入 KAGGLE_USERNAME / KAGGLE_KEY
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import KAGGLE_DATASET, MONTHS, RAW_DIR


def load_env_file() -> None:
    """讀取專案根目錄的 .env（如果有的話），寫進環境變數。

    刻意不用 python-dotenv，少一個相依套件。格式簡單所以自己解析就好。
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def check_credentials() -> None:
    """確認金鑰存在，沒有的話給出可執行的指示（而不是一句 KeyError）。"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if kaggle_json.exists() or has_env:
        return

    print(
        "\n[X] 找不到 Kaggle API 金鑰。\n"
        "\n取得步驟：\n"
        "  1. 登入 kaggle.com → 右上角頭像 → Settings\n"
        "  2. 往下找到 API 區塊 → 點 'Create New Token'\n"
        "  3. 瀏覽器會下載 kaggle.json\n"
        f"  4. 把它移到：{kaggle_json}\n"
        "\n（也可以複製 .env.example 成 .env，把金鑰填進去）\n",
        file=sys.stderr,
    )
    sys.exit(1)


def already_downloaded() -> bool:
    """5 個月的 CSV 都在就不重複下載 —— 這包有 450MB，別浪費頻寬。"""
    return all((RAW_DIR / f"{m}.csv").exists() for m in MONTHS)


def unzip_manual_download() -> bool:
    """處理手動下載的 zip。

    不想設定 API 金鑰的話，可以直接在 Kaggle 網站按 Download，
    把下載到的 zip 丟進 data/raw/，這支程式會自動解開。
    """
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        return False

    for zip_path in zips:
        size_mb = zip_path.stat().st_size / 1024 ** 2
        print(f"[*] 發現手動下載的壓縮檔：{zip_path.name} ({size_mb:.0f} MB)")
        print("[*] 解壓縮中...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        print(f"[OK] 已解開 {zip_path.name}")

    return already_downloaded()


def download() -> None:
    print(f"[*] 下載資料集：{KAGGLE_DATASET}")
    print(f"[*] 存放位置：{RAW_DIR}")
    print("[*] 大小約 450MB，依網速可能需要 5-15 分鐘...\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(RAW_DIR),
            "--unzip",
        ],
        check=False,
    )

    if result.returncode != 0:
        print(
            "\n[X] 下載失敗。常見原因：\n"
            "  - 金鑰無效或過期 → 重新產生一次 kaggle.json\n"
            "  - 沒有在 Kaggle 網站上接受該資料集的使用條款\n"
            f"    → 用瀏覽器開 https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            "      按一次 Download，接受條款後再回來跑這支程式\n",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def report() -> None:
    print("\n[*] 下載結果：")
    total_mb = 0.0
    for month in MONTHS:
        path = RAW_DIR / f"{month}.csv"
        if path.exists():
            size_mb = path.stat().st_size / 1024 ** 2
            total_mb += size_mb
            print(f"    [OK] {month}.csv  {size_mb:>8.1f} MB")
        else:
            print(f"    [--] {month}.csv  缺少")
    print(f"\n    合計 {total_mb:.1f} MB")


def main() -> None:
    load_env_file()

    if already_downloaded():
        print("[*] 5 個月的資料都已存在，略過下載。")
        report()
        return

    # 先看有沒有手動下載的 zip，有的話就不需要 API 金鑰
    if unzip_manual_download():
        report()
        print("\n[OK] 下一步：python -m src.prepare")
        return

    check_credentials()
    download()
    report()
    print("\n[OK] 下一步：python -m src.prepare")


if __name__ == "__main__":
    main()
