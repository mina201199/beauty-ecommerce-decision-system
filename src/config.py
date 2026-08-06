"""專案設定：資料集註冊表、路徑、隨機種子。

所有可調參數集中在這裡（沿用你 RAG 專案 config.py 的習慣）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- 路徑 ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"

for _d in (RAW_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- 再現性 -------------------------------------------------------------
SEED = 42

# --- 評估參數 -----------------------------------------------------------
TEST_SIZE = 0.30
N_BOOTSTRAP = 200          # 信賴區間用；先設小一點，跑順了再加
UPLIFT_AT_K = (0.10, 0.20, 0.30, 0.50)

# --- 商業參數（決策層用，之後在 app 裡做敏感度分析）--------------------
CONTACT_COST = 1.0         # 每次接觸成本
MARGIN_PER_CONVERSION = 30.0


@dataclass
class Dataset:
    """一個 uplift 資料集的宣告。

    treatment_col / outcome_col 先給常見值，載入時 data.py 會驗證，
    對不上就會列出實際欄位讓你改。不要憑印象硬寫。
    """

    key: str
    kaggle_slug: str
    filename: str
    treatment_col: str
    outcome_col: str
    # treatment 欄若是字串（例如 Hillstrom 的 segment），指定哪個值算「有處理」
    treated_values: tuple[str, ...] = ()
    control_values: tuple[str, ...] = ()
    drop_cols: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def path(self) -> Path:
        return RAW_DIR / self.key / self.filename


DATASETS: dict[str, Dataset] = {
    # 階段 0：先在這個上面驗證估計器。它附 ground_truth.json，是唯一
    # 能證明「你的 Qini 實作沒寫錯」的資料。0 個公開 notebook。
    "synthetic": Dataset(
        key="synthetic",
        kaggle_slug="krishnaharish1/marketing-ab-test-synthetic-uplift",
        filename="",  # 下載後用 --describe 看實際檔名再填
        treatment_col="treatment",
        outcome_col="conversion",
        note="附 ground_truth.json，估計器驗證用。先跑這個。",
    ),
    # 階段 1：主力。Hillstrom 郵件行銷 RCT。
    "hillstrom": Dataset(
        key="hillstrom",
        kaggle_slug="davinwijaya/customer-retention",
        filename="data.csv",
        treatment_col="segment",
        outcome_col="visit",
        treated_values=("Womens E-Mail", "Mens E-Mail"),
        control_values=("No E-Mail",),
        drop_cols=("spend", "conversion"),  # 這兩個是 post-treatment，不可當共變數
        note="真實 RCT。三組設計，先做二元（有寄/沒寄）再考慮多臂。",
    ),
    # 階段 2：規模驗證。1,300 萬列。
    "criteo": Dataset(
        key="criteo",
        kaggle_slug="arashnic/uplift-modeling",
        filename="criteo-uplift-v2.1.csv",
        treatment_col="treatment",
        outcome_col="conversion",
        drop_cols=("visit", "exposure"),
        note="~324MB。特徵是匿名的 f0..f11，沒有商業敘事，純粹用來證明規模與方法。",
    ),
    # 階段 3：真實多渠道 CRM。只在前面做完後才碰。
    "rees46_messaging": Dataset(
        key="rees46_messaging",
        kaggle_slug="mkechinov/direct-messaging",
        filename="messages-demo.csv",
        treatment_col="",  # 需要自己從 campaign metadata 建構，沒有現成 treatment 欄
        outcome_col="purchase",
        note="7 個 notebook，幾乎沒有參考實作。注意：campaign_id 只在 campaign_type 內唯一，"
        "單獨 join campaign_id 會 fan-out。holidays.csv 是俄羅斯節日。",
    ),
}

DEFAULT_DATASET = "hillstrom"
