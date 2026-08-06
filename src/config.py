"""專案設定：路徑、資料型別、模型參數都集中在這裡。

為什麼要有這個檔案：
所有「會被改動的數字和路徑」放同一個地方，之後調參數不用翻遍每個檔案。
這是工程化的基本功，面試看你的 repo 時會注意到。
"""

import sys
from pathlib import Path

# Windows 終端機預設用 cp950 編碼，印中文會變亂碼。
# 所有模組都會 import config，所以在這裡統一修掉。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 路徑
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # Kaggle 下載的原始 CSV（已 gitignore）
INTERIM_DIR = DATA_DIR / "interim"  # 轉檔後的 parquet（已 gitignore）

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (RAW_DIR, INTERIM_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 資料來源
KAGGLE_DATASET = "mkechinov/ecommerce-events-history-in-cosmetics-shop"

# 資料涵蓋 2019/10 - 2020/02，共 5 個月
MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan", "2020-Feb"]


# ---------------------------------------------------------------- 資料型別
# 明確指定 dtype 是這個專案最重要的效能技巧。
# 不指定的話 pandas 會把所有欄位當 object(字串)，2000 萬列會吃掉十幾 GB。
CSV_DTYPES = {
    "event_type": "category",     # 只有 4 種值，category 省超多記憶體
    "product_id": "int32",
    "category_id": "int64",       # 這欄是 19 位數的大整數，不能用 int32
    "category_code": "category",  # 有大量空值，category 處理得比 object 好
    "brand": "category",
    "price": "float32",           # 價格不需要 float64 的精度
    "user_id": "int64",
    "user_session": "string",     # UUID 字串，轉 parquet 後會被字典編碼壓縮
}

# 事件類型（漏斗順序）
EVENT_VIEW = "view"
EVENT_CART = "cart"
EVENT_REMOVE = "remove_from_cart"
EVENT_PURCHASE = "purchase"
EVENT_ORDER = [EVENT_VIEW, EVENT_CART, EVENT_REMOVE, EVENT_PURCHASE]


# ---------------------------------------------------------------- 建模設定
RANDOM_SEED = 42

# 【防止 Data Leakage 的關鍵設定】
# 預測「這個 session 會不會購買」時，只能用 session 前 N 分鐘的行為當特徵。
# 如果用整個 session 的資料，就等於偷看了購買前後的所有動作 → AUC 會假性飆到 0.99。
OBSERVATION_WINDOW_MINUTES = 10

# 時間切分：用前面的月份訓練、最後一個月驗證。
# 絕對不要用 train_test_split 隨機切！行為資料有時間性，
# 隨機切等於「用 12 月的資料預測 10 月」，是穿越，面試會被問死。
TRAIN_MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec"]
VALID_MONTHS = ["2020-Jan"]
TEST_MONTHS = ["2020-Feb"]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",  # PR-AUC，不平衡資料要看這個不是 accuracy
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1000,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------- 商業換算假設
# 模組 5 要把模型指標翻譯成錢。這些假設一定要寫在 README 裡公開，
# 不能藏起來 —— 面試官問「這個數字怎麼來的」你要答得出來。
ASSUMED_DISCOUNT_RATE = 0.10        # 對高意圖客發的折扣幅度
ASSUMED_GROSS_MARGIN = 0.45         # 美妝電商的典型毛利率

# 注意：「折扣能挽回多少人」這個參數刻意不寫死在這裡。
# 它無法從歷史資料推得，只能靠 A/B test 實測。模組 5 改用敏感度分析
# 掃過一個區間，並計算損益兩平點 —— 那個數字不依賴任何樂觀假設。
