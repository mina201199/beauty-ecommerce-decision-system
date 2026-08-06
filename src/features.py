"""模組 2 特徵工程：把事件流轉成 session 層級的特徵表。

用法：
    python -m src.features

【三道防洩漏機制】
  1. 只用 session 開始後 W 分鐘內的事件
  2. W 分鐘內已購買的 session 直接排除（沒東西可預測）
  3. purchase 事件永遠不進特徵，只用來產生標籤

【一個資料特性造成的設計限制】
  模組 1 發現 74.14% 的加購沒有對應的瀏覽記錄 —— view 事件本身記錄不完整。
  因此本模組刻意不做「加購/瀏覽比」這類跨事件比率特徵：那種特徵會讓模型
  去學「這個 session 的追蹤有沒有壞掉」，而不是學購買意圖，測試集上會崩。
  remove/cart 比率則保留，因為這兩種事件都記錄完整。
"""

import numpy as np
import pandas as pd

from src.config import (
    EVENT_CART,
    EVENT_PURCHASE,
    EVENT_REMOVE,
    EVENT_VIEW,
    INTERIM_DIR,
    MONTHS,
    OBSERVATION_WINDOW_MINUTES,
)

FEATURE_DIR = INTERIM_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = OBSERVATION_WINDOW_MINUTES * 60


def build_month(month: str) -> pd.DataFrame:
    df = pd.read_parquet(
        INTERIM_DIR / f"{month}.parquet",
        columns=["event_time", "event_type", "product_id", "brand", "price", "user_session"],
    )

    # ---- session 邊界與購買時點 ----
    g = df.groupby("user_session", observed=True)["event_time"]
    meta = g.agg(start="min", end="max")

    buys = df[df["event_type"] == EVENT_PURCHASE]
    meta["first_purchase"] = buys.groupby("user_session", observed=True)["event_time"].min()
    del buys

    meta["duration_sec"] = (meta["end"] - meta["start"]).dt.total_seconds()
    meta["ttp_sec"] = (meta["first_purchase"] - meta["start"]).dt.total_seconds()

    # ---- 資格篩選 ----
    alive = meta["duration_sec"] >= WINDOW_SEC
    not_yet_bought = meta["ttp_sec"].isna() | (meta["ttp_sec"] > WINDOW_SEC)
    meta = meta[alive & not_yet_bought].copy()

    # 標籤：觀察窗之後是否發生購買
    meta["label"] = (meta["ttp_sec"] > WINDOW_SEC).fillna(False).astype("int8")

    # ---- 只留下合格 session 在觀察窗內的非購買事件 ----
    df = df[df["user_session"].isin(meta.index)]
    df = df.merge(
        meta[["start"]], left_on="user_session", right_index=True, how="inner"
    )
    df["elapsed"] = (df["event_time"] - df["start"]).dt.total_seconds()
    df = df[(df["elapsed"] <= WINDOW_SEC) & (df["event_type"] != EVENT_PURCHASE)]

    # ---- 特徵 ----
    df["is_view"] = df["event_type"] == EVENT_VIEW
    df["is_cart"] = df["event_type"] == EVENT_CART
    df["is_remove"] = df["event_type"] == EVENT_REMOVE

    grp = df.groupby("user_session", observed=True)

    feat = grp.agg(
        n_events=("event_time", "count"),
        n_view=("is_view", "sum"),
        n_cart=("is_cart", "sum"),
        n_remove=("is_remove", "sum"),
        n_products=("product_id", "nunique"),
        n_brands=("brand", "nunique"),
        price_mean=("price", "mean"),
        price_max=("price", "max"),
        price_min=("price", "min"),
        price_std=("price", "std"),
        last_event_sec=("elapsed", "max"),
    )

    # 購物車金額：加購商品的價格總和，是意圖強度最直接的訊號
    cart_rows = df[df["is_cart"]]
    cart_stat = cart_rows.groupby("user_session", observed=True)["price"].agg(
        cart_value="sum", cart_price_max="max"
    )
    feat = feat.join(cart_stat)
    feat[["cart_value", "cart_price_max"]] = feat[["cart_value", "cart_price_max"]].fillna(0.0)

    # 事件間隔：反映瀏覽節奏。猶豫的人間隔長，目標明確的人間隔短。
    df = df.sort_values(["user_session", "elapsed"])
    df["gap"] = df.groupby("user_session", observed=True)["elapsed"].diff()
    gap_stat = df.groupby("user_session", observed=True)["gap"].agg(
        gap_median="median", gap_max="max"
    )
    feat = feat.join(gap_stat)

    # ---- 衍生特徵 ----
    # 淨購物車件數：加購扣掉移除。負值代表這個人在清空購物車。
    feat["net_cart"] = feat["n_cart"] - feat["n_remove"]
    # 移除/加購比：兩種事件都記錄完整，這個比率可以安心用
    feat["remove_rate"] = np.where(
        feat["n_cart"] > 0, feat["n_remove"] / feat["n_cart"], 0.0
    )
    feat["events_per_min"] = feat["n_events"] / OBSERVATION_WINDOW_MINUTES
    feat["price_range"] = feat["price_max"] - feat["price_min"]
    # 每個商品平均看幾次：反覆看同一件商品是猶豫的訊號
    feat["events_per_product"] = feat["n_events"] / feat["n_products"].clip(lower=1)

    # ---- 時間特徵 ----
    feat = feat.join(meta[["start", "label"]])
    feat["hour"] = feat["start"].dt.hour
    feat["weekday"] = feat["start"].dt.weekday
    feat["is_weekend"] = (feat["weekday"] >= 5).astype("int8")
    feat = feat.drop(columns=["start"])

    feat["month"] = month
    return feat.reset_index()


def main() -> None:
    print(f"[*] 觀察窗 = {OBSERVATION_WINDOW_MINUTES} 分鐘")

    for month in MONTHS:
        feat = build_month(month)
        out = FEATURE_DIR / f"{month}.parquet"
        feat.to_parquet(out, engine="pyarrow", compression="snappy", index=False)

        pos = feat["label"].mean()
        print(f"    {month}  樣本 {len(feat):>9,}   正樣本 {pos:>6.2%}   欄位 {feat.shape[1]}")
        del feat

    print(f"\n[OK] 特徵表已存至 {FEATURE_DIR}")
    print("     下一步：python -m src.model_intent")


if __name__ == "__main__":
    main()
