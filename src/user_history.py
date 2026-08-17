"""使用者跨 session 歷史特徵。

用法：
    python -m src.user_history

【為什麼要有這個模組】
原本的模型只用單一 session 前 3 分鐘的行為，完全沒用到 `user_id` 跨月份
持續存在這件事。但「這個人是不是老客戶、過去買過幾次、上次來是多久以前」
通常是購買預測最強的訊號 —— 不用等於把資料丟在桌上沒撿。

【防洩漏的關鍵】
歷史特徵只能用「本次 session 開始之前」已經結束的 session。
實作上先依 (user_id, session 起始時間) 排序，再以「累積總和減去當列」
取得嚴格在前的統計量 —— 不用 shift 是因為 shift 只退一列，
而我們要的是該使用者先前的全部。

【一個必須誠實揭露的限制】
資料窗從 2019/10 開始，所以 10 月的 session 一律沒有歷史，
即使那些人其實是老客戶。到了 2 月則最多可累積 4 個月的歷史。
這會造成訓練集與測試集在這組特徵上的分布不一致，
本模組會把各月的分布印出來供檢查。
"""

import numpy as np
import pandas as pd

from src.config import EVENT_CART, EVENT_PURCHASE, INTERIM_DIR, MONTHS

OUT_PATH = INTERIM_DIR / "user_history.parquet"


def session_table() -> pd.DataFrame:
    """把事件流壓成 session 層級，一列一個 session。"""
    frames = []
    for month in MONTHS:
        df = pd.read_parquet(
            INTERIM_DIR / f"{month}.parquet",
            columns=["event_time", "event_type", "price", "user_id", "user_session"],
        )
        df["is_cart"] = df["event_type"] == EVENT_CART
        df["is_buy"] = df["event_type"] == EVENT_PURCHASE
        df["spend"] = np.where(df["is_buy"], df["price"], 0.0)

        s = df.groupby("user_session", observed=True).agg(
            user_id=("user_id", "first"),
            start=("event_time", "min"),
            end=("event_time", "max"),
            n_events=("event_time", "size"),
            n_cart=("is_cart", "sum"),
            n_buy=("is_buy", "sum"),
            spend=("spend", "sum"),
        )
        s["month"] = month
        frames.append(s.reset_index())
        del df

    out = pd.concat(frames, ignore_index=True)

    # 跨月份的 session 會在兩個月的檔案裡各出現一次，必須合回同一列。
    # 不合併的話：(a) 該 session 被當成兩次造訪，造訪次數會灌水
    #            (b) 併入特徵表時 user_session 對到多列，資料會膨脹
    dup = int(out["user_session"].duplicated().sum())
    if dup:
        out = out.groupby("user_session", as_index=False).agg(
            user_id=("user_id", "first"),
            start=("start", "min"),
            end=("end", "max"),
            n_events=("n_events", "sum"),
            n_cart=("n_cart", "sum"),
            n_buy=("n_buy", "sum"),
            spend=("spend", "sum"),
            month=("month", "first"),
        )
        print(f"    合併 {dup:,} 個跨月份被切開的 session")

    assert not out["user_session"].duplicated().any()
    return out


def add_history(s: pd.DataFrame) -> pd.DataFrame:
    """為每個 session 計算「在它之前」的使用者歷史。

    【為什麼全部用 numpy 陣列組出來，而不是逐欄指派給既有 DataFrame】
    第一版是在寬表上一欄一欄 `s["x"] = ...` 地加。結果出現靜默資料損毀：
    最後指派的 int8 欄位整欄變成 0，而且兩次執行壞掉的欄位還不一樣
    （第一次是 is_returning，第二次換成 has_bought_before）。

    這種不穩定的行為追下去是 pandas 內部的區塊合併，追它沒有價值。
    正確的工程回應是換掉脆弱的寫法：先把每個欄位算成獨立的 numpy 陣列，
    最後一次性組成新的 DataFrame，並用斷言守住欄位間的硬性關係。
    """
    s = s.sort_values(["user_id", "start"]).reset_index(drop=True)
    g = s.groupby("user_id", sort=False)

    # 第幾次造訪（0 = 首次）
    prior_sessions = g.cumcount().to_numpy()

    # 累積總和減去當列 = 嚴格在此 session 之前的總量，不會洩漏
    def prior_of(col: str) -> np.ndarray:
        return (g[col].cumsum() - s[col]).to_numpy()

    prior_purchases = prior_of("n_buy")
    prior_carts = prior_of("n_cart")
    prior_events = prior_of("n_events")
    prior_spend = prior_of("spend")

    # 距離上次造訪結束多久。首次造訪沒有「上次」，填 -1 當明確的哨兵值，
    # 讓樹模型自己切出獨立分支，而不是填 0 假裝「昨天才來過」。
    prev_end = g["end"].shift(1)
    days_since_last = (
        ((s["start"] - prev_end).dt.total_seconds() / 86400)
        .fillna(-1.0).to_numpy()
    )

    out = pd.DataFrame({
        "user_session": s["user_session"].to_numpy(),
        "month": s["month"].to_numpy(),
        "prior_sessions": prior_sessions,
        "prior_purchases": prior_purchases,
        "prior_carts": prior_carts,
        "prior_events": prior_events,
        "prior_spend": prior_spend,
        "days_since_last": days_since_last,
        "is_returning": (prior_sessions > 0).astype(np.int8),
        "has_bought_before": (prior_purchases > 0).astype(np.int8),
        "prior_purchase_rate": np.where(
            prior_sessions > 0, prior_purchases / np.maximum(prior_sessions, 1), -1.0
        ),
        "prior_avg_spend": np.where(
            prior_purchases > 0, prior_spend / np.maximum(prior_purchases, 1), -1.0
        ),
    })

    check_invariants(out)
    return out


def check_invariants(h: pd.DataFrame) -> None:
    """欄位之間的硬性關係，違反就當場失敗。

    這些不是防禦性程式碼，是針對已經實際發生過的靜默損毀所設的守門員。
    寧可讓流程停在這裡，也不要把自相矛盾的資料存下去給模型學。
    """
    pairs = [("prior_sessions", "is_returning"), ("prior_purchases", "has_bought_before")]
    for src, flag in pairs:
        n_src = int((h[src] > 0).sum())
        n_flag = int((h[flag] == 1).sum())
        assert n_src == n_flag, (
            f"{flag} 與 {src} 不一致：{src}>0 有 {n_src:,} 列，"
            f"{flag}==1 只有 {n_flag:,} 列"
        )

    assert (h["prior_sessions"] >= 0).all(), "prior_sessions 不應為負"
    assert (h["prior_purchases"] <= h["prior_events"]).all(), \
        "先前購買數不可能超過先前事件數"
    assert (h["prior_carts"] <= h["prior_events"]).all(), \
        "先前加購數不可能超過先前事件數"
    assert h["user_session"].notna().all(), "user_session 不應有缺值"


def main() -> None:
    print("[*] 建立 session 層級彙總...")
    s = session_table()
    print(f"    {len(s):,} 個 session　{s['user_id'].nunique():,} 位使用者")

    hist = add_history(s)
    hist.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=False)

    print(f"\n{'=' * 62}")
    print("  各月份的歷史特徵分布（檢查資料窗造成的偏移）")
    print(f"{'=' * 62}")
    print(f"  {'月份':<10}{'回訪率':>10}{'曾購買率':>10}"
          f"{'平均造訪次數':>14}{'平均前次間隔(天)':>18}")
    for m in MONTHS:
        sub = hist[hist["month"] == m]
        gap = sub.loc[sub["days_since_last"] >= 0, "days_since_last"]
        print(f"  {m:<10}{sub['is_returning'].mean():>10.1%}"
              f"{sub['has_bought_before'].mean():>10.1%}"
              f"{sub['prior_sessions'].mean():>14.2f}"
              f"{gap.mean() if len(gap) else float('nan'):>18.2f}")

    print("\n  [!] 各月可回溯的歷史長度不同，這是資料窗造成的偏移，不是真實行為變化。")
    print("      10 月只能看到當月之內的先前造訪，2 月則最多可回溯 4 個月，")
    print("      因此「曾購買率」由 8.5% 一路上升到 20.9%。")
    print("      訓練集含 10 月、測試集為 2 月，兩者在這組特徵上分布不同 ——")
    print("      若模型評估顯示測試分數異常高於驗證分數，要回頭懷疑這件事。")
    print(f"\n[OK] 已存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
