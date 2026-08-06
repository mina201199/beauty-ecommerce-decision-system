"""模組 2 前置作業：用實際分布決定觀察窗長度。

用法：
    python -m src.sessions

【要解決的問題】
規格書原本寫「用 session 前 10 分鐘的行為預測會不會購買」，但那個 10 分鐘
是拍腦袋來的。如果多數購買在 3 分鐘內就發生，10 分鐘的窗會把購買事件本身
包進特徵裡 —— 那就是 data leakage。

【預測問題的正確定義】
    「一個 session 已經瀏覽了 W 分鐘、目前還沒下單，他最終會不會買？」

這個定義有三個好處：
  1. 不會洩漏：W 分鐘前已經購買的 session 直接排除，沒有東西可預測
  2. 可上線：這正是即時介入的時機點 —— 人還在站上、還沒買
  3. 標籤明確：label = W 分鐘之後是否發生購買

所以 W 要選在「還留得住夠多正樣本」與「觀察期夠長到有訊號」之間。
這支程式就是用來找那個平衡點。
"""

import pandas as pd

from src.config import EVENT_PURCHASE, INTERIM_DIR, MONTHS, TABLE_DIR

# 候選觀察窗（分鐘）
CANDIDATE_WINDOWS = [1, 2, 3, 5, 10, 15, 30]


def session_profile() -> pd.DataFrame:
    """每個 session 一列：起訖時間、事件數、首次購買時間。"""
    frames = []

    for month in MONTHS:
        df = pd.read_parquet(
            INTERIM_DIR / f"{month}.parquet",
            columns=["event_time", "event_type", "user_session"],
        )

        base = df.groupby("user_session", observed=True)["event_time"].agg(
            start="min", end="max", n_events="count"
        )

        buys = df[df["event_type"] == EVENT_PURCHASE]
        first_buy = buys.groupby("user_session", observed=True)["event_time"].min()
        base["first_purchase"] = first_buy

        base["month"] = month
        frames.append(base.reset_index())
        del df, buys, first_buy

    return pd.concat(frames, ignore_index=True)


def describe(prof: pd.DataFrame) -> None:
    dur_min = (prof["end"] - prof["start"]).dt.total_seconds() / 60
    prof["duration_min"] = dur_min

    print(f"\n{'=' * 58}")
    print("  Session 時長分布（分鐘）")
    print(f"{'=' * 58}")
    for q in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  P{q * 100:<5.0f}{dur_min.quantile(q):>18.2f}")
    print(f"  平均 {dur_min.mean():>22.2f}")

    print(f"\n{'=' * 58}")
    print("  每 session 事件數")
    print(f"{'=' * 58}")
    for q in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  P{q * 100:<5.0f}{prof['n_events'].quantile(q):>18.1f}")
    print(f"  平均 {prof['n_events'].mean():>22.2f}")

    # 單事件 session：這些人一進站就走，沒有任何行為訊號可用
    single = (prof["n_events"] == 1).mean()
    print(f"\n  只有 1 個事件的 session 佔比 {single:>10.2%}")

    buyers = prof[prof["first_purchase"].notna()].copy()
    ttp = (buyers["first_purchase"] - buyers["start"]).dt.total_seconds() / 60
    prof.loc[buyers.index, "time_to_purchase_min"] = ttp

    print(f"\n{'=' * 58}")
    print("  首次購買發生在 session 開始後幾分鐘")
    print(f"{'=' * 58}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  P{q * 100:<5.0f}{ttp.quantile(q):>18.2f}")
    print(f"  平均 {ttp.mean():>22.2f}")
    print(f"\n  購買發生在 1 分鐘內的比例 {(ttp <= 1).mean():>12.2%}")
    print(f"  購買發生在 5 分鐘內的比例 {(ttp <= 5).mean():>12.2%}")


def window_tradeoff(prof: pd.DataFrame) -> pd.DataFrame:
    """掃描候選觀察窗，看每個 W 留下多少可預測樣本與正樣本比例。

    符合資格的 session 必須同時滿足：
      - 活過 W 分鐘（否則沒有完整的觀察期）
      - 在 W 分鐘內尚未購買（已經買了就沒東西可預測）
    標籤 = W 分鐘之後是否發生購買。
    """
    dur = prof["duration_min"]
    ttp = prof["time_to_purchase_min"]
    rows = []

    for w in CANDIDATE_WINDOWS:
        alive = dur >= w
        not_yet_bought = ttp.isna() | (ttp > w)
        eligible = alive & not_yet_bought

        n_elig = int(eligible.sum())
        n_pos = int((eligible & ttp.notna() & (ttp > w)).sum())

        rows.append({
            "觀察窗(分)": w,
            "符合資格 session": n_elig,
            "佔全體%": n_elig / len(prof) * 100,
            "正樣本數": n_pos,
            "正樣本比例%": n_pos / n_elig * 100 if n_elig else 0.0,
            "保留的購買者%": n_pos / prof["first_purchase"].notna().sum() * 100,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("[*] 建立 session 輪廓...")
    prof = session_profile()
    print(f"    共 {len(prof):,} 個 session")

    describe(prof)

    tradeoff = window_tradeoff(prof)
    tradeoff.to_csv(
        TABLE_DIR / "observation_window_tradeoff.csv",
        index=False, encoding="utf-8-sig",
    )

    print(f"\n{'=' * 58}")
    print("  觀察窗取捨")
    print(f"{'=' * 58}")
    print(tradeoff.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n  讀法：觀察窗越長，訊號越多，但符合資格的 session 越少、")
    print("        且會排除掉越多早期就購買的人（保留的購買者%下降）。")
    print("\n[OK] 完成。下一步依此結果設定 OBSERVATION_WINDOW_MINUTES。")


if __name__ == "__main__":
    main()
