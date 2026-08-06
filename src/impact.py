"""模組 5：商業影響量化。

用法：
    python -m src.impact

這個模組回答的問題是「照模型建議做，一年多賺多少錢」。

【為什麼在模型之前先做這個】
先把錢的算式定下來，才知道模型該優化什麼。反過來做的話，很容易花三週
把 AUC 從 0.82 調到 0.85，卻發現對營收根本沒有差別。

【本模組最重要的產出不是那個營收數字，而是損益兩平點】
「這個方案要成立，挽回率至少要 X%」比「預估年增 Y 萬」有說服力得多，
因為前者不依賴任何樂觀假設。
"""

import numpy as np
import pandas as pd

from src.config import (
    ASSUMED_DISCOUNT_RATE,
    ASSUMED_GROSS_MARGIN,
    EVENT_PURCHASE,
    INTERIM_DIR,
    MONTHS,
    TABLE_DIR,
)

# 模型鎖定的高意圖 session 比例（模組 2 完成後會用真實的 lift curve 取代）
TARGET_SHARE = 0.20

# 【關鍵參數】被鎖定的前 20% 裡，涵蓋了多少比例的實際購買者。
#
# 第一版我漏掉這個參數，等於假設「被鎖定的人購買率 = 全站平均 3.45%」。
# 那是錯的：模型專門挑高意圖的人，這群人本來就會買的比例遠高於平均，
# 所以折扣浪費被嚴重低估。
#
# 一個堪用的模型，前 20% 大概能涵蓋 50-70% 的購買者。這裡先取 60%，
# 模組 2 完成後換成實際的 lift curve。
CAPTURE_RATE = 0.60

# 折扣挽回率的假設區間。這是全篇最不確定的參數，所以要做敏感度分析。
RECOVERY_RATES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def session_economics() -> dict:
    """從實際資料算出營收基準，不用任何外部假設。"""
    orders = []
    session_rows = []

    for month in MONTHS:
        df = pd.read_parquet(
            INTERIM_DIR / f"{month}.parquet",
            columns=["event_type", "price", "user_session"],
        )

        # 訂單 = 一個 session 內所有 purchase 事件的總和
        buys = df[df["event_type"] == EVENT_PURCHASE]
        order = buys.groupby("user_session", observed=True)["price"].agg(["sum", "count"])
        order["month"] = month
        orders.append(order)

        session_rows.append({
            "month": month,
            "sessions": df["user_session"].nunique(),
            "buying_sessions": buys["user_session"].nunique(),
        })
        del df, buys

    orders = pd.concat(orders)
    sessions = pd.DataFrame(session_rows)

    total_sessions = int(sessions["sessions"].sum())
    total_buying = int(sessions["buying_sessions"].sum())

    return {
        "月數": len(MONTHS),
        "總 session 數": total_sessions,
        "有購買的 session 數": total_buying,
        "session 購買轉換率": total_buying / total_sessions,
        "訂單數": len(orders),
        "總營收": float(orders["sum"].sum()),
        "平均訂單金額 AOV": float(orders["sum"].mean()),
        "訂單金額中位數": float(orders["sum"].median()),
        "每筆訂單平均件數": float(orders["count"].mean()),
        "平均商品單價": float(orders["sum"].sum() / orders["count"].sum()),
        "月均營收": float(orders["sum"].sum() / len(MONTHS)),
    }


def breakeven_recovery_rate(
    organic: float,
    non_buyers: float,
    discount: float = ASSUMED_DISCOUNT_RATE,
    margin: float = ASSUMED_GROSS_MARGIN,
) -> float:
    """損益兩平的挽回率。

    推導：
      被鎖定的 session 分成兩群 ——
        organic    本來就會買的人。對他們發折扣是純成本，不產生任何增額。
        non_buyers 本來不會買的人。其中比例 r 被折扣挽回，這才是增額。

      淨效益 = non_buyers x r x AOV x (margin - discount)
             - organic x AOV x discount

      令淨效益 = 0，解得：
        r* = organic x discount / (non_buyers x (margin - discount))

    這個數字不依賴任何樂觀假設，是整份分析最硬的結論。
    """
    if margin <= discount or non_buyers <= 0:
        return np.inf
    return (organic * discount) / (non_buyers * (margin - discount))


def scenario(
    econ: dict,
    recovery_rate: float,
    target_share: float = TARGET_SHARE,
    capture_rate: float = CAPTURE_RATE,
    discount: float = ASSUMED_DISCOUNT_RATE,
    margin: float = ASSUMED_GROSS_MARGIN,
) -> dict:
    """單一情境試算（以 5 個月資料換算成年度）。"""
    aov = econ["平均訂單金額 AOV"]
    months = econ["月數"]

    targeted = econ["總 session 數"] * target_share

    # 本來就會買卻拿到折扣的人 —— 這是被吃掉的毛利，不是效益。
    # 用「實際購買者 x 模型涵蓋率」計算，不是用全站平均轉換率。
    organic = econ["訂單數"] * capture_rate
    discount_waste = organic * aov * discount

    # 真正被挽回的增額訂單
    non_buyers = max(targeted - organic, 0.0)
    incremental = non_buyers * recovery_rate
    incremental_revenue = incremental * aov
    incremental_margin = incremental_revenue * (margin - discount)

    net = incremental_margin - discount_waste
    annual = net / months * 12

    # 【合理性檢核】增額訂單相對於現有訂單量的比例。
    # 沒有這道檢核，敏感度分析很容易產出幻想數字：挽回率 10% 看起來只是
    # 一個小數字，但它隱含「總訂單數增加 56%」—— 沒有任何促銷做得到。
    # 分析師的責任是自己先擋掉不可信的情境，不要等別人來抓。
    lift_vs_baseline = incremental / econ["訂單數"]

    return {
        "挽回率假設": recovery_rate,
        "鎖定 session 數": targeted,
        "其中本來就會買": organic,
        "本來不會買": non_buyers,
        "增額訂單數": incremental,
        "增額訂單佔現有%": lift_vs_baseline * 100,
        "增額營收": incremental_revenue,
        "增額毛利": incremental_margin,
        "折扣浪費成本": discount_waste,
        "淨效益(5個月)": net,
        "淨效益(年化)": annual,
        "合理性": "合理" if lift_vs_baseline <= 0.15 else "偏樂觀" if lift_vs_baseline <= 0.30 else "不可信",
    }


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def main() -> None:
    print("[*] 從實際資料計算營收基準...")
    econ = session_economics()

    print(f"\n{'=' * 56}")
    print("  營收基準（2019/10 - 2020/02，實際資料）")
    print(f"{'=' * 56}")
    for k, v in econ.items():
        if "率" in k:
            print(f"  {k:<22}{v:>18.2%}")
        elif isinstance(v, float):
            print(f"  {k:<22}{v:>18,.2f}")
        else:
            print(f"  {k:<22}{v:>18,}")

    pd.DataFrame([econ]).to_csv(
        TABLE_DIR / "baseline_economics.csv", index=False, encoding="utf-8-sig"
    )

    # ---- 損益兩平點 ----
    ref = scenario(econ, recovery_rate=0.0)
    organic, non_buyers = ref["其中本來就會買"], ref["本來不會買"]
    be = breakeven_recovery_rate(organic, non_buyers)
    be_orders = non_buyers * be
    be_lift = be_orders / econ["訂單數"]

    print(f"\n{'=' * 56}")
    print("  損益兩平分析")
    print(f"{'=' * 56}")
    print(f"  鎖定比例（前 X% session）{TARGET_SHARE:>18.0%}")
    print(f"  模型涵蓋率（含多少購買者）{CAPTURE_RATE:>17.0%}")
    print(f"  折扣幅度                 {ASSUMED_DISCOUNT_RATE:>18.2%}")
    print(f"  毛利率                   {ASSUMED_GROSS_MARGIN:>18.2%}")
    print("  ---")
    print(f"  鎖定 session 數          {ref['鎖定 session 數']:>18,.0f}")
    print(f"  其中本來就會買           {organic:>18,.0f}")
    print(f"  被鎖定者的購買率         {organic / ref['鎖定 session 數']:>18.2%}"
          f"   (全站平均 {econ['session 購買轉換率']:.2%})")
    print("  ---")
    print(f"  損益兩平所需挽回率       {be:>18.2%}")
    print(f"  對應的增額訂單           {be_orders:>18,.0f}  (佔現有訂單 {be_lift:.1%})")
    print("\n  解讀：折扣方案要不虧錢，被鎖定但原本不會買的人裡，")
    print(f"        至少要有 {be:.2%} 因為折扣而回來下單。")

    # ---- 敏感度分析 ----
    rows = [scenario(econ, r) for r in RECOVERY_RATES]
    sens = pd.DataFrame(rows)
    sens.to_csv(TABLE_DIR / "impact_sensitivity.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 56}")
    print(f"  敏感度分析（鎖定前 {TARGET_SHARE:.0%} 高意圖 session）")
    print(f"{'=' * 56}")
    print(f"  {'挽回率':>7}{'增額訂單':>11}{'佔現有訂單':>12}{'年化淨效益':>16}{'合理性':>10}")
    for r in rows:
        print(
            f"  {r['挽回率假設']:>6.0%}"
            f"{r['增額訂單數']:>11,.0f}"
            f"{r['增額訂單佔現有%']:>11.1f}%"
            f"{fmt_money(r['淨效益(年化)']):>16}"
            f"{r['合理性']:>10}"
        )

    credible = sens[sens["合理性"] == "合理"]
    if not credible.empty:
        lo = credible["淨效益(年化)"].min()
        hi = credible["淨效益(年化)"].max()
        print(f"\n  可信區間內的年化淨效益：{fmt_money(lo)} ~ {fmt_money(hi)}")

    # ---- 折扣幅度該訂多少 ----
    # 只說「這個方案不划算」是半套分析。要回答「那要怎樣才划算」，
    # 分析才有行動價值。折扣幅度是這裡唯一真正可控的槓桿。
    print(f"\n{'=' * 56}")
    print("  折扣幅度 × 損益兩平門檻")
    print(f"{'=' * 56}")
    print(f"  {'折扣':>6}{'兩平挽回率':>14}{'對應訂單成長':>16}{'可行性':>12}")

    disc_rows = []
    for d in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        r = breakeven_recovery_rate(organic, non_buyers, discount=d)
        lift = non_buyers * r / econ["訂單數"]
        verdict = "可行" if lift <= 0.15 else "困難" if lift <= 0.30 else "不可行"
        disc_rows.append({
            "折扣幅度": d, "兩平挽回率": r,
            "對應訂單成長%": lift * 100, "可行性": verdict,
        })
        print(f"  {d:>5.0%}{r:>13.2%}{lift:>15.1%}{verdict:>12}")

    pd.DataFrame(disc_rows).to_csv(
        TABLE_DIR / "breakeven_by_discount.csv", index=False, encoding="utf-8-sig"
    )

    print("\n  [!] 這些是情境試算不是實測結果。")
    print("      模組 2 完成後，鎖定比例會換成模型的實際 lift curve；")
    print("      挽回率則必須靠 A/B test 實測，無法從歷史資料推得。")
    print("      挽回率超過 3% 的情境隱含訂單量成長超過 16%，")
    print("      沒有促銷方案做得到，因此不納入結論。")
    print("\n[OK] 模組 5 骨架完成。表格已存至 outputs/tables/")


if __name__ == "__main__":
    main()
