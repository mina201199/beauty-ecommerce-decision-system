"""機會量化：探索優化能把數字改善到什麼程度。

用法：
    python -m src.opportunity

【這個模組補的是什麼洞】
模組 3 證明了折扣挽回不划算，模組 4 做出了不花錢的替代方案，
但一直沒有回答最關鍵的問題：**這個替代方案能改善多少？**

沒有這個數字，整份分析就停在「別做折扣」，沒有正面主張。

【怎麼算】
從實際資料量出「觀察窗內加購件數 → 之後的購買率」曲線，
再推估「若探索優化讓部分零加購 session 加到第一件」的效果。

【這是相關性不是因果，必須講清楚】
加購 1 件的 session 購買率是零加購的 2.76 倍，但這兩群人本來就不同 ——
會加購的人本來就更想買。硬把人推去加購，不會自動獲得同樣的購買率。
因此本模組的估算是**效果上限**，真實值必須靠 A/B test 測得。
"""

import pandas as pd

from src.config import (
    ASSUMED_GROSS_MARGIN,
    MONTHS,
    TABLE_DIR,
)
from src.features import FEATURE_DIR

# 探索優化的假設成本。這不是折扣那種變動成本，而是一次性的開發投入。
# 取 30,000 美元約當 3 個工程師月（含管銷）。與營收同幣別。
ASSUMED_DEV_COST = 30_000

# 情境：探索優化把多少比例的「零加購」session 推進到「至少加購一件」
CONVERSION_SCENARIOS = [0.01, 0.02, 0.05, 0.10, 0.20]


def cart_curve() -> pd.DataFrame:
    """觀察窗內加購件數與之後購買率的實測關係。"""
    df = pd.concat(
        [pd.read_parquet(FEATURE_DIR / f"{m}.parquet", columns=["n_cart", "label"])
         for m in MONTHS],
        ignore_index=True,
    )
    base = df["label"].mean()
    grouped = df.groupby(df["n_cart"].clip(upper=6))

    rows = []
    for k, sub in grouped:
        rows.append({
            "加購件數": f"{k}+" if k == 6 else str(k),
            "session 數": len(sub),
            "佔比%": len(sub) / len(df) * 100,
            "購買率%": sub["label"].mean() * 100,
            "vs 整體": sub["label"].mean() / base,
        })
    curve = pd.DataFrame(rows)

    meta = {
        "n_sessions": len(df),
        "base_rate": float(base),
        "zero_cart_sessions": int((df["n_cart"] == 0).sum()),
        "zero_cart_rate": float(df.loc[df["n_cart"] == 0, "label"].mean()),
        "one_cart_rate": float(df.loc[df["n_cart"] == 1, "label"].mean()),
    }
    return curve, meta


def scenarios(meta: dict, econ: pd.Series) -> pd.DataFrame:
    """若探索優化把 X% 的零加購 session 推到加購一件。"""
    lift_pp = meta["one_cart_rate"] - meta["zero_cart_rate"]
    months = econ["月數"]
    aov = econ["平均訂單金額 AOV"]

    rows = []
    for share in CONVERSION_SCENARIOS:
        moved = meta["zero_cart_sessions"] * share
        extra_orders = moved * lift_pp
        revenue = extra_orders * aov
        margin = revenue * ASSUMED_GROSS_MARGIN

        annual_orders = extra_orders / months * 12
        annual_margin = margin / months * 12
        order_growth = annual_orders / (econ["訂單數"] / months * 12)

        rows.append({
            "轉化比例": share,
            "被推動的 session": moved,
            "增額訂單(5個月)": extra_orders,
            "年化增額訂單": annual_orders,
            "訂單成長%": order_growth * 100,
            "年化增額毛利": annual_margin,
            "首年淨效益": annual_margin - ASSUMED_DEV_COST,
            "合理性": "合理" if order_growth <= 0.15
                      else "偏樂觀" if order_growth <= 0.30 else "不可信",
        })
    return pd.DataFrame(rows)


def breakeven_conversion(meta: dict, econ: pd.Series) -> float:
    """開發成本要回本，需要推動多少比例的零加購 session。"""
    lift_pp = meta["one_cart_rate"] - meta["zero_cart_rate"]
    aov = econ["平均訂單金額 AOV"]
    months = econ["月數"]

    margin_per_session = lift_pp * aov * ASSUMED_GROSS_MARGIN
    annual_pool = meta["zero_cart_sessions"] / months * 12
    return ASSUMED_DEV_COST / (annual_pool * margin_per_session)


def main() -> None:
    econ = pd.read_csv(
        TABLE_DIR / "baseline_economics.csv", encoding="utf-8-sig"
    ).iloc[0]

    curve, meta = cart_curve()
    curve.to_csv(TABLE_DIR / "cart_curve.csv", index=False, encoding="utf-8-sig")

    print(f"{'=' * 62}")
    print("  觀察窗內加購件數 vs 之後的購買率（實測）")
    print(f"{'=' * 62}")
    print(curve.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    jump = (meta["one_cart_rate"] - meta["zero_cart_rate"]) * 100
    print(f"\n  0 → 1 件的跳躍：{meta['zero_cart_rate']:.2%} → "
          f"{meta['one_cart_rate']:.2%}（+{jump:.2f} 個百分點，"
          f"{meta['one_cart_rate'] / meta['zero_cart_rate']:.2f} 倍）")
    print(f"  加購 2 件以上幾乎沒有額外提升 —— 關鍵是「有沒有加購」，不是「加幾件」。")
    print(f"\n  零加購 session：{meta['zero_cart_sessions']:,} "
          f"（佔合格 session 的 {meta['zero_cart_sessions'] / meta['n_sessions']:.1%}）")
    print("  這就是探索優化要處理的池子。")

    # ---- 損益兩平 ----
    be = breakeven_conversion(meta, econ)
    print(f"\n{'=' * 62}")
    print("  損益兩平（一次性開發成本）")
    print(f"{'=' * 62}")
    print(f"  假設開發成本            {ASSUMED_DEV_COST:>18,}")
    print(f"  需推動的零加購比例      {be:>17.3%}")
    print(f"  對應 session 數(年)     {meta['zero_cart_sessions'] / econ['月數'] * 12 * be:>18,.0f}")
    print("\n  對照：10% 折扣方案需要 2.34% 的挽回率才打平，且該成本每年重複發生；")
    print("        探索優化的開發成本只付一次，之後每年的效益都是淨賺。")

    # ---- 情境 ----
    sc = scenarios(meta, econ)
    sc.to_csv(TABLE_DIR / "opportunity_scenarios.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 62}")
    print("  情境：探索優化把 X% 的零加購 session 推到加購一件")
    print(f"{'=' * 62}")
    print(f"  {'轉化率':>7}{'年化增額訂單':>14}{'訂單成長':>10}"
          f"{'年化增額毛利':>16}{'首年淨效益':>16}{'合理性':>9}")
    for _, r in sc.iterrows():
        print(f"  {r['轉化比例']:>6.0%}{r['年化增額訂單']:>14,.0f}"
              f"{r['訂單成長%']:>9.1f}%{r['年化增額毛利']:>16,.0f}"
              f"{r['首年淨效益']:>16,.0f}{r['合理性']:>9}")

    print("\n  [!] 這是效果上限，不是預測值。")
    print("      加購 1 件的 session 購買率較高，部分原因是那群人本來就更想買。")
    print("      把人推去加購不會自動獲得同樣的購買率，真實效果必須靠 A/B test 測得。")
    print("\n[OK] 完成。表格已存至 outputs/tables/")


if __name__ == "__main__":
    main()
