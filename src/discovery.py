"""模組 4：膚況導向商品探索引擎。

用法：
    python -m src.discovery

【這個模組要證明的事】
現行電商普遍用「全站評分」排序商品。但如果同膚質的評價與全站評價差很多，
那個排序就是在把不適合的商品推給人。

本模組計算的核心指標是：**同膚質排序與全站排序的 Top-N 重疊率**。
重疊率越低，代表現行排序方式誤導越多人，個人化的價值就越大。
這個數字不依賴任何假設，直接從資料算得出來。

【接回前面的分析】
模組 1 發現槓桿在「瀏覽→加購」這一步，模組 5 證明折扣挽回不划算。
探索優化正是一個作用在該環節、且沒有毛利成本的介入手段 ——
折扣要 2.34% 挽回率才打平，探索優化的損益兩平門檻趨近於零。
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TABLE_DIR
from src.sephora_prep import NEED_KEYWORDS, OUT_DIR

SKIN_TYPES = ["combination", "dry", "normal", "oily"]

SKIN_TYPE_ZH = {
    "combination": "混合肌",
    "dry": "乾性肌",
    "normal": "中性肌",
    "oily": "油性肌",
}

# 排序權重。刻意讓同膚質推薦率主導 —— 那才是個人化的訊號，
# 全站評分只作為次要參考。
W_COHORT_REC = 0.45
W_COHORT_RATING = 0.30
W_POPULARITY = 0.15
W_LOW_IRRITANT = 0.10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.read_parquet(OUT_DIR / "products.parquet")
    cohorts = pd.read_parquet(OUT_DIR / "cohorts.parquet")
    return products, cohorts


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def score(df: pd.DataFrame, rec_col: str, rating_col: str) -> pd.Series:
    """同一套加權公式，只換評分來源。

    【為什麼一定要共用這個函式】
    第一版的對照組用了不同的權重，而且少了低刺激那一項。
    這樣測出來的「重疊率」同時混雜了兩件事：個人化訊號的效果，
    以及我用了不同公式。那樣的比較是無效的 —— 要測個人化的價值，
    就只能讓「評分來源」這一個變數不同，其他全部固定。
    """
    return (
        W_COHORT_REC * minmax(df[rec_col])
        + W_COHORT_RATING * minmax(df[rating_col])
        + W_POPULARITY * minmax(np.log1p(df["loves_count"]))
        + W_LOW_IRRITANT * (1 - minmax(df["irritant_score"]))
    )


def rank_for(
    products: pd.DataFrame,
    cohorts: pd.DataFrame,
    skin_type: str,
    need: str | None = None,
    budget: float | None = None,
    avoid_irritants: bool = False,
    top_n: int = 10,
) -> pd.DataFrame:
    """依膚質與需求排序商品。"""
    coh = cohorts[cohorts["skin_type"] == skin_type]
    df = products.merge(coh, on="product_id", how="inner")

    if need:
        df = df[df["needs"].str.contains(need, na=False)]
    if budget is not None:
        df = df[df["price_final"] <= budget]
    if avoid_irritants:
        df = df[~df["has_drying_alcohol"] & ~df["has_fragrance"]]

    if df.empty:
        return df

    df = df.copy()
    df["score"] = score(df, "rec_score", "rating_score")
    return df.nlargest(top_n, "score")


def rank_global(
    products: pd.DataFrame,
    cohorts: pd.DataFrame,
    need: str | None = None,
    budget: float | None = None,
    avoid_irritants: bool = False,
    top_n: int = 10,
) -> pd.DataFrame:
    """對照組：不分膚質的全站排序。

    公式與 rank_for 完全相同，唯一差別是評分來源用全站平均而非同膚質平均。
    這樣兩者的差異才能歸因於個人化本身。
    """
    overall = (
        cohorts.groupby("product_id")[["rating_all_s", "rec_all_s"]].first().reset_index()
    )
    df = products.merge(overall, on="product_id", how="inner")

    if need:
        df = df[df["needs"].str.contains(need, na=False)]
    if budget is not None:
        df = df[df["price_final"] <= budget]
    if avoid_irritants:
        df = df[~df["has_drying_alcohol"] & ~df["has_fragrance"]]
    if df.empty:
        return df

    df = df.copy()
    df["score"] = score(df, "rec_all_s", "rating_all_s")
    return df.nlargest(top_n, "score")


def overlap_analysis(
    products: pd.DataFrame, cohorts: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """核心指標：同膚質排序與全站排序的 Top-N 重疊率。

    重疊率低 = 現行的全站排序推給你的東西，跟真正適合你膚質的東西不一樣。
    """
    rows = []
    scenarios = [(None, "全部商品")] + [(n, n) for n in NEED_KEYWORDS]

    for need, label in scenarios:
        g = rank_global(products, cohorts, need=need, top_n=top_n)
        if g.empty:
            continue
        g_ids = set(g["product_id"])

        for st in SKIN_TYPES:
            p = rank_for(products, cohorts, st, need=need, top_n=top_n)
            if p.empty:
                continue
            shared = len(g_ids & set(p["product_id"]))
            rows.append({
                "需求": label,
                "膚質": SKIN_TYPE_ZH[st],
                "重疊數": shared,
                "重疊率%": shared / top_n * 100,
                "被換掉的商品": top_n - shared,
            })

    return pd.DataFrame(rows)


def cohort_spread(cohorts: pd.DataFrame) -> pd.DataFrame:
    """同一支產品在不同膚質之間的評價差異有多大。"""
    piv = cohorts.pivot_table(
        index="product_id", columns="skin_type", values="rec_score"
    ).dropna()
    spread = (piv.max(axis=1) - piv.min(axis=1)) * 100

    rating_piv = cohorts.pivot_table(
        index="product_id", columns="skin_type", values="rating_score"
    ).dropna()
    rating_spread = rating_piv.max(axis=1) - rating_piv.min(axis=1)

    return pd.DataFrame({
        "指標": ["推薦率最大差距(百分點)", "評分最大差距(星)"],
        "中位數": [spread.median(), rating_spread.median()],
        "P75": [spread.quantile(0.75), rating_spread.quantile(0.75)],
        "P90": [spread.quantile(0.90), rating_spread.quantile(0.90)],
        "最大": [spread.max(), rating_spread.max()],
    })


def build_similarity(products: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """成分相似度矩陣，用於找替代品。"""
    df = products[products["ingredient_text"].str.len() > 20]
    tfidf = TfidfVectorizer(max_features=5000, token_pattern=r"[a-z][a-z\- ]{2,}")
    mat = tfidf.fit_transform(df["ingredient_text"])
    return cosine_similarity(mat), df["product_id"].reset_index(drop=True)


def find_alternatives(
    products: pd.DataFrame, sim: np.ndarray, ids: pd.Index,
    product_id: str, max_price: float | None = None, top_n: int = 5,
) -> pd.DataFrame:
    """找成分最接近的替代品，可限定更便宜的。"""
    pos = ids[ids == product_id].index
    if len(pos) == 0:
        return pd.DataFrame()

    scores = sim[pos[0]]
    order = np.argsort(-scores)[1:200]
    cand = products.set_index("product_id").loc[ids.iloc[order]].reset_index()
    cand["成分相似度"] = scores[order]

    if max_price is not None:
        cand = cand[cand["price_final"] < max_price]
    return cand.head(top_n)


def main() -> None:
    products, cohorts = load_data()
    print(f"[*] 商品 {len(products):,} 項　同膚質評分 {len(cohorts):,} 組\n")

    # ---- 膚質間的評價差異 ----
    spread = cohort_spread(cohorts)
    spread.to_csv(TABLE_DIR / "skin_cohort_spread.csv", index=False, encoding="utf-8-sig")
    print(f"{'=' * 58}")
    print("  同一支產品在不同膚質之間的評價差距")
    print(f"{'=' * 58}")
    print(spread.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- 核心指標：排序重疊率 ----
    ov = overlap_analysis(products, cohorts)
    ov.to_csv(TABLE_DIR / "ranking_overlap.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 58}")
    print("  Top 10 排序重疊率：同膚質 vs 全站")
    print(f"{'=' * 58}")
    piv = ov.pivot_table(index="需求", columns="膚質", values="重疊率%")
    print(piv.to_string(float_format=lambda x: f"{x:,.0f}%"))

    mean_ov = ov["重疊率%"].mean()
    mean_swap = ov["被換掉的商品"].mean()
    print(f"\n  平均重疊率 {mean_ov:.1f}%　平均被換掉 {mean_swap:.1f} / 10 項")
    print(f"  解讀：以全站評分排序時，Top 10 有 {mean_swap:.1f} 項")
    print("        不是該膚質族群評價最好的商品。")

    # ---- 範例輸出 ----
    print(f"\n{'=' * 58}")
    print("  範例：油性肌 × 痘痘粉刺 × 預算 50 美元")
    print(f"{'=' * 58}")
    demo = rank_for(products, cohorts, "oily", need="痘痘粉刺", budget=50, top_n=5)
    cols = ["product_name", "brand_name", "price_final", "rec_score",
            "rating_score", "rating_delta", "n_cohort"]
    print(demo[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    demo.to_csv(TABLE_DIR / "discovery_demo.csv", index=False, encoding="utf-8-sig")

    print("\n[OK] 模組 4 完成。表格已存至 outputs/tables/")


if __name__ == "__main__":
    main()
