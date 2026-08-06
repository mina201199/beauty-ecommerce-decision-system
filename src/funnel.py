"""模組 1：轉換漏斗診斷。

用法：
    python -m src.funnel

回答三個問題：
  Q1 用戶從瀏覽到購買，在哪一段流失最嚴重？
  Q2 加入購物車後又移除的行為，集中在哪些品牌與價格帶？
  Q3 流失有沒有時間規律（星期／時段）？

分析粒度說明（這段面試會被問，要講得出來）：
  漏斗以 (session, 商品) 配對為單位，不是以事件為單位。
  理由：同一個 session 裡同一件商品可能被看 5 次，用事件數算轉換率會
  嚴重低估。「這個人對這件商品做到哪一步」才是我們要問的問題。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    EVENT_CART,
    EVENT_PURCHASE,
    EVENT_REMOVE,
    EVENT_VIEW,
    FIGURE_DIR,
    INTERIM_DIR,
    MONTHS,
    TABLE_DIR,
)

# 每個維度至少要有這麼多配對才納入分析，避免小樣本雜訊上榜。
MIN_PAIRS = 500

# 「加購後放棄率」的分母是加購數，所以排行榜要用加購數設門檻。
# 用配對數設門檻是錯的 —— 某品牌可能有 700 個配對但只有 30 次加購，
# 那個 96% 放棄率只是 30 分之 29，毫無統計意義。
MIN_CARTS = 200


def build_pairs(month: str) -> pd.DataFrame:
    """把事件流壓成 (session, 商品) 配對，記錄它走到漏斗哪一步。"""
    path = INTERIM_DIR / f"{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}\n請先執行：python -m src.prepare")

    df = pd.read_parquet(path)

    df["viewed"] = df["event_type"] == EVENT_VIEW
    df["carted"] = df["event_type"] == EVENT_CART
    df["removed"] = df["event_type"] == EVENT_REMOVE
    df["purchased"] = df["event_type"] == EVENT_PURCHASE

    # observed=True 很重要：brand / category_code 是 category 型別，
    # 不加這個參數 pandas 會展開所有類別組合，記憶體會直接爆掉。
    pairs = (
        df.groupby(["user_session", "product_id"], observed=True)
        .agg(
            viewed=("viewed", "max"),
            carted=("carted", "max"),
            removed=("removed", "max"),
            purchased=("purchased", "max"),
            brand=("brand", "first"),
            category_code=("category_code", "first"),
            price=("price", "median"),
            first_seen=("event_time", "min"),
        )
        .reset_index()
    )

    pairs["month"] = month
    return pairs


def add_dimensions(pairs: pd.DataFrame) -> pd.DataFrame:
    """加上分析維度：價格帶、星期、時段。"""
    # 價格用五等分位切，不用固定金額 —— 不同品類價格級距差很多，
    # 用分位數才能公平比較
    pairs["price_band"] = pd.qcut(
        pairs["price"],
        q=5,
        labels=["最低價 20%", "偏低", "中價位", "偏高", "最高價 20%"],
        duplicates="drop",
    )
    pairs["weekday"] = pairs["first_seen"].dt.day_name()
    pairs["hour"] = pairs["first_seen"].dt.hour
    return pairs


def funnel_metrics(g: pd.DataFrame) -> pd.Series:
    """計算一組配對的漏斗指標。

    【重要：漏斗必須是巢狀的】
    第一版我把「有瀏覽」和「有加購」當成獨立旗標各自加總，結果算出
    108% 的轉換率。原因是有一部分商品被直接加入購物車卻沒有瀏覽記錄
    （用戶在更早的 session 看過，或追蹤漏掉）。

    正確做法：每一階段的分母只能是「確實走到前一階段」的那群。
    """
    n_pairs = len(g)
    n_view = int(g["viewed"].sum())
    n_cart = int(g["carted"].sum())
    n_remove = int(g["removed"].sum())
    n_purchase = int(g["purchased"].sum())

    # 巢狀計數
    n_view_cart = int((g["viewed"] & g["carted"]).sum())
    n_cart_buy = int((g["carted"] & g["purchased"]).sum())
    n_cart_no_buy = int((g["carted"] & ~g["purchased"]).sum())

    # 沒有瀏覽記錄就直接加購 —— 這是資料品質指標，要一起報告
    n_cart_no_view = int((g["carted"] & ~g["viewed"]).sum())

    return pd.Series({
        "配對數": n_pairs,
        "有瀏覽": n_view,
        "有加購": n_cart,
        "有移除": n_remove,
        "有購買": n_purchase,
        "瀏覽→加購%": n_view_cart / n_view * 100 if n_view else np.nan,
        "加購→購買%": n_cart_buy / n_cart * 100 if n_cart else np.nan,
        "加購後放棄%": n_cart_no_buy / n_cart * 100 if n_cart else np.nan,
        "放棄配對數": n_cart_no_buy,
        "無瀏覽直接加購%": n_cart_no_view / n_cart * 100 if n_cart else np.nan,
    })


def breakdown(pairs: pd.DataFrame, dim: str) -> pd.DataFrame:
    """依指定維度拆解漏斗，並濾掉樣本太少的組。

    兩道門檻缺一不可：配對數確保這個維度值本身夠常見，
    加購數確保「加購後放棄率」這個比率的分母夠大。
    """
    out = (
        pairs.groupby(dim, observed=True)
        .apply(funnel_metrics, include_groups=False)
        .reset_index()
    )
    out = out[(out["配對數"] >= MIN_PAIRS) & (out["有加購"] >= MIN_CARTS)]
    return out.sort_values("加購後放棄%", ascending=False)


def plot_funnel(overall: pd.Series) -> None:
    """畫整體漏斗圖。"""
    fig = go.Figure(
        go.Funnel(
            y=["瀏覽商品", "加入購物車", "完成購買"],
            x=[overall["有瀏覽"], overall["有加購"], overall["有購買"]],
            textinfo="value+percent initial",
            marker={"color": ["#5B8FF9", "#5AD8A6", "#F6BD16"]},
        )
    )
    fig.update_layout(
        title="轉換漏斗（以 session × 商品 配對計）",
        font={"family": "Microsoft JhengHei", "size": 14},
        height=500,
    )
    out = FIGURE_DIR / "funnel_overall.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  已存圖 → {out.name}")


def plot_abandonment_by_price(by_price: pd.DataFrame) -> None:
    """加購後放棄率 × 價格帶 —— 最可能出現反直覺發現的圖。"""
    fig = go.Figure(
        go.Bar(
            x=by_price["price_band"].astype(str),
            y=by_price["加購後放棄%"],
            text=by_price["加購後放棄%"].round(1),
            textposition="outside",
            marker_color="#E8684A",
        )
    )
    fig.update_layout(
        title="加入購物車後放棄率 × 價格帶",
        yaxis_title="加購後放棄率 (%)",
        font={"family": "Microsoft JhengHei", "size": 14},
        height=500,
    )
    out = FIGURE_DIR / "abandonment_by_price.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  已存圖 → {out.name}")


def save(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已存表 → {path.name}")


def main() -> None:
    print("[*] 讀取各月資料並建立 (session, 商品) 配對...")
    frames = []
    for month in MONTHS:
        p = build_pairs(month)
        print(f"    {month}  配對數 {len(p):>12,}")
        frames.append(p)

    pairs = pd.concat(frames, ignore_index=True)
    del frames
    pairs = add_dimensions(pairs)
    print(f"\n[*] 總配對數：{len(pairs):,}")

    # ---- 整體漏斗 ----
    overall = funnel_metrics(pairs)
    print(f"\n{'=' * 52}")
    print("  整體漏斗")
    print(f"{'=' * 52}")
    for k, v in overall.items():
        fmt = f"{v:>14,.2f}" if "%" in k else f"{v:>14,.0f}"
        print(f"  {k:<16}{fmt}")

    save(overall.to_frame("值").reset_index().rename(columns={"index": "指標"}),
         "funnel_overall")
    plot_funnel(overall)

    # ---- 各維度拆解 ----
    # 維度覆蓋率：brand 缺失約 40%、category_code 缺失高達 98%。
    # 這一定要印出來並寫進報告 —— 拿覆蓋率 1.6% 的欄位下結論是嚴重的分析錯誤，
    # 面試官只要問一句「這個品類分析涵蓋多少資料」就會穿幫。
    print("\n  -- 維度覆蓋率 --")
    for col in ("brand", "category_code"):
        cov = pairs[col].notna().mean()
        print(f"  {col:<16}{cov:>8.2%}")

    dims = {
        "price_band": "funnel_by_price_band",
        "brand": "funnel_by_brand",
        "category_code": "funnel_by_category",
        "weekday": "funnel_by_weekday",
        "hour": "funnel_by_hour",
    }
    results = {}
    print()
    for dim, name in dims.items():
        table = breakdown(pairs, dim)
        results[dim] = table
        save(table, name)

    plot_abandonment_by_price(results["price_band"].sort_values("price_band"))

    # ---- 流失熱點 Top 20 ----
    hotspots = (
        results["brand"]
        .nlargest(20, "放棄配對數")
        [["brand", "配對數", "有加購", "放棄配對數", "加購後放棄%", "瀏覽→加購%"]]
    )
    save(hotspots, "abandonment_hotspots_top20")

    print(f"\n{'=' * 52}")
    print("  加購後放棄率最高的品牌（樣本數 >= 500）")
    print(f"{'=' * 52}")
    print(results["brand"].head(10).to_string(index=False))

    print(f"\n{'=' * 52}")
    print("  加購後放棄率 × 價格帶")
    print(f"{'=' * 52}")
    print(results["price_band"].sort_values("price_band").to_string(index=False))

    print("\n[OK] 模組 1 完成。圖表在 outputs/figures/，表格在 outputs/tables/")


if __name__ == "__main__":
    main()
