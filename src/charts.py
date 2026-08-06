"""圖表模組：每張圖負責讓一個結論被看懂。

設計原則：
  一張圖只講一件事，而且那件事要能在三秒內看懂。
  看得懂比好看重要 —— 好看但要讀十秒才懂結論的圖，是失敗的圖。
"""

import pandas as pd
import plotly.graph_objects as go

from src.config import TABLE_DIR

FONT = "Microsoft JhengHei, Noto Sans TC, sans-serif"

# 高飽和配色。題材是美妝電商，用紫→桃紅→金這組帶妝感的漸層，
# 比預設的商務藍灰有記憶點，漏斗的層次感也做得出來。
VIOLET = "#7C3AED"
PINK = "#EC4899"
GOLD = "#F59E0B"
BLUE = "#3B82F6"
RED = "#EF4444"
GREEN = "#10B981"
GREY = "#CBD5E1"
AMBER = "#F59E0B"

FUNNEL_COLORS = [VIOLET, PINK, GOLD]

PRICE_ORDER = ["最低價 20%", "偏低", "中價位", "偏高", "最高價 20%"]


def _base(fig: go.Figure, title: str, subtitle: str = "", height: int = 460) -> go.Figure:
    """統一版面。副標用來寫「所以呢」，這比標題重要。"""
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:#5F6368'>{subtitle}</span>"
    fig.update_layout(
        title={"text": text, "x": 0, "xanchor": "left"},
        font={"family": FONT, "size": 13},
        height=height,
        margin={"l": 60, "r": 30, "t": 90, "b": 50},
        plot_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#DADCE0")
    fig.update_yaxes(gridcolor="#F1F3F4", zeroline=False)
    return fig


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"{name}.csv", encoding="utf-8-sig")


# ------------------------------------------------------------------ 模組 1

def price_contrast() -> go.Figure:
    """本專案的核心發現：一條陡降、一條水平。

    刻意把兩條線畫在同一個 0-100 軸上。用雙 Y 軸可以讓兩條線
    看起來「一樣陡」，那會誤導 —— 而「不陡」正是我們的結論。
    """
    df = load("funnel_by_price_band")
    df["price_band"] = pd.Categorical(df["price_band"], PRICE_ORDER, ordered=True)
    df = df.sort_values("price_band")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["price_band"].astype(str), y=df["加購後放棄%"],
        name="加購後放棄率", mode="lines+markers+text",
        line={"color": RED, "width": 3}, marker={"size": 10},
        text=[f"{v:.1f}%" for v in df["加購後放棄%"]], textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=df["price_band"].astype(str), y=df["瀏覽→加購%"],
        name="瀏覽→加購率", mode="lines+markers+text",
        line={"color": BLUE, "width": 3}, marker={"size": 10},
        text=[f"{v:.1f}%" for v in df["瀏覽→加購%"]], textposition="bottom center",
    ))

    fig.add_annotation(
        x=2, y=93, text="幾乎水平：價格不影響「加了之後買不買」",
        showarrow=False, font={"color": RED, "size": 12},
    )
    fig.add_annotation(
        x=2, y=4, text="陡降一半：價格強烈影響「要不要加購」",
        showarrow=False, font={"color": BLUE, "size": 12},
    )

    fig.update_yaxes(range=[0, 100], title="比率 (%)")
    return _base(
        fig, "價格影響的是加購決策，不是結帳決策",
        "折扣預算應投在商品頁，而非購物車挽回", height=500,
    )


def funnel_chart() -> go.Figure:
    """漏斗必須用巢狀計數畫。

    若直接把「有瀏覽 8.31M / 有加購 4.81M / 有購買 1.28M」丟進漏斗圖，
    Plotly 算出的階段百分比會是 58% 與 27% —— 但真正的瀏覽→加購率是
    14.98%。差異來自 74% 的加購沒有瀏覽記錄，那些配對不該計入「瀏覽後
    加購」。圖上的數字與結論矛盾，是作品集最容易被抓到的破綻。
    """
    df = load("funnel_overall").set_index("指標")["值"]
    vals = [
        float(df["有瀏覽"]),
        float(df["瀏覽且加購"]),
        float(df["瀏覽加購且購買"]),
    ]

    fig = go.Figure(go.Funnel(
        y=["<b>瀏覽商品</b>", "<b>瀏覽後加購</b>", "<b>加購後購買</b>"],
        x=vals,
        textinfo="text",
        text=[f"<b>{v:,.0f}</b>" for v in vals],
        textfont={"size": 17, "color": "white", "family": FONT},
        textposition="inside",
        marker={
            "color": FUNNEL_COLORS,
            "line": {"color": "white", "width": 3},
        },
        connector={"line": {"color": "#E2E8F0", "width": 2}},
        hovertemplate="%{y}<br>配對數 %{x:,.0f}<extra></extra>",
    ))

    # 每一段流失多少 —— 漏斗真正要講的是「掉了多少」，不是「剩下多少」。
    # 只畫剩下的量，圖就會很呆板；把流失標出來，落差才有戲劇性。
    for i in (1, 2):
        lost = vals[i - 1] - vals[i]
        kept = vals[i] / vals[i - 1]
        fig.add_annotation(
            x=vals[0] * 0.62, y=i - 0.5, xref="x", yref="y",
            text=(f"<b style='color:#EF4444;font-size:15px'>▼ {lost:,.0f}</b>"
                  f"<br><span style='color:#94A3B8;font-size:12px'>"
                  f"僅 {kept:.1%} 進入下一階段</span>"),
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.92)", borderpad=8,
            bordercolor="#F1F5F9", borderwidth=1,
        )

    rate = float(df["加購→購買%"])
    nested_rate = vals[2] / vals[1] * 100
    fig.update_yaxes(showgrid=False)
    return _base(
        fig, "轉換漏斗：完整觀察到的路徑",
        f"僅計算三階段都有記錄的配對。先瀏覽再加購者的購買率 "
        f"{nested_rate:.2f}%，高於全部加購的 {rate:.2f}% —— 有瀏覽行為的加購意圖較強",
        height=520,
    )


# ------------------------------------------------------------------ 模組 2

def decile_lift() -> go.Figure:
    df = load("model_decile_lift")
    base = df["購買者"].sum() / df["session 數"].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["十分位"], y=df["該組購買率%"], name="該組購買率",
        marker_color=[BLUE if i < 3 else GREY for i in range(len(df))],
        text=[f"{v:.1f}%" for v in df["該組購買率%"]], textposition="outside",
    ))
    fig.add_hline(
        y=base, line={"color": RED, "dash": "dash"},
        annotation={"text": f"整體平均 {base:.2f}%", "font": {"color": RED}},
    )
    fig.update_xaxes(title="預測分數十分位（1 = 分數最高）", dtick=1)
    fig.update_yaxes(title="實際購買率 (%)")

    top = df.iloc[0]["該組購買率%"] / df.iloc[-1]["該組購買率%"]
    return _base(
        fig, "模型排序能力",
        f"最高分組的購買率是最低分組的 {top:.1f} 倍 —— 這決定「一萬張券發給誰」",
    )


def feature_importance(top_n: int = 12) -> go.Figure:
    df = load("model_feature_importance").head(top_n).iloc[::-1]
    cart_kw = ("cart", "remove")
    colors = [RED if any(k in f for k in cart_kw) else GREY for f in df["特徵"]]

    fig = go.Figure(go.Bar(
        x=df["gain佔比%"], y=df["特徵"], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["gain佔比%"]], textposition="outside",
    ))
    fig.update_xaxes(title="預測貢獻度 gain 佔比 (%)")
    return _base(
        fig, "什麼行為最能預測購買",
        "紅色為購物車相關特徵，合計佔 70% 預測力；純活躍度幾乎無用",
        height=520,
    )


# ------------------------------------------------------------------ 模組 5

def sensitivity_chart() -> go.Figure:
    df = load("impact_sensitivity")
    cmap = {"合理": GREEN, "偏樂觀": AMBER, "不可信": GREY}

    fig = go.Figure(go.Bar(
        x=[f"{v:.0%}" for v in df["挽回率假設"]],
        y=df["淨效益(年化)"],
        marker_color=[cmap[v] for v in df["合理性"]],
        text=[f"{v:,.0f}" for v in df["淨效益(年化)"]], textposition="outside",
        customdata=df[["增額訂單佔現有%", "合理性"]],
        hovertemplate="挽回率 %{x}<br>年化淨效益 %{y:,.0f}"
                      "<br>訂單成長 %{customdata[0]:.1f}%"
                      "<br>判定：%{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": "#3C4043", "width": 1})
    fig.update_xaxes(title="折扣挽回率假設")
    fig.update_yaxes(title="年化淨效益")
    return _base(
        fig, "10% 折扣方案的敏感度分析",
        "綠=合理　黃=偏樂觀　灰=不可信（訂單成長超過 30%，無促銷做得到）",
    )


def breakeven_chart() -> go.Figure:
    df = load("breakeven_by_discount")
    cmap = {"可行": GREEN, "困難": AMBER, "不可行": RED}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"{v:.0%}" for v in df["折扣幅度"]], y=df["兩平挽回率"] * 100,
        mode="lines+markers+text", line={"color": BLUE, "width": 3},
        marker={"size": 14, "color": [cmap[v] for v in df["可行性"]]},
        text=[f"{v * 100:.2f}%" for v in df["兩平挽回率"]], textposition="top center",
        name="打平所需挽回率",
    ))
    fig.update_xaxes(title="折扣幅度")
    fig.update_yaxes(title="打平所需的挽回率 (%)")
    return _base(
        fig, "折扣訂多少才可能划算",
        "綠=可行　黃=困難　紅=不可行。折扣越高，需要的挽回率越難達成",
    )


def window_tradeoff_chart() -> go.Figure:
    df = load("observation_window_tradeoff")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["觀察窗(分)"], y=df["保留的購買者%"], name="保留的購買者%",
        mode="lines+markers", line={"color": RED, "width": 3}, marker={"size": 9},
    ))
    fig.add_trace(go.Scatter(
        x=df["觀察窗(分)"], y=df["正樣本比例%"], name="正樣本比例%",
        mode="lines+markers", line={"color": BLUE, "width": 3}, marker={"size": 9},
    ))
    fig.add_vline(
        x=3, line={"color": GREEN, "dash": "dash"},
        annotation={"text": "採用 3 分鐘", "font": {"color": GREEN}},
    )
    fig.update_xaxes(title="觀察窗長度（分鐘）", type="log", tickvals=df["觀察窗(分)"])
    fig.update_yaxes(title="比率 (%)")
    return _base(
        fig, "觀察窗長度的取捨",
        "窗開越久訊號越多，但會漏掉越多早期就下單的購買者",
    )
