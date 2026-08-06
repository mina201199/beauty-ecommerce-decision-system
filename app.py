"""電商轉換漏斗診斷儀表板。

啟動：
    streamlit run app.py

【設計原則】
第一屏就要給結論，不要先給圖表。看的人（面試官、營運主管）只有 30 秒
決定要不要繼續看下去 —— 這 30 秒要用來講「發現了什麼」，不是「用了什麼技術」。
"""

import json

import pandas as pd
import streamlit as st

from src import charts
from src.config import (
    ASSUMED_GROSS_MARGIN,
    MODEL_DIR,
    OBSERVATION_WINDOW_MINUTES,
    TABLE_DIR,
)

st.set_page_config(
    page_title="電商轉換漏斗診斷系統",
    page_icon="📉",
    layout="wide",
)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"{name}.csv", encoding="utf-8-sig")


@st.cache_data
def load_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


econ = load_table("baseline_economics").iloc[0]
funnel = load_table("funnel_overall").set_index("指標")["值"]
metrics = load_metrics()
test_key = "測試(2月)"

# ---------------------------------------------------------------- 標題
st.title("電商轉換漏斗診斷與流失預警系統")
st.caption(
    "資料來源：Kaggle / REES46 Open CDP — 某美妝電商 2019/10–2020/02 "
    "共 2,069 萬筆真實用戶行為"
)

st.markdown(
    """
    <div style="background:#FFF4F2;border-left:5px solid #E8684A;
                padding:18px 22px;border-radius:6px;margin:8px 0 26px">
      <div style="font-size:19px;font-weight:700;color:#3C4043">
        最大的漏水點在購物車，但問題不是價格
      </div>
      <div style="font-size:15px;color:#5F6368;margin-top:8px;line-height:1.7">
        87.0% 的加購最終沒有成交。但放棄率在五個價格帶之間僅相差 2.5 個百分點 ——
        <b>價格決定的是「要不要加入購物車」，不是「加了之後會不會結帳」</b>。
        因此把折扣預算投在購物車挽回的效益有限：試算顯示 10% 折扣方案在合理假設下
        仍為虧損，建議改為 5%。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("加購後放棄率", f"{funnel['加購後放棄%']:.1f}%",
          f"{funnel['放棄配對數']:,.0f} 組未成交", delta_color="inverse")
c2.metric("平均訂單金額", f"{econ['平均訂單金額 AOV']:.2f}",
          f"每單 {econ['每筆訂單平均件數']:.1f} 件")
c3.metric("Session 購買轉換率", f"{econ['session 購買轉換率']:.2%}",
          f"{econ['訂單數']:,.0f} 筆訂單")
if metrics:
    r = metrics["results"][test_key]
    c4.metric("模型 PR-AUC", f"{r['pr_auc']:.4f}",
              f"baseline {r['baseline']:.2%} 的 {r['pr_auc'] / r['baseline']:.2f} 倍")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 漏斗診斷", "② 購買意圖預測", "③ 商業決策試算", "④ 資料限制"]
)

# ---------------------------------------------------------------- 漏斗
with tab1:
    st.plotly_chart(charts.price_contrast(), width='stretch')
    st.info(
        "**這是本專案最重要的一張圖。** 一般直覺認為「東西貴所以放棄」，"
        "但資料顯示放棄率在各價格帶幾乎持平（85.6%–88.1%），"
        "而瀏覽→加購率卻從 19.7% 腰斬到 9.5%。"
        "兩個階段對價格的敏感度完全不同，因此介入點應該在商品頁而非購物車。"
    )
    st.plotly_chart(charts.funnel_chart(), width='stretch')

    with st.expander("依品牌檢視流失熱點"):
        st.dataframe(load_table("abandonment_hotspots_top20"),
                     width='stretch', hide_index=True)
        st.caption(
            "門檻：配對數 ≥ 500 且加購數 ≥ 200。"
            "放棄率的分母是加購數，若只用配對數設門檻，"
            "會讓「只有 30 次加購、放棄率 96.7%」這種雜訊登上榜首。"
        )

# ---------------------------------------------------------------- 模型
with tab2:
    st.markdown(
        f"**預測問題**：一個 session 已經瀏覽了 "
        f"{OBSERVATION_WINDOW_MINUTES} 分鐘、目前還沒下單，他最終會不會買？"
    )
    st.caption(
        f"{OBSERVATION_WINDOW_MINUTES} 分鐘內已購買的 session 直接排除 —— "
        "已經轉換了就沒有東西可預測，留著就是資料洩漏。"
        "訓練 2019/10–12、驗證 2020/01、測試 2020/02，採時間切分而非隨機切分。"
    )

    if metrics:
        m1, m2, m3 = st.columns(3)
        va, te = metrics["results"]["驗證(1月)"], metrics["results"][test_key]
        m1.metric("測試集 PR-AUC", f"{te['pr_auc']:.4f}",
                  f"baseline {te['baseline']:.2%}")
        m2.metric("測試集 ROC-AUC", f"{te['roc_auc']:.4f}",
                  "0.95 以上會觸發洩漏警報")
        m3.metric("驗證 vs 測試差距", f"{abs(va['pr_auc'] - te['pr_auc']):.4f}",
                  "差距極小 = 無過擬合")

    st.plotly_chart(charts.decile_lift(), width='stretch')
    st.plotly_chart(charts.feature_importance(), width='stretch')
    st.warning(
        "**重要度必須看 gain，不能看預設的 split。** "
        "split 只數特徵被拿來分裂幾次，會嚴重偏袒價格、時間間隔這類連續型高基數特徵。"
        "本專案兩者都輸出：`n_cart` 的 split 排名接近墊底，gain 卻是第一名（34.9%），"
        "兩種指標會導向完全相反的結論。"
    )
    st.plotly_chart(charts.window_tradeoff_chart(), width='stretch')

# ---------------------------------------------------------------- 商業
with tab3:
    st.markdown("### 折扣方案試算")
    st.caption(
        "模型鎖定比例與涵蓋率取自模組 2 的實際 lift curve，非假設值。"
        "挽回率無法從歷史資料推得，只能靠 A/B test 實測 —— "
        "因此本頁的核心產出是**損益兩平門檻**，而非單一預估值。"
    )

    s1, s2 = st.columns(2)
    discount = s1.slider("折扣幅度 (%)", 3, 20, 10, 1) / 100
    recovery = s2.slider("假設挽回率 (%)", 0.5, 5.0, 2.0, 0.1) / 100

    targeted = econ["總 session 數"] * 0.20
    capture = metrics.get("capture_rate_top20", 0.60)
    organic = econ["訂單數"] * capture
    non_buyers = max(targeted - organic, 0.0)
    aov = econ["平均訂單金額 AOV"]

    incremental = non_buyers * recovery
    gain = incremental * aov * (ASSUMED_GROSS_MARGIN - discount)
    waste = organic * aov * discount
    net_annual = (gain - waste) / econ["月數"] * 12
    breakeven = organic * discount / (non_buyers * (ASSUMED_GROSS_MARGIN - discount))
    order_lift = incremental / econ["訂單數"]

    k1, k2, k3 = st.columns(3)
    k1.metric("年化淨效益", f"{net_annual:,.0f}",
              "獲利" if net_annual > 0 else "虧損",
              delta_color="normal" if net_annual > 0 else "inverse")
    k2.metric("打平所需挽回率", f"{breakeven:.2%}",
              f"目前假設 {recovery:.2%}")
    k3.metric("隱含訂單成長", f"{order_lift:.1%}",
              "合理" if order_lift <= 0.15 else "偏樂觀" if order_lift <= 0.30 else "不可信",
              delta_color="off")

    if order_lift > 0.30:
        st.error(
            f"此情境隱含訂單量成長 {order_lift:.1%}，沒有促銷方案做得到。"
            "數字雖然算得出來，但不應納入決策。"
        )
    elif recovery < breakeven:
        st.warning(
            f"假設的挽回率 {recovery:.2%} 低於打平門檻 {breakeven:.2%}，此方案虧損。"
            f"若折扣降至 5%，門檻可降到約 1.02%。"
        )
    else:
        st.success(f"挽回率 {recovery:.2%} 高於打平門檻 {breakeven:.2%}，方案成立。")

    st.plotly_chart(charts.breakeven_chart(), width='stretch')
    st.plotly_chart(charts.sensitivity_chart(), width='stretch')

    st.markdown("#### 為什麼模型越準，這個方案反而虧越多")
    st.markdown(
        f"""
        被鎖定的前 20% session，購買率為 **{organic / targeted:.2%}**，
        是全站平均 {econ['session 購買轉換率']:.2%} 的
        **{(organic / targeted) / econ['session 購買轉換率']:.2f} 倍**。
        這些人**本來就會買**，折扣對他們是純粹的毛利流失。

        這暴露一個根本性的錯配：購買意圖模型預測的是「**誰會買**」（propensity），
        但發折扣需要知道的是「**誰因為折扣才會買**」（uplift）。
        模型越擅長找出會買的人，折扣浪費就越嚴重。

        真正要解決需要 uplift 模型，而 uplift 模型需要實驗組／對照組資料。
        本資料集為觀察性資料，不具備此條件 —— 這是本專案無法克服的限制。
        """
    )

# ---------------------------------------------------------------- 限制
with tab4:
    st.markdown("### 資料限制與分析邊界")
    st.caption("主動揭露限制，比多做三個模型更能證明分析的可信度。")

    st.error(
        "**74.14% 的加購配對在同一 session 內沒有瀏覽記錄。** "
        "`view` 事件記錄不完整，因此「瀏覽→加購」的絕對數值不可直接解讀為真實轉換率。"
        "本專案的核心指標因此鎖定在「加購→購買」——`cart` 與 `purchase` 都記錄完整。"
    )
    st.info(
        "不過這個缺失並非隨機：無瀏覽直接加購的比例隨價格單調下降"
        "（最低價 81.79% → 最高價 61.01%），符合「低價熟客直接補貨、"
        "高價會先看再決定」的消費行為。"
    )

    st.markdown(
        """
        | 限制 | 影響 | 處理方式 |
        |---|---|---|
        | `category_code` 覆蓋率僅 1.81% | 品類分析幾乎不可行 | 不列入主要結論 |
        | `brand` 覆蓋率 58.09% | 品牌排行有偏誤風險 | 標註覆蓋率並設樣本門檻 |
        | 2020/02 的 `price <= 0` 暴增 10 倍 | 測試集含資料異常 | 已於清洗階段移除並記錄 |
        | 觀察性資料，無實驗組 | 無法建立 uplift 模型 | 改以損益兩平分析處理 |
        | Session 可能跨月份切斷 | 極少數 session 特徵不完整 | 影響量級可忽略，未做處理 |
        """
    )

    st.markdown("#### 若要實際上線，建議的驗證設計")
    st.markdown(
        """
        1. **A/B test**：對模型判定的前 20% session 隨機分派實驗組（發券）與對照組（不發券）
        2. **主要指標**：兩組的最終轉換率差異，即真實的挽回率
        3. **樣本量**：以偵測 2% 絕對差異、power 0.8 估算所需 session 數
        4. **護欄指標**：整體毛利率不得下降、對照組轉換率需維持穩定
        5. **期間**：至少涵蓋兩個完整週期，避免週間效應干擾
        """
    )

st.divider()
st.caption(
    "資料集：[eCommerce Events History in Cosmetics Shop]"
    "(https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop)"
    " by REES46 Marketing Platform　|　"
    "程式碼：[github.com/ruru0109lee-cpu/ML](https://github.com/ruru0109lee-cpu/ML)"
)
