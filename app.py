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

from src import charts, discovery
from src.config import (
    ASSUMED_GROSS_MARGIN,
    MODEL_DIR,
    OBSERVATION_WINDOW_MINUTES,
    TABLE_DIR,
)
from src.sephora_prep import NEED_KEYWORDS

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


@st.cache_data
def load_skincare():
    """模組 4 的商品與同膚質評分表。快取避免每次互動都重讀 parquet。"""
    return discovery.load_data()


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
    <div style="background:linear-gradient(90deg,#F5F3FF,#FDF2F8);
                border-left:5px solid #7C3AED;
                padding:20px 24px;border-radius:8px;margin:8px 0 20px">
      <div style="font-size:13px;font-weight:700;color:#7C3AED;
                  letter-spacing:1px;margin-bottom:6px">結論與建議</div>
      <div style="font-size:20px;font-weight:700;color:#1E293B;line-height:1.5">
        別再加碼折扣。把資源投在「讓人加到第一件商品」，
        預估首年淨效益 +123,448。
      </div>
      <div style="font-size:15px;color:#475569;margin-top:10px;line-height:1.8">
        87% 的加購沒有成交，但放棄行為與價格無關，因此折扣挽回在合理假設下
        <b>全數虧損（年化 −382,511 ～ −96,849）</b>。<br>
        真正的斷點在更前面：<b>57.6% 的 session 連一件都沒加</b>，
        而加到第一件會讓購買率從 7.34% 跳到 20.23%。
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 決策鏈：讓人三秒看懂整個論證
chain = [
    ("問題", "加購後放棄 87%", "419 萬組未成交", "#EF4444"),
    ("排除", "折扣不是解方", "合理情境全數虧損", "#F59E0B"),
    ("解方", "推動第一件加購", "購買率 7.3% → 20.2%", "#7C3AED"),
    ("效益", "首年 +123,448", "打平只需 0.98% 轉化", "#10B981"),
]
cols = st.columns(len(chain))
for col, (tag, head, sub, color) in zip(cols, chain):
    col.markdown(
        f"""
        <div style="border-top:4px solid {color};background:#F8FAFC;
                    padding:14px 16px;border-radius:0 0 8px 8px;height:120px">
          <div style="font-size:12px;font-weight:700;color:{color};
                      letter-spacing:2px">{tag}</div>
          <div style="font-size:16px;font-weight:700;color:#1E293B;
                      margin-top:6px;line-height:1.4">{head}</div>
          <div style="font-size:12px;color:#64748B;margin-top:6px">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

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

tab1, tab3, tab5, tab2, tab4 = st.tabs(
    ["① 問題在哪", "② 折扣行不行", "③ 那該做什麼",
     "④ 模型細節", "⑤ 怎麼驗證與限制"]
)

# ---------------------------------------------------------------- 漏斗
with tab1:
    st.plotly_chart(charts.price_contrast(), width='stretch')
    st.info(
        "**這張圖排除了最直覺的假設。** 一般認為「東西貴所以放棄」，"
        "但資料顯示放棄率在各價格帶幾乎持平（85.6%–88.1%），"
        "而瀏覽→加購率卻從 19.7% 腰斬到 9.5%。"
        "兩個階段對價格的敏感度完全不同 —— 這是後面判定「折扣不是解方」的第一個依據，"
        "也把注意力導向更前面的加購環節（見第 ③ 頁）。"
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
    st.markdown("## 模型細節")
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
    st.markdown("## 折扣行不行：答案是不行")
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

# ---------------------------------------------------------------- 探索引擎
with tab5:
    st.markdown("## 該做什麼：讓人加到第一件商品")

    st.plotly_chart(charts.cart_curve_chart(), width="stretch")
    st.success(
        "**這張圖決定了整個方案的方向。** 購買率的跳躍完全發生在 0 → 1 件"
        "（7.34% → 20.23%，2.76 倍），加第 2、3 件幾乎沒有額外效果。"
        "所以目標不是「讓人多買」，而是**讓那 57.6% 一件都沒加的人加到第一件**。"
    )

    st.markdown("### 這能改善多少")
    opp = load_table("opportunity_scenarios")

    q1, q2, q3 = st.columns(3)
    row5 = opp[opp["轉化比例"] == 0.05].iloc[0]
    q1.metric("打平所需轉化率", "0.98%", "把零加購推到加購一件")
    q2.metric("5% 轉化的首年淨效益", f"+{row5['首年淨效益']:,.0f}",
              f"訂單成長 {row5['訂單成長%']:.1f}%")
    q3.metric("對照：10% 折扣方案", "−96,849 ～ −382,511",
              "合理情境全數虧損", delta_color="inverse")

    show_opp = opp[["轉化比例", "年化增額訂單", "訂單成長%",
                    "年化增額毛利", "首年淨效益", "合理性"]].copy()
    show_opp["轉化比例"] = (show_opp["轉化比例"] * 100).map("{:.0f}%".format)
    st.dataframe(
        show_opp, width="stretch", hide_index=True,
        column_config={
            "年化增額訂單": st.column_config.NumberColumn(format="%.0f"),
            "訂單成長%": st.column_config.NumberColumn(format="%.1f%%"),
            "年化增額毛利": st.column_config.NumberColumn(format="%.0f"),
            "首年淨效益": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "首年淨效益已扣除假設的一次性開發成本 30,000（約 3 個工程師月）。"
        "第二年起沒有這筆成本，效益等於年化增額毛利。"
    )

    st.plotly_chart(charts.solution_comparison(), width="stretch")

    st.error(
        "**這是效果上限，不是預測值。** 加購 1 件的 session 購買率較高，"
        "部分原因是那群人本來就更想買 —— 把人推去加購不會自動獲得同樣的購買率。"
        "真實效果必須靠 A/B test 測得，設計見第 ⑤ 頁。"
    )

    st.divider()
    st.markdown("### 怎麼推動第一件加購：膚況導向的商品探索")
    st.caption(
        "既然關鍵是「找到一件想買的」，那商品排序就是最直接的槓桿。"
        "以下用另一份含膚質評論的資料，建立解決方案原型。"
    )
    st.warning(
        "**資料邊界**：本頁使用 Sephora 商品與評論資料（2023 年爬取，"
        "2,420 項 Skincare 商品、109 萬則評論），與前面的 REES46 行為資料"
        "（2019–2020）分屬不同來源，**不可串接**。這裡建立的是解決方案原型，"
        "不是對同一批用戶的分析。"
    )

    products, cohorts = load_skincare()

    f1, f2, f3, f4 = st.columns([1, 1.2, 1, 1])
    skin = f1.selectbox(
        "你的膚質", discovery.SKIN_TYPES,
        format_func=lambda s: discovery.SKIN_TYPE_ZH[s],
    )
    need = f2.selectbox("主要需求", ["（不限）"] + list(NEED_KEYWORDS))
    budget = f3.slider("預算上限 (USD)", 10, 200, 60, 5)
    avoid = f4.checkbox("避開刺激成分", value=False,
                        help="排除含變性酒精或香料的商品")

    need_arg = None if need == "（不限）" else need

    personal = discovery.rank_for(
        products, cohorts, skin, need=need_arg,
        budget=budget, avoid_irritants=avoid, top_n=10,
    )
    generic = discovery.rank_global(
        products, cohorts, need=need_arg,
        budget=budget, avoid_irritants=avoid, top_n=10,
    )

    if personal.empty:
        st.info("此條件下沒有商品，試著放寬預算或需求。")
    else:
        shared = len(set(personal["product_id"]) & set(generic["product_id"]))
        o1, o2, o3 = st.columns(3)
        o1.metric("與全站排序重疊", f"{shared} / 10")
        o2.metric("被換掉的商品", f"{10 - shared} 項",
                  "全站排序看不到這些", delta_color="off")
        o3.metric("符合條件的商品數", f"{len(products):,} 中篩選")

        show = personal[[
            "product_name", "brand_name", "price_final",
            "rec_score", "rating_score", "rating_delta",
            "n_cohort", "irritant_score",
        ]].copy()
        # ProgressColumn 的 format 是 printf 風格，不會自動把 0-1 轉成百分比。
        # 直接餵 0.9 會顯示成「0.9%」，所以先自己乘 100。
        show["rec_score"] = show["rec_score"] * 100
        show = show.rename(columns={
            "product_name": "商品", "brand_name": "品牌",
            "price_final": "價格", "rec_score": "同膚質推薦率",
            "rating_score": "同膚質評分", "rating_delta": "vs 該品平均",
            "n_cohort": "同膚質評論數", "irritant_score": "刺激分數",
        })
        st.dataframe(
            show, width="stretch", hide_index=True,
            column_config={
                "同膚質推薦率": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0.0, max_value=100.0),
                "價格": st.column_config.NumberColumn(format="$%.0f"),
                "同膚質評分": st.column_config.NumberColumn(format="%.2f"),
                "vs 該品平均": st.column_config.NumberColumn(
                    format="%+.2f", help="正值代表這支產品對你的膚質優於它自己的平均"),
            },
        )
        st.caption(
            "「vs 該品平均」是本頁最重要的欄位：它問的是「這支產品對你的膚質，"
            "比它自己的整體表現好多少」，而不是「它是不是熱門商品」。"
        )

        with st.expander("對照：不分膚質的全站排序會推薦什麼"):
            swapped = set(generic["product_id"]) - set(personal["product_id"])
            g = generic[["product_name", "brand_name", "price_final",
                         "rating_all_s", "rec_all_s"]].copy()
            g["個人化後被剔除"] = generic["product_id"].isin(swapped).map(
                {True: "是", False: ""})
            st.dataframe(g.rename(columns={
                "product_name": "商品", "brand_name": "品牌",
                "price_final": "價格", "rating_all_s": "全站評分",
                "rec_all_s": "全站推薦率",
            }), width="stretch", hide_index=True)

    st.divider()
    st.plotly_chart(charts.overlap_heatmap(), width="stretch")
    st.plotly_chart(charts.cohort_spread_chart(), width="stretch")
    st.info(
        "**必須誠實說明的一點**：膚質之間的評分差距中位數只有 0.19 星，"
        "絕對值並不大。排序之所以被大幅打亂，是因為頂端商品的評價高度同質"
        "（全站平均 4.30，82% 的評論給 4–5 星），全站排序本身幾乎沒有鑑別度。"
        "膚質訊號雖小，卻足以決定誰排在前面 —— 而使用者只看得到前面幾個。"
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
    "程式碼：[github.com/ruru0109lee-cpu/beauty-ecommerce-decision-system](https://github.com/ruru0109lee-cpu/beauty-ecommerce-decision-system)"
)
