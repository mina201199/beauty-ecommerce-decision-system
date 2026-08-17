"""針對已發生過的錯誤設置的回歸測試。

執行：
    python -m pytest tests/ -v

【選這幾個測試的理由】
測試不是寫越多越好，是要擋住「已經證明會發生」的錯誤。
本專案開發過程中有三個錯誤實際發生並產出過錯誤結論，
每一個都是寫個測試就能提前擋掉的。這裡就針對那三個。

其餘的驗證（資料品質、分布檢查）已經內建在各模組的輸出報告裡，
不重複用測試覆蓋。
"""

import numpy as np
import pandas as pd
import pytest

from src.funnel import funnel_metrics
from src.impact import breakeven_recovery_rate
from src.sephora_prep import flag_ingredients, ingredient_tokens


# ---------------------------------------------------------------- 錯誤 1
# 曾經發生：把「有瀏覽」「有加購」當獨立旗標各自加總後相除，
# 導致最低價帶算出 108% 的瀏覽→加購率。
# 原因是有商品被直接加入購物車卻沒有瀏覽記錄。

def _pairs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_轉換率不得超過100百分比_即使存在無瀏覽的加購():
    # 3 個配對，其中 2 個沒有瀏覽記錄就直接加購
    g = _pairs([
        {"viewed": True,  "carted": True,  "removed": False, "purchased": False},
        {"viewed": False, "carted": True,  "removed": False, "purchased": True},
        {"viewed": False, "carted": True,  "removed": False, "purchased": False},
    ])
    m = funnel_metrics(g)

    assert m["瀏覽→加購%"] <= 100.0, "巢狀計數失效，轉換率超過 100%"
    assert m["加購→購買%"] <= 100.0
    assert m["加購後放棄%"] <= 100.0
    # 只有 1 個配對有瀏覽，且它有加購 → 100%
    assert m["瀏覽→加購%"] == pytest.approx(100.0)
    # 3 個加購中 2 個沒瀏覽記錄
    assert m["無瀏覽直接加購%"] == pytest.approx(200 / 3)


def test_漏斗巢狀計數不得超過前一階段():
    g = _pairs([
        {"viewed": True, "carted": True,  "removed": False, "purchased": True},
        {"viewed": True, "carted": False, "removed": False, "purchased": False},
        {"viewed": True, "carted": True,  "removed": True,  "purchased": False},
    ])
    m = funnel_metrics(g)

    assert m["瀏覽且加購"] <= m["有瀏覽"]
    assert m["瀏覽加購且購買"] <= m["瀏覽且加購"]


# ---------------------------------------------------------------- 錯誤 2
# 曾經發生：用子字串比對找刺激性酒精，"ethanol" 命中了
# phenoxyethanol（防腐劑）與 triethanolamine（酸鹼調節劑），
# 把含刺激性酒精的商品比例由真實的 16.8% 灌水到 56.6%。

@pytest.mark.parametrize("ingredient", [
    "phenoxyethanol",            # 防腐劑，溫和
    "triethanolamine",           # 酸鹼調節劑
    "cetyl alcohol",             # 脂肪醇，保濕
    "cetearyl alcohol",          # 脂肪醇，乳化劑
    "stearyl alcohol",           # 脂肪醇
])
def test_溫和成分不得被標記為刺激性酒精(ingredient):
    flags = flag_ingredients([ingredient])
    assert not flags["has_drying_alcohol"], f"{ingredient} 被誤判為刺激性酒精"


@pytest.mark.parametrize("ingredient", [
    "alcohol",                   # INCI 單獨列出的 Alcohol 即乙醇
    "alcohol denat.",
    "sd alcohol 40",
    "isopropyl alcohol",
])
def test_真正的刺激性酒精必須被標記(ingredient):
    flags = flag_ingredients([ingredient])
    assert flags["has_drying_alcohol"], f"{ingredient} 應被標記為刺激性酒精"


def test_成分表同時含脂肪醇與變性酒精時仍須標記():
    tokens = ingredient_tokens(
        "['Water, Cetyl Alcohol, Alcohol Denat., Phenoxyethanol, Glycerin']"
    )
    flags = flag_ingredients(tokens)
    assert flags["has_drying_alcohol"], "遮蔽脂肪醇時誤把變性酒精一起遮掉"


def test_歐盟列管致敏香料須被計數():
    tokens = ingredient_tokens("['Water, Limonene, Linalool, Glycerin']")
    flags = flag_ingredients(tokens)
    assert flags["n_fragrance_allergens"] == 2


# ---------------------------------------------------------------- 錯誤 3
# 曾經發生：折扣浪費用全站平均轉換率估算，低估約 3 倍。
# 損益兩平公式是整份商業分析最硬的結論，必須守住邊界行為。

def test_損益兩平門檻隨折扣加深而上升():
    organic, non_buyers = 68_666.0, 838_761.0
    rates = [
        breakeven_recovery_rate(organic, non_buyers, discount=d, margin=0.45)
        for d in (0.03, 0.05, 0.10, 0.20)
    ]
    assert rates == sorted(rates), "折扣越深，打平門檻應越高"
    assert all(r > 0 for r in rates)


def test_折扣超過毛利率時方案不可能成立():
    r = breakeven_recovery_rate(1000.0, 10_000.0, discount=0.50, margin=0.45)
    assert r == np.inf, "折扣高於毛利率時，賣越多虧越多，不應回傳有限門檻"


def test_沒有可挽回對象時門檻為無限大():
    r = breakeven_recovery_rate(1000.0, 0.0, discount=0.10, margin=0.45)
    assert r == np.inf


def test_損益兩平定義_門檻挽回率下淨效益為零():
    organic, non_buyers = 68_666.0, 838_761.0
    disc, margin, aov = 0.10, 0.45, 40.54

    r = breakeven_recovery_rate(organic, non_buyers, discount=disc, margin=margin)
    gain = non_buyers * r * aov * (margin - disc)
    waste = organic * aov * disc

    assert gain == pytest.approx(waste, rel=1e-9), "門檻挽回率下淨效益應恰為零"
