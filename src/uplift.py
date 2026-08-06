"""Uplift 估計器：baseline、T-learner、S-learner。

刻意都用 sklearn 現成模型組成 —— 這個專案的價值在推論嚴謹度，
不在模型複雜度。T-learner 的核心就是兩條你已經手刻過的邏輯迴歸。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def _make_base(kind: str, seed: int):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        )
    if kind == "gbdt":
        return GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
    raise ValueError(f"未知的 base learner: {kind}")


@dataclass
class RandomBaseline:
    """隨機排序。所有模型都必須打敗它，否則就是沒學到東西。"""

    seed: int = SEED
    name: str = "random"

    def fit(self, X, t, y):
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(X))


@dataclass
class PropensityBaseline:
    """只預測「誰會轉換」，完全忽略因果 —— 也就是絕大多數行銷模型在做的事。

    刻意放進來當對照組。它通常會挑出 sure things（本來就會買的人），
    在 Qini 上表現平庸甚至為負。把它放進結果表，就等於用數據講清楚
    「回應模型」和「增量模型」的差別 —— 這是整份報告最有說服力的一段。
    """

    base: str = "logistic"
    seed: int = SEED
    name: str = "propensity (錯誤示範)"

    def fit(self, X, t, y):
        self.model_ = _make_base(self.base, self.seed).fit(X, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


@dataclass
class TLearner:
    """兩個獨立模型：處理組一個、對照組一個，相減。

    優點是直觀、好除錯。缺點是對照組樣本少時方差大。
    """

    base: str = "logistic"
    seed: int = SEED

    @property
    def name(self) -> str:
        return f"T-learner ({self.base})"

    def fit(self, X, t, y):
        t = np.asarray(t)
        self.m_t_ = _make_base(self.base, self.seed).fit(X[t == 1], np.asarray(y)[t == 1])
        self.m_c_ = _make_base(self.base, self.seed).fit(X[t == 0], np.asarray(y)[t == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.m_t_.predict_proba(X)[:, 1] - self.m_c_.predict_proba(X)[:, 1]

    def predict_control_prob(self, X) -> np.ndarray:
        """對照組（沒被投放）的預測轉換機率。四象限分群需要這個維度。"""
        return self.m_c_.predict_proba(X)[:, 1]


@dataclass
class SLearner:
    """單一模型，把 treatment 當成一個特徵，再用 t=1 與 t=0 兩次預測相減。

    樣本利用率較好，但若模型把 treatment 特徵當成不重要，
    預測出來的 uplift 會被壓成接近 0 —— 這是它的典型失敗模式，要注意。
    """

    base: str = "gbdt"
    seed: int = SEED

    @property
    def name(self) -> str:
        return f"S-learner ({self.base})"

    def fit(self, X, t, y):
        Xt = X.copy()
        Xt["__treatment__"] = np.asarray(t)
        self.model_ = _make_base(self.base, self.seed).fit(Xt, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X1, X0 = X.copy(), X.copy()
        X1["__treatment__"] = 1
        X0["__treatment__"] = 0
        return self.model_.predict_proba(X1)[:, 1] - self.model_.predict_proba(X0)[:, 1]

    def predict_control_prob(self, X) -> np.ndarray:
        X0 = X.copy()
        X0["__treatment__"] = 0
        return self.model_.predict_proba(X0)[:, 1]


def all_models(seed: int = SEED) -> list:
    """結果表的完整階梯：從最笨的到最好的，缺一不可。"""
    return [
        RandomBaseline(seed=seed),
        PropensityBaseline(seed=seed),
        TLearner(base="logistic", seed=seed),
        TLearner(base="gbdt", seed=seed),
        SLearner(base="gbdt", seed=seed),
    ]


def run_comparison(dataset: str = "hillstrom", seed: int = SEED) -> pd.DataFrame:
    """跑完整比較，回傳可以直接貼進 README 的表。"""
    from src.balance import report as balance_report
    from src.config import TEST_SIZE
    from src.data import load, split
    from src.evaluate import qini_auc, uplift_at_k

    X, t, y = load(dataset)
    balance_report(X, t, seed=seed)

    X_tr, X_te, t_tr, t_te, y_tr, y_te = split(X, t, y, TEST_SIZE, seed)
    print(f"訓練 {len(X_tr):,} / 測試 {len(X_te):,}\n")

    rows = []
    for m in all_models(seed):
        m.fit(X_tr, t_tr, y_tr)
        u = m.predict_uplift(X_te)
        rows.append(
            {
                "方法": m.name,
                "Qini AUC": round(qini_auc(u, t_te, y_te), 4),
                "uplift@10%": round(uplift_at_k(u, t_te, y_te, 0.10), 4),
                "uplift@30%": round(uplift_at_k(u, t_te, y_te, 0.30), 4),
                "uplift@50%": round(uplift_at_k(u, t_te, y_te, 0.50), 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("Qini AUC", ascending=False)
    print(df.to_string(index=False))
    print("\n判讀提示：任何模型的 Qini AUC 若沒明顯高於 random，就是沒學到增量訊號。")
    print("propensity 那一列若排名不差，通常代表資料裡的效果很同質，值得在報告中討論。")
    return df


if __name__ == "__main__":
    import argparse

    from src.config import DATASETS, OUT_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hillstrom", choices=list(DATASETS))
    a = p.parse_args()
    result = run_comparison(a.dataset)
    out = OUT_DIR / f"comparison_{a.dataset}.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已存到 {out}")
