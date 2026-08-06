"""共變數平衡檢定 —— 這是整個專案的差異化武器。

在估計任何因果效果之前，要先問：這個實驗的隨機分派到底有沒有壞掉？
Bootcamp 出來的作品集不會有這一步；經濟系的訓練會。

兩個檢定：
1. 逐欄標準化平均差 (SMD)。慣例：|SMD| > 0.1 視為不平衡。
2. 聯合檢定：用共變數預測 treatment。如果隨機分派成功，
   應該預測不出來，AUC 會接近 0.5。AUC 明顯高於 0.5 代表分派與
   共變數有關，不能直接宣稱因果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 常見的 post-treatment 欄位名稱片段。這只是提醒，不是判準 ——
# 最終要靠你對資料產生過程的理解。
_POST_TREATMENT_HINTS = (
    "spend", "revenue", "conversion", "purchase", "open", "click",
    "response", "visit", "redeem", "order", "churn", "return",
)


def standardized_mean_diff(X: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """逐欄 SMD。連續與二元變數都適用同一個公式。"""
    treated, control = X[t == 1], X[t == 0]
    mt, mc = treated.mean(), control.mean()
    vt, vc = treated.var(ddof=1), control.var(ddof=1)
    pooled = np.sqrt((vt + vc) / 2.0)
    smd = (mt - mc) / pooled.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "mean_treated": mt,
            "mean_control": mc,
            "smd": smd,
            "abs_smd": smd.abs(),
        }
    )
    out["balanced"] = out["abs_smd"] < 0.10
    return out.sort_values("abs_smd", ascending=False)


def joint_balance_auc(X: pd.DataFrame, t: pd.Series, seed: int = 42) -> float:
    """用共變數預測 treatment 的 cross-val AUC。隨機分派成功時應接近 0.5。"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    if t.nunique() < 2:
        return float("nan")
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    scores = cross_val_score(clf, X, t, cv=3, scoring="roc_auc")
    return float(scores.mean())


def flag_post_treatment(columns) -> list[str]:
    """依名稱標記可疑的 post-treatment 欄位。是提醒，不是判準。"""
    return [c for c in columns if any(h in str(c).lower() for h in _POST_TREATMENT_HINTS)]


def report(X: pd.DataFrame, t: pd.Series, seed: int = 42, verbose: bool = True) -> dict:
    """跑完整平衡檢查，回傳結果並印出人看得懂的判讀。"""
    smd = standardized_mean_diff(X, t)
    auc = joint_balance_auc(X, t, seed)
    suspicious = flag_post_treatment(X.columns)
    n_unbalanced = int((~smd["balanced"]).sum())

    verdict = "PASS"
    reasons = []
    if n_unbalanced > 0:
        verdict = "WARN"
        reasons.append(f"{n_unbalanced} 個共變數 |SMD| >= 0.10")
    if not np.isnan(auc) and auc > 0.55:
        verdict = "FAIL"
        reasons.append(f"聯合檢定 AUC = {auc:.3f}（明顯可預測 treatment）")
    if suspicious:
        reasons.append(f"疑似 post-treatment 欄位：{suspicious}")

    if verbose:
        print("\n" + "=" * 62)
        print("共變數平衡檢定")
        print("=" * 62)
        print(smd.round(4).to_string())
        print(f"\n聯合檢定 AUC: {auc:.4f}  (0.5 = 完全平衡)")
        print(f"不平衡欄位數: {n_unbalanced} / {len(smd)}")
        if suspicious:
            print(f"\n[!] 疑似 post-treatment 欄位（處理後才發生，不可當共變數）:")
            for c in suspicious:
                print(f"      - {c}")
            print("    這是依名稱的提醒，請自己確認資料產生順序。")
        print(f"\n判定: {verdict}")
        for r in reasons:
            print(f"  - {r}")
        if verdict == "FAIL":
            print("\n  隨機分派看起來壞掉了。此時 ATE 不能直接解讀為因果效果，")
            print("  需要改用傾向分數加權或雙重穩健估計，並在報告中明講。")
        print("=" * 62 + "\n")

    return {
        "smd": smd,
        "joint_auc": auc,
        "n_unbalanced": n_unbalanced,
        "suspicious_columns": suspicious,
        "verdict": verdict,
        "reasons": reasons,
    }


if __name__ == "__main__":
    import argparse

    from src.config import DATASETS, SEED
    from src.data import load

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hillstrom", choices=list(DATASETS))
    a = p.parse_args()
    X, t, y = load(a.dataset)
    report(X, t, seed=SEED)
