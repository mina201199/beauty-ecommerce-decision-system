"""Uplift 評估：Qini、uplift@k、四象限、決策輸出。

**第 3 週先把這個檔寫完並驗證，再去寫模型。**

理由：uplift 沒有個體層級的 ground truth（同一個人不可能同時被寄信和沒被寄信），
所以模型算錯不會拋例外，只會安靜地給你一個看起來合理的數字。唯一的防線是
先在有已知答案的合成資料上證明評估程式是對的。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative_frames(uplift_score, t, y):
    """依預測 uplift 由高到低排序，回傳各前綴的累積統計。"""
    order = np.argsort(-np.asarray(uplift_score, dtype=float))
    t_s = np.asarray(t, dtype=float)[order]
    y_s = np.asarray(y, dtype=float)[order]

    n_t = np.cumsum(t_s)                 # 前 k 名中的處理組人數
    n_c = np.cumsum(1.0 - t_s)           # 前 k 名中的對照組人數
    y_t = np.cumsum(y_s * t_s)           # 前 k 名中處理組的成效總和
    y_c = np.cumsum(y_s * (1.0 - t_s))   # 前 k 名中對照組的成效總和
    return n_t, n_c, y_t, y_c


def qini_curve(uplift_score, t, y):
    """Qini 曲線。

    Qini(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

    那個 N_t/N_c 比例項是關鍵：不做這個校正而直接把兩組成效相減，
    是最常見的實作錯誤，會讓處理組佔比較高的前段看起來憑空變好。
    """
    n_t, n_c, y_t, y_c = _cumulative_frames(uplift_score, t, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / n_c, 0.0)
    q = y_t - y_c * ratio
    x = np.arange(1, len(q) + 1) / len(q)
    return x, q


def qini_auc(uplift_score, t, y) -> float:
    """Qini 曲線相對於隨機排序基準線的面積，正規化到 [-1, 1] 附近。

    值越大越好；0 代表跟隨機排序一樣。
    """
    x, q = qini_curve(uplift_score, t, y)
    if len(q) == 0 or q[-1] == 0:
        return 0.0
    random_line = q[-1] * x
    return float(np.trapezoid(q - random_line, x) / abs(q[-1]))


def uplift_at_k(uplift_score, t, y, k: float) -> float:
    """前 k 比例名單的實際增量轉換率（處理組轉換率 - 對照組轉換率）。"""
    n = int(round(len(t) * k))
    if n == 0:
        return float("nan")
    n_t, n_c, y_t, y_c = _cumulative_frames(uplift_score, t, y)
    i = n - 1
    if n_t[i] == 0 or n_c[i] == 0:
        return float("nan")
    return float(y_t[i] / n_t[i] - y_c[i] / n_c[i])


def ate(t, y) -> dict:
    """平均處理效果，附 bootstrap 之外的解析標準誤（兩個比例差）。

    同時回傳「天真比較」——這是拿來對照用的：很多行銷報告把
    有回應 vs 沒回應的差異當成活動效果，那個數字通常大得離譜。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yt, yc = y[t == 1], y[t == 0]
    if len(yt) == 0 or len(yc) == 0:
        return {"ate": float("nan")}
    pt, pc = yt.mean(), yc.mean()
    se = np.sqrt(pt * (1 - pt) / len(yt) + pc * (1 - pc) / len(yc))
    return {
        "ate": float(pt - pc),
        "se": float(se),
        "ci_low": float(pt - pc - 1.96 * se),
        "ci_high": float(pt - pc + 1.96 * se),
        "rate_treated": float(pt),
        "rate_control": float(pc),
        "n_treated": int(len(yt)),
        "n_control": int(len(yc)),
    }


def four_quadrants(uplift_score, control_prob=None, eps: float | None = None) -> pd.Series:
    """把每個人分到四象限。

    真正的四象限需要**兩個**維度，不是只有 uplift 的正負：

                     控制組本來就會轉換    控制組本來不會轉換
        uplift > 0        (罕見)              可說服者 persuadables
        uplift ~ 0      鐵粉 sure_things      無效客 lost_causes
        uplift < 0      反效果客 sleeping_dogs

    只看 uplift 分不出「鐵粉」和「無效客」—— 兩者的 uplift 都接近 0，
    但一個本來就會買、另一個怎樣都不會買，行銷處置完全相反。

    control_prob 傳入對照組的預測轉換機率（模型的 predict_control_prob()）。
    沒傳的話會退化成只依 uplift 的三段切分，並在標籤上明講是未分離的。
    """
    u = pd.Series(np.asarray(uplift_score, dtype=float)).reset_index(drop=True)
    if eps is None:
        # 用 uplift 分布的尺度決定「接近 0」的範圍，避免寫死門檻
        eps = float(u.abs().quantile(0.25))

    if control_prob is None:
        labels = pd.Series("neutral_未分離", index=u.index)
        labels[u > eps] = "persuadables"
        labels[u < -eps] = "sleeping_dogs"
        return labels

    c = pd.Series(np.asarray(control_prob, dtype=float)).reset_index(drop=True)
    high_base = c >= c.median()

    labels = pd.Series("lost_causes", index=u.index)
    labels[high_base] = "sure_things"
    labels[u > eps] = "persuadables"
    labels[u < -eps] = "sleeping_dogs"
    return labels


def decision_table(
    uplift_score,
    t,
    y,
    contact_cost: float,
    margin: float,
    depths=(0.10, 0.20, 0.30, 0.50, 1.00),
) -> pd.DataFrame:
    """不同投放深度下的商業結果。這張表就是要放進 README 的東西。"""
    n = len(t)
    rows = []
    for d in depths:
        u = uplift_at_k(uplift_score, t, y, d)
        contacted = int(round(n * d))
        inc_conv = u * contacted if not np.isnan(u) else float("nan")
        profit = inc_conv * margin - contacted * contact_cost
        rows.append(
            {
                "投放深度": f"{d:.0%}",
                "接觸人數": contacted,
                "增量轉換率": round(u, 5) if not np.isnan(u) else None,
                "增量轉換數": round(inc_conv, 1) if not np.isnan(inc_conv) else None,
                "增量利潤": round(profit, 1) if not np.isnan(profit) else None,
            }
        )
    return pd.DataFrame(rows)


def summary(uplift_score, t, y, ks=(0.10, 0.20, 0.30, 0.50)) -> dict:
    out = {"qini_auc": qini_auc(uplift_score, t, y)}
    for k in ks:
        out[f"uplift@{k:.0%}"] = uplift_at_k(uplift_score, t, y, k)
    out.update({f"ate_{k}": v for k, v in ate(t, y).items()})
    return out


# --- 自我檢查 -----------------------------------------------------------
def self_test(seed: int = 42) -> None:
    """在已知答案的合成資料上驗證評估程式本身。

    造一組資料：一半的人有真實 +0.20 的效果，另一半是 0。
    完美的 uplift 分數應該拿到明顯為正的 Qini AUC；隨機分數應該接近 0。
    第 3 週跑這個，通過了才往下走。
    """
    rng = np.random.default_rng(seed)
    n = 40_000
    responsive = rng.random(n) < 0.5
    t = rng.integers(0, 2, n)
    base = 0.10
    p = base + responsive * t * 0.20
    y = (rng.random(n) < p).astype(int)

    perfect = responsive.astype(float)
    random_score = rng.random(n)

    q_perfect = qini_auc(perfect, t, y)
    q_random = qini_auc(random_score, t, y)
    a = ate(t, y)

    print("=" * 62)
    print("評估程式自我檢查（合成資料，真實效果已知）")
    print("=" * 62)
    print(f"真實 ATE 應約為 0.10（一半的人有 +0.20 效果）")
    print(f"  估計 ATE          : {a['ate']:.4f}  [{a['ci_low']:.4f}, {a['ci_high']:.4f}]")
    print(f"  完美排序 Qini AUC : {q_perfect:.4f}  (應明顯 > 0)")
    print(f"  隨機排序 Qini AUC : {q_random:.4f}  (應接近 0)")
    print(f"  uplift@30% 完美   : {uplift_at_k(perfect, t, y, 0.30):.4f}  (應接近 0.20)")
    print(f"  uplift@30% 隨機   : {uplift_at_k(random_score, t, y, 0.30):.4f}  (應接近 0.10)")
    ok = q_perfect > 0.05 and abs(q_random) < 0.03 and abs(a["ate"] - 0.10) < 0.02
    print(f"\n結果: {'PASS' if ok else 'FAIL — 評估程式有問題，先修這裡再寫模型'}")
    print("=" * 62)


if __name__ == "__main__":
    self_test()
