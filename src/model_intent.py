"""模組 2：Session 購買意圖預測。

用法：
    python -m src.model_intent

預測問題：
    「一個 session 已經瀏覽了 3 分鐘、目前還沒下單，他最終會不會買？」

【為什麼不用 scale_pos_weight / SMOTE】
正樣本佔 13.5%，這不算嚴重不平衡（嚴重是指 1% 以下）。加權會扭曲預測
機率的校準度，而模組 5 的營收試算需要能反映真實機率的分數。
既然 LightGBM 在這個比例下本來就處理得很好，就不要為了「看起來有做
不平衡處理」而破壞校準。

【本模組最重要的產出不是 AUC，是 lift curve】
模組 5 需要知道「鎖定前 20% 的 session 能涵蓋多少購買者」。
那個數字直接決定折扣方案的成本效益，比 AUC 有用得多。
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    FIGURE_DIR,
    LGBM_PARAMS,
    MODEL_DIR,
    OBSERVATION_WINDOW_MINUTES,
    TABLE_DIR,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VALID_MONTHS,
)
from src.features import FEATURE_DIR

DROP_COLS = ["user_session", "label", "month"]


def load(months: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.concat(
        [pd.read_parquet(FEATURE_DIR / f"{m}.parquet") for m in months],
        ignore_index=True,
    )
    y = df["label"]
    X = df.drop(columns=DROP_COLS)
    return X, y


def decile_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """十分位提升表 —— 營運端唯一看得懂的模型績效表。

    把 session 依預測分數由高到低排序切成 10 等分，看每一等分裡
    實際的購買者比例。第 1 等分的購買率應該遠高於整體平均。
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    total_pos = y_sorted.sum()

    rows = []
    cum = 0
    for i in range(10):
        lo, hi = i * n // 10, (i + 1) * n // 10
        chunk = y_sorted[lo:hi]
        pos = int(chunk.sum())
        cum += pos
        rows.append({
            "十分位": i + 1,
            "session 數": len(chunk),
            "購買者": pos,
            "該組購買率%": pos / len(chunk) * 100,
            "提升倍數": (pos / len(chunk)) / (total_pos / n),
            "累積涵蓋購買者%": cum / total_pos * 100,
        })
    return pd.DataFrame(rows)


def top_k_capture(y_true: np.ndarray, y_score: np.ndarray, ks=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()
    n = len(y_sorted)

    rows = []
    for k in ks:
        cut = int(n * k)
        pos = int(y_sorted[:cut].sum())
        rows.append({
            "鎖定比例": k,
            "session 數": cut,
            "涵蓋購買者": pos,
            "涵蓋率%": pos / total_pos * 100,
            "該群購買率%": pos / cut * 100,
            "提升倍數": (pos / cut) / (total_pos / n),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[*] 觀察窗 {OBSERVATION_WINDOW_MINUTES} 分鐘")
    print(f"[*] 訓練 {TRAIN_MONTHS} / 驗證 {VALID_MONTHS} / 測試 {TEST_MONTHS}")
    print("    時間切分，不用隨機切分 —— 隨機切等於用未來預測過去。\n")

    X_tr, y_tr = load(TRAIN_MONTHS)
    X_va, y_va = load(VALID_MONTHS)
    X_te, y_te = load(TEST_MONTHS)

    print(f"  訓練 {len(X_tr):>9,}  正樣本 {y_tr.mean():.2%}")
    print(f"  驗證 {len(X_va):>9,}  正樣本 {y_va.mean():.2%}")
    print(f"  測試 {len(X_te):>9,}  正樣本 {y_te.mean():.2%}")
    print(f"  特徵 {X_tr.shape[1]} 個\n")

    params = {k: v for k, v in LGBM_PARAMS.items() if k != "metric"}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"[*] 早停於第 {model.best_iteration_} 棵樹\n")

    # ---- 評估 ----
    results = {}
    print(f"{'=' * 58}")
    print("  模型績效")
    print(f"{'=' * 58}")
    print(f"  {'資料集':<8}{'baseline':>12}{'PR-AUC':>12}{'ROC-AUC':>12}{'提升':>10}")

    for name, X, y in (("驗證(1月)", X_va, y_va), ("測試(2月)", X_te, y_te)):
        p = model.predict_proba(X)[:, 1]
        base = y.mean()
        pr = average_precision_score(y, p)
        roc = roc_auc_score(y, p)
        results[name] = {"baseline": base, "pr_auc": pr, "roc_auc": roc}
        print(f"  {name:<8}{base:>11.2%}{pr:>12.4f}{roc:>12.4f}{pr / base:>9.2f}x")

    p_te = model.predict_proba(X_te)[:, 1]
    roc_te = results["測試(2月)"]["roc_auc"]

    # ---- 洩漏檢查 ----
    print(f"\n{'=' * 58}")
    print("  Data Leakage 檢查")
    print(f"{'=' * 58}")
    if roc_te > 0.95:
        print(f"  [X] ROC-AUC {roc_te:.4f} 過高，極可能有洩漏。停下來檢查特徵。")
    elif roc_te > 0.90:
        print(f"  [!] ROC-AUC {roc_te:.4f} 偏高，值得再確認一次特徵定義。")
    else:
        print(f"  [OK] ROC-AUC {roc_te:.4f} 落在合理範圍，未見洩漏跡象。")

    # ---- 十分位提升表 ----
    dec = decile_table(y_te.to_numpy(), p_te)
    dec.to_csv(TABLE_DIR / "model_decile_lift.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 58}")
    print("  十分位提升表（測試集 2 月）")
    print(f"{'=' * 58}")
    print(dec.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- Top-K 涵蓋率（模組 5 要用） ----
    topk = top_k_capture(y_te.to_numpy(), p_te)
    topk.to_csv(TABLE_DIR / "model_topk_capture.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 58}")
    print("  Top-K 涵蓋率  <-- 模組 5 的 CAPTURE_RATE 由此而來")
    print(f"{'=' * 58}")
    print(topk.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---- 特徵重要度 ----
    # 用 gain 不用預設的 split。split 只數「這個特徵被拿來分裂幾次」，
    # 會嚴重偏袒連續型高基數特徵（價格、時間間隔天生就有很多切點）。
    # gain 衡量的是「每次分裂實際降低多少損失」，才是真的重要度。
    imp = pd.DataFrame({
        "特徵": X_tr.columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp["gain佔比%"] = imp["gain"] / imp["gain"].sum() * 100
    imp.to_csv(TABLE_DIR / "model_feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 58}")
    print("  特徵重要度 Top 12（依 gain）")
    print(f"{'=' * 58}")
    print(imp.head(12).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ---- 存檔 ----
    # 不能用 booster_.save_model()：LightGBM 的 C 函式庫在 Windows 上
    # 無法處理含非 ASCII 字元的路徑，而專案資料夾叫「ML_專題」。
    # 改成先轉字串再用 Python 寫檔，Python 的檔案 API 沒有這個限制。
    (MODEL_DIR / "intent_lgbm.txt").write_text(
        model.booster_.model_to_string(), encoding="utf-8"
    )
    capture_20 = float(topk.loc[topk["鎖定比例"] == 0.20, "涵蓋率%"].iloc[0]) / 100
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps({
            "observation_window_min": OBSERVATION_WINDOW_MINUTES,
            "best_iteration": int(model.best_iteration_),
            "results": results,
            "capture_rate_top20": capture_20,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[OK] 模型已存 → {MODEL_DIR / 'intent_lgbm.txt'}")
    print(f"     前 20% 涵蓋率 {capture_20:.2%}  <-- 拿去更新 src/impact.py 的 CAPTURE_RATE")


if __name__ == "__main__":
    main()
