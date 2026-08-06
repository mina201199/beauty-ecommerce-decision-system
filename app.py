"""Gradio 展示介面 —— 沿用你 RAG 專案 app.py 的結構。

上傳活動名單 CSV -> 選 treatment / outcome 欄 -> 平衡檢定 -> uplift 模型 -> 決策名單。

提醒：這裡刻意保持樸素。你喜歡美的事物，這是整個專案最好玩的部分，
也是最容易吃掉兩週的部分。介面美化上限一個下午。
"""
from __future__ import annotations

import io

import gradio as gr
import pandas as pd

from src.balance import report as balance_report
from src.config import CONTACT_COST, MARGIN_PER_CONVERSION, OUT_DIR, SEED, TEST_SIZE
from src.evaluate import decision_table, four_quadrants, qini_auc, uplift_at_k
from src.uplift import all_models

_STATE: dict = {}


def load_csv(file):
    if file is None:
        return gr.update(choices=[]), gr.update(choices=[]), "請先上傳 CSV"
    df = pd.read_csv(file.name)
    _STATE["df"] = df
    cols = list(df.columns)
    msg = f"已載入 {len(df):,} 列 x {len(cols)} 欄\n\n欄位：{', '.join(cols)}"
    return gr.update(choices=cols), gr.update(choices=cols), msg


def check_balance(treat_col, out_col):
    df = _STATE.get("df")
    if df is None:
        return "請先上傳 CSV"
    if not treat_col or not out_col:
        return "請選擇 treatment 與 outcome 欄位"

    t = pd.to_numeric(df[treat_col], errors="coerce").fillna(0)
    t = (t > 0).astype(int)
    X = df.drop(columns=[treat_col, out_col], errors="ignore")
    X = pd.get_dummies(X, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    buf = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(buf):
        res = balance_report(X, t, seed=SEED, verbose=True)

    _STATE.update({"X": X, "t": t, "balance": res})
    head = f"### 判定：{res['verdict']}\n\n"
    if res["verdict"] == "FAIL":
        head += (
            "隨機分派看起來壞掉了。此時**不應**直接把 ATE 當成因果效果 —— "
            "這正是這個工具存在的理由：它會擋下來，而不是照算給你一個好看的數字。\n\n"
        )
    return head + "```\n" + buf.getvalue() + "\n```"


def run_uplift(out_col, contact_cost, margin):
    df = _STATE.get("df")
    X, t = _STATE.get("X"), _STATE.get("t")
    if df is None or X is None:
        return None, None, "請先跑平衡檢定"

    y = pd.to_numeric(df[out_col], errors="coerce").fillna(0)
    y = (y > 0).astype(int)

    from src.data import split

    X_tr, X_te, t_tr, t_te, y_tr, y_te = split(X, t, y, TEST_SIZE, SEED)

    rows, best, best_score = [], None, -1e9
    for m in all_models(SEED):
        m.fit(X_tr, t_tr, y_tr)
        u = m.predict_uplift(X_te)
        q = qini_auc(u, t_te, y_te)
        rows.append(
            {
                "方法": m.name,
                "Qini AUC": round(q, 4),
                "uplift@10%": round(uplift_at_k(u, t_te, y_te, 0.10), 4),
                "uplift@30%": round(uplift_at_k(u, t_te, y_te, 0.30), 4),
            }
        )
        if q > best_score and not m.name.startswith(("random", "propensity")):
            best_score, best = q, (m.name, u, t_te, y_te, m, X_te)

    comparison = pd.DataFrame(rows).sort_values("Qini AUC", ascending=False)
    if best is None:
        return comparison, None, "沒有可用模型"

    name, u, t_te, y_te, model, X_eval = best
    decisions = decision_table(u, t_te, y_te, contact_cost, margin)

    control_prob = (
        model.predict_control_prob(X_eval)
        if hasattr(model, "predict_control_prob")
        else None
    )
    segments = four_quadrants(u, control_prob)
    quad = segments.value_counts().to_dict()

    out = OUT_DIR / "send_list.csv"
    pd.DataFrame({"uplift_score": u, "segment": segments.values}).sort_values(
        "uplift_score", ascending=False
    ).to_csv(out, index=False, encoding="utf-8-sig")

    note = (
        f"**最佳模型：{name}**（Qini AUC {best_score:.4f}）\n\n"
        f"四象限：{quad}\n\n"
        f"名單已匯出到 `{out}`\n\n"
        "> sleeping_dogs 是重點：這些人被投放後反而更不會轉換。"
        "相關性模型永遠找不到他們。"
    )
    return comparison, decisions, note


with gr.Blocks(title="增量投放決策引擎") as demo:
    gr.Markdown(
        "# 增量投放決策引擎\n"
        "行銷團隊衡量「誰有回應」。這個工具衡量「誰是**因為**這檔活動才回應」。"
    )

    with gr.Row():
        f = gr.File(label="上傳活動名單 CSV", file_types=[".csv"])
    info = gr.Markdown()

    with gr.Row():
        treat = gr.Dropdown(label="Treatment 欄（有無投放）", choices=[])
        outcome = gr.Dropdown(label="Outcome 欄（轉換與否）", choices=[])

    f.change(load_csv, inputs=f, outputs=[treat, outcome, info])

    gr.Markdown("## 步驟 1：驗證實驗有效性")
    btn_bal = gr.Button("跑共變數平衡檢定", variant="secondary")
    bal_out = gr.Markdown()
    btn_bal.click(check_balance, inputs=[treat, outcome], outputs=bal_out)

    gr.Markdown("## 步驟 2：估計增量效果並產出名單")
    with gr.Row():
        cost = gr.Number(label="每次接觸成本", value=CONTACT_COST)
        marg = gr.Number(label="每次轉換毛利", value=MARGIN_PER_CONVERSION)
    btn_run = gr.Button("估計 uplift", variant="primary")
    tbl_cmp = gr.Dataframe(label="模型比較")
    tbl_dec = gr.Dataframe(label="不同投放深度的商業結果")
    run_note = gr.Markdown()
    btn_run.click(run_uplift, inputs=[outcome, cost, marg], outputs=[tbl_cmp, tbl_dec, run_note])


if __name__ == "__main__":
    demo.launch()
