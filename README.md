# 增量投放決策引擎 (Incremental Targeting Engine)

> 行銷團隊衡量「誰有回應」。這個工具衡量「誰是**因為**這檔活動才回應」——並告訴你名單裡哪 30% 值得聯繫。

---

## 這是什麼

一個因果推論驅動的行銷投放決策工具。上傳一份活動名單（含 treatment/control 標記與成效欄位），它會：

1. **驗證實驗有效性** — 共變數平衡檢定 (SMD)。如果隨機分派壞掉，它會拒絕給出因果宣稱。
2. **估計平均效果 (ATE)** — 附信賴區間，並同時顯示「天真的有回應 vs 沒回應比較」會錯報成多少。兩個數字的差距就是整個工具的價值主張。
3. **估計異質效果 (CATE)** — T-learner / S-learner，Qini 曲線與 uplift@k。
4. **分成四象限** — 可說服者 (Persuadables)、鐵粉 (Sure Things)、無效客 (Lost Causes)、**反效果客 (Sleeping Dogs)**。反效果客是重點：相關性模型永遠找不到他們，但每個行銷人都秒懂為什麼這很重要。
5. **輸出決策** — 排序後的名單 CSV：聯繫這 N 位、預期增量轉換數、預期增量利潤、相對於全量發送省下的預算。

---

## 為什麼是這個題目（給我自己看的）

| 條件 | 這個專案怎麼滿足 |
|---|---|
| 經濟系背景 | 因果推論是經濟系的本科訓練。內生性、選擇偏誤、DiD/IV — bootcamp 畢業生幾乎沒有人會。這是唯一一個學歷是**資產**而不是要解釋的缺點的題目。 |
| 電商行銷企劃 3-5 年 | 這個工具說的是行銷人的語言。定位成「會量化增量的行銷企劃」＝**年資延續**，不是轉職歸零。 |
| 測試業 PM | 實驗設計、樣本數、驗收標準 — 直接對應 A/B test 的 MDE 與 guardrail 設計。 |
| 時間有限 | 主資料集只有 2.68 MB。秒級迭代，不會被資料工程吃掉整個學期。 |
| 未飽和 | 主資料集 9 個 notebook、合成驗證集 0 個、Criteo 13 個。 |

⚠️ **定位警告**：這個專案要包裝成「行銷企劃 → 行銷數據/CRM 分析」的**升級**，不是「轉職進資料領域」。以初階數據分析師應徵，起薪反而低於現有年資行情。這是本專案最重要的一句話。

---

## 資料集

| 階段 | 資料集 | 大小 | Notebooks | 授權 | 用途 |
|---|---|---|---|---|---|
| 0 驗證 | [Marketing A/B Test - Synthetic (Known Uplift)](https://www.kaggle.com/datasets/krishnaharish1/marketing-ab-test-synthetic-uplift) | ~300 KB | **0** | CC0 | 有 ground_truth.json，先證明估計器算對了 |
| 1 主力 | [Marketing Promotion Campaign Uplift Modelling (Hillstrom)](https://www.kaggle.com/datasets/davinwijaya/customer-retention) | 2.68 MB | 9 | CC0 | 真實 RCT，有明確 treatment/control |
| 2 規模 | [Uplift Modeling, Marketing Campaign Data (Criteo v2.1)](https://www.kaggle.com/datasets/arashnic/uplift-modeling) | ~324 MB | 13 | CC0 | 1,300 萬列，證明能處理規模 |
| 3 真實 CRM | [E-commerce multichannel direct messaging (REES46)](https://www.kaggle.com/datasets/mkechinov/direct-messaging) | ~345 MB | **7** | CDLA-Permissive-1.0 | 四渠道 × 三活動類型的真實 CRM outcome log |

> 階段 0 先跑。用已知答案的合成資料驗證估計器，是這個專案跟其他 uplift 專案最大的差別——大部分人直接上真資料，然後永遠不知道自己算錯了。

---

## 快速開始

```powershell
.\run.ps1 setup
```

```powershell
.\run.ps1 demo
```

完整指令見 [PLAN.md](PLAN.md)。

---

## 結果

<!-- 做完後把這張表填滿。這是整個 README 最重要的區塊，放在最上面。 -->

| 方法 | Qini AUC | Uplift@30% | 增量轉換 | 對照：全量發送 |
|---|---|---|---|---|
| 隨機排序 (baseline) | — | — | — | — |
| 傾向分數（**錯誤**做法，示範用） | — | — | — | — |
| T-learner (Logistic) | — | — | — | — |
| T-learner (GBDT) | — | — | — | — |
| S-learner | — | — | — | — |

**Headline**：在前 30% 名單投放，取得全量發送 __% 的增量轉換，節省 __% 預算。

---

## 專案結構

```
ML_專題/
├── README.md              # 你正在看的
├── PLAN.md                # 12 週計畫 + 求職定位策略
├── requirements.txt
├── .env.example           # 複製成 .env（已被 gitignore）
├── run.ps1                # Windows 進入點
├── app.py                 # Gradio 展示介面
├── scripts/
│   └── download_data.py   # Kaggle API 下載
├── src/
│   ├── config.py          # 資料集註冊表與欄位對應
│   ├── data.py            # 載入 + schema 檢查
│   ├── balance.py         # 共變數平衡檢定（經濟系的差異化武器）
│   ├── uplift.py          # T-learner / S-learner
│   └── evaluate.py        # Qini、uplift@k、四象限、決策輸出
├── sql/                   # 漏斗與世代分析（履歷關鍵字：SQL）
├── notebooks/             # 只放探索，不是交付物
└── data/                  # gitignored
```

---

## 授權與致謝

各資料集授權見上表。REES46 資料需標註來源。本專案程式碼供學習與作品集用途。
