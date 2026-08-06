-- 活動成效分析 —— 用 DuckDB 直接查 CSV，不需要架資料庫
--
-- 為什麼要有這個檔：台灣資料職缺的實際篩選堆疊是
-- SQL + BI 工具 + GA4 + Excel。一個只有 Python notebook 的作品集
-- 在關鍵字比對上會全部落空。這是投報率最高的補強，不要跳過。
--
-- 跑法：
--   pip install duckdb
--   duckdb -c ".read sql/campaign_analysis.sql"
-- 或在 Python 裡：duckdb.sql(open('sql/campaign_analysis.sql').read())

-- 依實際欄位調整。先跑 python -m src.data --describe 確認 schema。
CREATE OR REPLACE VIEW campaign AS
SELECT * FROM read_csv_auto('data/raw/hillstrom/*.csv');


-- 1. 分組成效總表：這是任何因果分析的第一張表
--    treated / control 的樣本數與轉換率，附標準誤
SELECT
    segment,
    COUNT(*)                                        AS n,
    ROUND(AVG(visit), 5)                            AS visit_rate,
    ROUND(SQRT(AVG(visit) * (1 - AVG(visit)) / COUNT(*)), 5) AS se,
    ROUND(AVG(spend), 3)                            AS avg_spend
FROM campaign
GROUP BY segment
ORDER BY n DESC;


-- 2. 增量效果 vs 天真比較
--    這兩個數字的差距就是整個專案的價值主張。
--    很多行銷報告拿「有回應 vs 沒回應」當活動效果，那個數字通常大得離譜。
WITH arms AS (
    SELECT
        AVG(CASE WHEN segment <> 'No E-Mail' THEN visit END) AS rate_treated,
        AVG(CASE WHEN segment  = 'No E-Mail' THEN visit END) AS rate_control
    FROM campaign
)
SELECT
    ROUND(rate_treated, 5)                    AS 處理組轉換率,
    ROUND(rate_control, 5)                    AS 對照組轉換率,
    ROUND(rate_treated - rate_control, 5)     AS 增量效果_ATE,
    ROUND((rate_treated - rate_control) / NULLIF(rate_control, 0) * 100, 1) AS 相對提升百分比
FROM arms;


-- 3. 分群異質性：哪一群的增量效果最大？
--    這是 uplift 模型要自動找出來的東西，先用 SQL 手動看一遍，
--    模型跑出來的結果才有東西可以對照 —— 模型說的跟這裡差太多就是有 bug。
SELECT
    history_segment,
    COUNT(*)                                                              AS n,
    ROUND(AVG(CASE WHEN segment <> 'No E-Mail' THEN visit END), 5)        AS 處理組,
    ROUND(AVG(CASE WHEN segment  = 'No E-Mail' THEN visit END), 5)        AS 對照組,
    ROUND(AVG(CASE WHEN segment <> 'No E-Mail' THEN visit END)
        - AVG(CASE WHEN segment  = 'No E-Mail' THEN visit END), 5)        AS 增量效果
FROM campaign
GROUP BY history_segment
HAVING COUNT(*) > 500          -- 樣本太小的分群不要下結論
ORDER BY 增量效果 DESC;


-- 4. 近期性 x 新客的交叉分析
--    找可說服者常見的模式：新客 + 近期有互動
SELECT
    newbie,
    CASE
        WHEN recency <= 3  THEN '1_近期(<=3月)'
        WHEN recency <= 6  THEN '2_中期(4-6月)'
        WHEN recency <= 9  THEN '3_較久(7-9月)'
        ELSE                    '4_久未互動(10月+)'
    END AS recency_band,
    COUNT(*) AS n,
    ROUND(AVG(CASE WHEN segment <> 'No E-Mail' THEN visit END)
        - AVG(CASE WHEN segment  = 'No E-Mail' THEN visit END), 5) AS 增量效果
FROM campaign
GROUP BY newbie, recency_band
HAVING COUNT(*) > 500
ORDER BY 增量效果 DESC;


-- 5. 平衡檢查的 SQL 版本
--    src/balance.py 會做得更完整（含 SMD 與聯合檢定），
--    但能用 SQL 講清楚同一件事，在面試裡是加分的。
SELECT
    CASE WHEN segment = 'No E-Mail' THEN 'control' ELSE 'treated' END AS arm,
    COUNT(*)                  AS n,
    ROUND(AVG(recency), 4)    AS avg_recency,
    ROUND(AVG(history), 2)    AS avg_history,
    ROUND(AVG(newbie), 4)     AS pct_newbie,
    ROUND(AVG(mens), 4)       AS pct_mens
FROM campaign
GROUP BY arm;
-- 兩列的每個平均值都應該很接近。差很多就代表隨機分派有問題，
-- 此時 ATE 不能直接解讀為因果效果。
