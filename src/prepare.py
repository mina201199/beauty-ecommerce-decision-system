"""資料清洗：把 5 個月的原始 CSV 轉成 parquet，並輸出資料品質報告。

用法：
    python -m src.prepare

為什麼要轉 parquet：
  - CSV 每次讀都要重新解析文字，parquet 是二進位欄式儲存，讀取快 10 倍以上
  - 檔案大小約剩 1/5
  - 型別會被保存，不用每次重新指定 dtype
這一步做完，後面每個模組的開發迭代都會快很多。
"""

import sys

import pandas as pd

from src.config import CSV_DTYPES, EVENT_ORDER, INTERIM_DIR, MONTHS, RAW_DIR


def load_month(month: str) -> pd.DataFrame:
    """讀取單月 CSV，套用型別並解析時間。"""
    csv_path = RAW_DIR / f"{month}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}\n請先執行：python -m src.download"
        )

    df = pd.read_csv(csv_path, dtype=CSV_DTYPES)

    # event_time 原始格式是 "2019-10-01 00:00:00 UTC"。
    # 直接丟給 to_datetime 讓它自動推斷會很慢（2000 萬列會跑好幾分鐘），
    # 所以先切掉尾巴的 " UTC" 再用固定格式解析，快非常多。
    df["event_time"] = pd.to_datetime(
        df["event_time"].str.slice(0, 19),
        format="%Y-%m-%d %H:%M:%S",
    )
    return df


def quality_report(df: pd.DataFrame, month: str) -> dict:
    """輸出資料品質摘要。

    這份報告不只是給你自己看的 —— README 裡放一段「資料品質檢核」
    會讓整個專案看起來專業很多。多數人的專題會直接跳過這步。
    """
    n = len(df)
    mem_gb = df.memory_usage(deep=True).sum() / 1024 ** 3

    counts = df["event_type"].value_counts()
    n_dup = int(df.duplicated().sum())
    n_bad_price = int((df["price"] <= 0).sum())

    print(f"\n{'=' * 58}")
    print(f"  {month}")
    print(f"{'=' * 58}")
    print(f"  列數              {n:>15,}")
    print(f"  記憶體            {mem_gb:>14.2f} GB")
    print(f"  不重複使用者      {df['user_id'].nunique():>15,}")
    print(f"  不重複 session    {df['user_session'].nunique():>15,}")
    print(f"  不重複商品        {df['product_id'].nunique():>15,}")

    print("\n  -- 事件組成 --")
    for event in EVENT_ORDER:
        c = int(counts.get(event, 0))
        print(f"  {event:<18}{c:>15,}  ({c / n:>6.2%})")

    print("\n  -- 缺失與異常 --")
    for col in ("brand", "category_code"):
        miss = df[col].isna().mean()
        print(f"  {col} 缺失率{'':<6}{miss:>14.2%}")
    print(f"  完全重複列        {n_dup:>15,}")
    print(f"  price <= 0        {n_bad_price:>15,}")

    return {
        "month": month,
        "rows": n,
        "users": int(df["user_id"].nunique()),
        "sessions": int(df["user_session"].nunique()),
        "products": int(df["product_id"].nunique()),
        "purchases": int(counts.get("purchase", 0)),
        "purchase_rate": float(counts.get("purchase", 0) / n),
        "brand_missing_rate": float(df["brand"].isna().mean()),
        "duplicated_rows": n_dup,
        "bad_price_rows": n_bad_price,
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """最小必要清洗。

    刻意「少做」—— 每刪一列都要說得出理由。
    面試官問「你為什麼刪掉這些資料」時，答不出來就是扣分。
    """
    before = len(df)

    # 1. 完全重複的列：同一使用者同一秒對同一商品的同一動作，是追蹤重複上報
    df = df.drop_duplicates()

    # 2. price <= 0：價格為 0 或負數是資料錯誤，無法用於營收換算
    df = df[df["price"] > 0]

    # 3. user_session 為空：沒有 session ID 就無法做 session 層級分析
    df = df[df["user_session"].notna()]

    removed = before - len(df)
    print(f"\n  清洗移除 {removed:,} 列 ({removed / before:.3%})")
    return df


def main() -> None:
    summaries = []

    for month in MONTHS:
        try:
            df = load_month(month)
        except FileNotFoundError as exc:
            print(f"[X] {exc}", file=sys.stderr)
            sys.exit(1)

        summaries.append(quality_report(df, month))
        df = clean(df)

        # 依時間排序後再存：後面做 session 特徵時需要事件的先後順序，
        # 先排好可以省掉每次重排的成本
        df = df.sort_values(["user_session", "event_time"]).reset_index(drop=True)

        out_path = INTERIM_DIR / f"{month}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        csv_mb = (RAW_DIR / f"{month}.csv").stat().st_size / 1024 ** 2
        pq_mb = out_path.stat().st_size / 1024 ** 2
        print(f"  已存檔 → {out_path.name}  ({csv_mb:.0f}MB → {pq_mb:.0f}MB)")

        del df  # 手動釋放，避免 5 個月的資料同時佔著記憶體

    summary_df = pd.DataFrame(summaries)
    summary_path = INTERIM_DIR / "_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 58}")
    print("  總計")
    print(f"{'=' * 58}")
    print(f"  總事件數      {summary_df['rows'].sum():>15,}")
    print(f"  總購買數      {summary_df['purchases'].sum():>15,}")
    print(f"  整體購買佔比  {summary_df['purchases'].sum() / summary_df['rows'].sum():>14.2%}")
    print(f"\n  品質報告已存 → {summary_path}")
    print("\n[OK] 下一步：python -m src.funnel")


if __name__ == "__main__":
    main()
