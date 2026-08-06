"""把所有圖表輸出成獨立的 HTML 檔。

用法：
    python -m src.export_figures

【為什麼要有這支程式】
原本漏斗圖的繪製邏輯同時存在於 funnel.py 和 charts.py 兩份程式裡，
改了一邊另一邊不會跟著變 —— 這種重複遲早會導致同一份報告出現兩個
不一致的數字。現在所有圖表只在 charts.py 定義一次，
儀表板和獨立 HTML 檔都從那裡取用。
"""

from src import charts
from src.config import FIGURE_DIR

FIGURES = {
    "price_contrast": charts.price_contrast,
    "funnel_overall": charts.funnel_chart,
    "model_decile_lift": charts.decile_lift,
    "model_feature_importance": charts.feature_importance,
    "impact_sensitivity": charts.sensitivity_chart,
    "impact_breakeven": charts.breakeven_chart,
    "observation_window": charts.window_tradeoff_chart,
    "cart_curve": charts.cart_curve_chart,
    "solution_comparison": charts.solution_comparison,
    "discovery_overlap": charts.overlap_heatmap,
    "discovery_cohort_spread": charts.cohort_spread_chart,
}


def main() -> None:
    for name, fn in FIGURES.items():
        path = FIGURE_DIR / f"{name}.html"
        fn().write_html(path, include_plotlyjs="cdn")
        print(f"  已輸出 → {path.name}")

    print(f"\n[OK] {len(FIGURES)} 張圖已存至 {FIGURE_DIR}")


if __name__ == "__main__":
    main()
