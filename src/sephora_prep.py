"""模組 4 前置：建立同膚質評分表與成分特徵。

用法：
    python -m src.sephora_prep

【核心論點】
全站平均評分對個人沒有意義。這份資料的評分嚴重右偏 —— 82% 的評論給 4-5 星，
全站平均被壓縮在高分區，幾乎沒有鑑別度。但同一支產品對不同膚質的表現可能
差很多，「跟你同膚質的人給幾分」才是有用的訊號。

【為什麼一定要做貝氏平滑】
每項商品的評論數中位數是 164 則，但 P25 只有 37 則。拆成 4 種膚質後，
四分之一的商品每組只剩約 9 則評論。直接取平均的話，
「某產品在乾性肌 5.0 分」可能只是 2 個人給了 5 星 —— 那是雜訊不是訊號。
"""

import ast
import glob
import re
import warnings

import pandas as pd

from src.config import INTERIM_DIR, SEPHORA_DIR

OUT_DIR = INTERIM_DIR / "sephora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_COLS = [
    "author_id", "rating", "is_recommended",
    "skin_type", "skin_tone", "product_id", "submission_time",
]

# 平滑強度：相當於「先驗上先給每組 M 則中庸評論」。
# M 越大，小樣本被拉回整體平均的力道越強。
# 取 20 是因為它約等於評論數 P25(37) 拆成 4 組後的兩倍，
# 讓樣本數低於這個量級的組別明顯往整體平均收斂。
SMOOTH_PRODUCT = 20
SMOOTH_COHORT = 20

# ---------------------------------------------------------------- 成分規則
#
# 【子字串比對在成分表上必定出錯，一律用詞邊界比對】
#
# 化妝品成分名大量互相包含，用 `"x" in text` 會產生嚴重誤判：
#
#   "alcohol" ⊂ "cetyl alcohol"      脂肪醇是保濕劑，不是刺激物
#   "ethanol" ⊂ "phenoxyethanol"     防腐劑，溫和
#   "ethanol" ⊂ "triethanolamine"    酸鹼調節劑，與酒精無關
#
# 第一版只擋掉脂肪醇就以為安全了，結果 ethanol 誤判 1,291 項，
# 把含刺激性酒精的比例灌水到 56.6%（真實約 10%）。
# 教訓：不要逐一修個案，要換掉「子字串比對」這個做法本身。
#
# 現在的做法是逐一檢視成分 token（已按逗號切開），用詞邊界正則比對。

# 依 INCI 命名規則，成分表上單獨列出的 "Alcohol" 就是乙醇
DRYING_ALCOHOL_PATTERNS = [
    r"^alcohol$",
    r"^alcohol\s+denat",
    r"\bsd\s+alcohol\b",
    r"\bdenatured\s+alcohol\b",
    r"\bisopropyl\s+alcohol\b",
    r"\bethanol\b",          # 詞邊界擋掉 phenoxyethanol / triethanolamine
]

# 脂肪醇：長鏈醇類，作用是保濕與乳化，對皮膚溫和
FATTY_ALCOHOL_PATTERN = (
    r"\b(cetyl|stearyl|cetearyl|behenyl|myristyl|lauryl|oleyl|arachidyl)"
    r"\s+alcohol\b"
)

# 歐盟化妝品規範 (EC) No 1223/2009 附錄三要求標示的常見致敏香料
FRAGRANCE_ALLERGENS = [
    "limonene", "linalool", "citronellol", "geraniol", "eugenol",
    "coumarin", "citral", "farnesol", "isoeugenol", "benzyl salicylate",
]
FRAGRANCE_TERMS = ["fragrance", "parfum", "perfume"]

ESSENTIAL_OILS = [
    "essential oil", "peppermint oil", "eucalyptus", "menthol",
    "lavender oil", "citrus oil", "lemon oil", "tea tree oil",
]

# 需求對應：用商品分類與 highlights 標籤比對，不用臆測
NEED_KEYWORDS = {
    "保濕乾燥": ["hydrating", "moisturizer", "moisturizing", "dry skin", "hyaluronic"],
    "痘痘粉刺": ["acne", "blemish", "salicylic", "pore", "oil control", "mattifying"],
    "暗沉不均": ["brightening", "dark spot", "vitamin c", "radiance", "tone"],
    "細紋抗老": ["anti-aging", "firming", "retinol", "wrinkle", "peptide"],
    "泛紅舒緩": ["soothing", "calming", "redness", "sensitive", "centella", "cica"],
    "清潔去角質": ["cleanser", "exfoliat", "peeling", "aha", "bha"],
}


def parse_list_field(value) -> list[str]:
    """product_info 的 ingredients / highlights 是字串化的 list。"""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        # 部分成分文字含有 "\A" 之類的序列，literal_eval 會噴 SyntaxWarning。
        # 那不影響解析結果，但會把輸出洗版，所以在這裡壓掉。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.lower()]
    if isinstance(parsed, list):
        return [str(x).lower() for x in parsed]
    return [str(parsed).lower()]


def ingredient_tokens(raw) -> list[str]:
    """把成分欄位攤平成乾淨的成分名清單。"""
    items = parse_list_field(raw)
    tokens = []
    for chunk in items:
        # 有些商品用 "Product variation 1:" 分段，那不是成分
        if re.match(r"^\s*(product\s+variation|variation)\b", chunk):
            continue
        for part in chunk.split(","):
            t = re.sub(r"\s+", " ", part).strip(" .;:*")
            if 2 < len(t) < 80:
                tokens.append(t)
    return tokens


_DRYING_RE = [re.compile(p) for p in DRYING_ALCOHOL_PATTERNS]
_FATTY_RE = re.compile(FATTY_ALCOHOL_PATTERN)


def _is_drying_alcohol(token: str) -> bool:
    """逐一 token 判斷，而不是在整串成分文字上比對。

    token 已經按逗號切開，所以 "cetyl alcohol" 是完整的一個 token，
    不會與單獨列出的 "alcohol" 混淆。
    """
    if _FATTY_RE.search(token):
        return False
    return any(r.search(token) for r in _DRYING_RE)


def flag_ingredients(tokens: list[str]) -> dict:
    """標記需要注意的成分類別。全部使用詞邊界比對。"""
    has_drying = any(_is_drying_alcohol(t) for t in tokens)

    blob = " | ".join(tokens)
    allergens = [
        a for a in FRAGRANCE_ALLERGENS
        if re.search(rf"\b{re.escape(a)}\b", blob)
    ]
    has_fragrance = any(
        re.search(rf"\b{re.escape(f)}\b", blob) for f in FRAGRANCE_TERMS
    )
    oils = [
        o for o in ESSENTIAL_OILS
        if re.search(rf"\b{re.escape(o)}\b", blob)
    ]

    return {
        "n_ingredients": len(tokens),
        "has_drying_alcohol": has_drying,
        "has_fragrance": has_fragrance,
        "n_fragrance_allergens": len(allergens),
        "fragrance_allergens": ", ".join(allergens),
        "has_essential_oil": bool(oils),
        "irritant_score": (
            int(has_drying) * 2 + int(has_fragrance) + len(allergens) + len(oils)
        ),
    }


def match_needs(row) -> str:
    """依分類與 highlights 標籤對應到需求。"""
    blob = " ".join(filter(None, [
        str(row.get("product_name", "")),
        str(row.get("secondary_category", "")),
        str(row.get("tertiary_category", "")),
        " ".join(parse_list_field(row.get("highlights"))),
    ])).lower()

    hits = [need for need, kws in NEED_KEYWORDS.items() if any(k in blob for k in kws)]
    return ", ".join(hits)


def build_products() -> pd.DataFrame:
    p = pd.read_csv(SEPHORA_DIR / "product_info.csv")
    p = p[p["primary_category"] == "Skincare"].copy()

    tokens = p["ingredients"].apply(ingredient_tokens)
    flags = pd.DataFrame(list(tokens.apply(flag_ingredients)), index=p.index)
    p = pd.concat([p, flags], axis=1)

    p["ingredient_text"] = tokens.apply(lambda t: " ".join(t))
    p["needs"] = p.apply(match_needs, axis=1)
    p["price_final"] = p["sale_price_usd"].fillna(p["price_usd"])
    return p


def load_reviews() -> pd.DataFrame:
    files = sorted(glob.glob(str(SEPHORA_DIR / "reviews_*.csv")))
    return pd.concat(
        [pd.read_csv(f, usecols=REVIEW_COLS, low_memory=False) for f in files],
        ignore_index=True,
    )


def build_cohorts(reviews: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """兩層貝氏平滑的同膚質評分表。

    第一層：商品整體平均往全站平均收斂
    第二層：該膚質的平均往商品整體平均收斂

    這樣「這支產品對你的膚質好不好」問的是「相對於它自己的平均」，
    而不是「相對於全站」—— 後者只會讓好產品的每個膚質都顯示為好。
    """
    r = reviews.dropna(subset=["skin_type"])
    global_mean = reviews["rating"].mean()
    global_rec = reviews["is_recommended"].mean()

    prod = reviews.groupby("product_id").agg(
        n_all=("rating", "size"),
        rating_all=("rating", "mean"),
        rec_all=("is_recommended", "mean"),
    )
    prod["rating_all_s"] = (
        (prod["n_all"] * prod["rating_all"] + SMOOTH_PRODUCT * global_mean)
        / (prod["n_all"] + SMOOTH_PRODUCT)
    )
    prod["rec_all_s"] = (
        (prod["n_all"] * prod["rec_all"].fillna(global_rec) + SMOOTH_PRODUCT * global_rec)
        / (prod["n_all"] + SMOOTH_PRODUCT)
    )

    coh = r.groupby(["product_id", "skin_type"]).agg(
        n_cohort=("rating", "size"),
        rating_cohort=("rating", "mean"),
        rec_cohort=("is_recommended", "mean"),
    ).reset_index()

    coh = coh.merge(
        prod[["n_all", "rating_all_s", "rec_all_s"]],
        left_on="product_id", right_index=True, how="left",
    )

    coh["rating_score"] = (
        (coh["n_cohort"] * coh["rating_cohort"] + SMOOTH_COHORT * coh["rating_all_s"])
        / (coh["n_cohort"] + SMOOTH_COHORT)
    )
    coh["rec_score"] = (
        (coh["n_cohort"] * coh["rec_cohort"].fillna(coh["rec_all_s"])
         + SMOOTH_COHORT * coh["rec_all_s"])
        / (coh["n_cohort"] + SMOOTH_COHORT)
    )

    # 這支產品對這個膚質，比它自己的平均好多少
    coh["rating_delta"] = coh["rating_score"] - coh["rating_all_s"]
    coh["rec_delta"] = coh["rec_score"] - coh["rec_all_s"]

    stats = {
        "global_mean_rating": float(global_mean),
        "global_rec_rate": float(global_rec),
        "n_reviews": int(len(reviews)),
        "n_reviews_with_skin": int(len(r)),
        "n_cohort_rows": int(len(coh)),
    }
    return coh, stats


def main() -> None:
    print("[*] 建立商品表（Skincare）...")
    products = build_products()
    print(f"    {len(products):,} 項商品")
    print(f"    含刺激性酒精 {products['has_drying_alcohol'].mean():.1%}")
    print(f"    含香料       {products['has_fragrance'].mean():.1%}")
    print(f"    含精油       {products['has_essential_oil'].mean():.1%}")
    print(f"    可對應到需求 {(products['needs'] != '').mean():.1%}")

    print("\n[*] 載入評論...")
    reviews = load_reviews()
    print(f"    {len(reviews):,} 則")

    print("\n[*] 建立同膚質評分表...")
    cohorts, stats = build_cohorts(reviews)
    print(f"    {len(cohorts):,} 組 (商品 × 膚質)")
    print(f"    全站平均評分 {stats['global_mean_rating']:.3f}")
    print(f"    全站推薦率   {stats['global_rec_rate']:.1%}")

    products.to_parquet(OUT_DIR / "products.parquet", index=False)
    cohorts.to_parquet(OUT_DIR / "cohorts.parquet", index=False)
    pd.Series(stats).to_json(OUT_DIR / "stats.json", force_ascii=False, indent=2)

    print(f"\n[OK] 已存至 {OUT_DIR}")
    print("     下一步：python -m src.discovery")


if __name__ == "__main__":
    main()
