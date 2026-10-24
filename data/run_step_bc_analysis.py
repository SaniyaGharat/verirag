import re
import json
import random
import pandas as pd
from pathlib import Path
from collections import Counter

random.seed(42)

print("=" * 60)
print("RUNNING STEP B & STEP C ANALYSIS")
print("=" * 60)

# -------------------------------------------------------------
# STEP B: Bare Roman Numerals in KEEP_NUMBER
# -------------------------------------------------------------
print("\n--- STEP B: BARE ROMAN NUMERALS IN KEEP_NUMBER ---")

# Load occurrences
df_occ = pd.read_parquet("data/processed/negation/number_occurrences.parquet")
print(f"Loaded {len(df_occ):,} occurrences.")

# Load blind 200 to exclude
blind_df = pd.read_csv("data/reports/gold_number_200_blind.csv")
blind_occ_ids = set(blind_df["occ_id"].values)
print(f"Loaded {len(blind_occ_ids)} blind gold occ_ids to exclude.")

# Patterns
DIGIT_PAT = re.compile(r"^\s*\d+\b")
DOT_DIGIT_PAT = re.compile(r"^\s*\.\s*\d+")
ROMAN_ALL_PAT = re.compile(r"^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", re.IGNORECASE)
BARE_IVX_PAT = re.compile(r"^\s*(i|v|x)\b", re.IGNORECASE)

keep_number_breakdown = Counter()
bare_ivx_matches = []

for idx, row in df_occ.iterrows():
    r_ctx = row["right_ctx"]
    if not r_ctx:
        continue
    
    # Check trigger type in priority order of KEEP_NUMBER
    # 1. dot + digit
    if DOT_DIGIT_PAT.match(r_ctx):
        keep_number_breakdown["dot_digit"] += 1
    # 2. digit
    elif DIGIT_PAT.match(r_ctx):
        keep_number_breakdown["digit"] += 1
    # 3. roman numeral
    else:
        m_roman = ROMAN_ALL_PAT.match(r_ctx)
        if m_roman:
            matched_roman = m_roman.group(1).lower()
            keep_number_breakdown["roman_numeral"] += 1
            keep_number_breakdown[f"roman_{matched_roman}"] += 1
            
            # Check bare i, v, x and exclude blind 200
            if matched_roman in ("i", "v", "x") and row["occ_id"] not in blind_occ_ids:
                left_10 = " ".join(row["left_ctx"].split()[-10:])
                right_10 = " ".join(row["right_ctx"].split()[:10])
                bare_ivx_matches.append({
                    "occ_id": row["occ_id"],
                    "doc_id": row["doc_id"],
                    "roman": matched_roman,
                    "rule": "KEEP_NUMBER",
                    "left_10": left_10,
                    "right_10": right_10
                })

print("KEEP_NUMBER Trigger Breakdown:")
print(f"  Digit string       : {keep_number_breakdown['digit']:,}")
print(f"  '.' + digit        : {keep_number_breakdown['dot_digit']:,}")
print(f"  Roman numeral      : {keep_number_breakdown['roman_numeral']:,}")
total_keep = keep_number_breakdown['digit'] + keep_number_breakdown['dot_digit'] + keep_number_breakdown['roman_numeral']
print(f"  Total KEEP_NUMBER  : {total_keep:,}")

print("\nRoman Numeral Specific Breakdown:")
for r in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]:
    print(f"  '{r}' : {keep_number_breakdown[f'roman_{r}']:,}")

print(f"\nTotal bare 'i', 'v', 'x' occurrences (excluding 200 blind): {len(bare_ivx_matches):,}")
sample_25 = random.sample(bare_ivx_matches, min(25, len(bare_ivx_matches)))

# -------------------------------------------------------------
# STEP C: Excluded-Context Residuals
# -------------------------------------------------------------
print("\n--- STEP C: EXCLUDED-CONTEXT RESIDUALS ---")
df_dedup = pd.read_parquet("data/processed/ildc_v4_dedup.parquet")
print(f"Loaded {len(df_dedup):,} deduplicated documents.")

PRECEDING_WORDS = ["a", "the", "in", "case", "any"]
FOLLOWING_WORDS = [
    "only", "less", "more", "exceeding", "later", "earlier", "having",
    "being", "to", "be", "been", "available", "permissible", "entitled", "necessary"
]

prec_group = "(?:" + "|".join(PRECEDING_WORDS) + ")"
foll_group = "(?:" + "|".join(FOLLOWING_WORDS) + ")"

residual_pat = re.compile(
    r"\b(" + "|".join(PRECEDING_WORDS) + r")\s+(number)\s+(" + "|".join(FOLLOWING_WORDS) + r")\b",
    re.IGNORECASE
)

pair_counts = Counter()
pair_examples = { (p, f): [] for p in PRECEDING_WORDS for f in FOLLOWING_WORDS }

for _, row in df_dedup.iterrows():
    text = row["text_cleaned"]
    doc_id = str(row["id"])
    if not text or "number" not in text.lower():
        continue
    
    for m in residual_pat.finditer(text):
        p_word = m.group(1).lower()
        f_word = m.group(3).lower()
        pair = (p_word, f_word)
        pair_counts[pair] += 1
        
        if len(pair_examples[pair]) < 3:
            s_start = max(0, m.start() - 60)
            while s_start > 0 and text[s_start] not in ' \t\n':
                s_start -= 1
            s_end = min(len(text), m.end() + 60)
            while s_end < len(text) and text[s_end] not in ' \t\n':
                s_end += 1
            snippet = text[s_start:s_end].replace("\n", " ").strip()
            pair_examples[pair].append({
                "doc_id": doc_id,
                "snippet": snippet
            })

print(f"Total residual occurrences found across all 75 pairs: {sum(pair_counts.values()):,}")
print(f"Unique pairs observed: {len([p for p, c in pair_counts.items() if c > 0])} / 75")

# Save Step B & C results
results_bc = {
    "step_b": {
        "keep_number_breakdown": dict(keep_number_breakdown),
        "sample_25": sample_25
    },
    "step_c": {
        "pair_counts": {f"{k[0]} ... {k[1]}": v for k, v in pair_counts.most_common()},
        "top10_pairs_examples": [
            {
                "pair": f"{pair[0]} ... {pair[1]}",
                "count": count,
                "examples": pair_examples[pair]
            }
            for pair, count in pair_counts.most_common(10)
        ]
    }
}

with open("data/reports/step_b_and_c_results.json", "w", encoding="utf-8") as f:
    json.dump(results_bc, f, indent=2)

print("\nSaved Step B & C analysis to data/reports/step_b_and_c_results.json")
