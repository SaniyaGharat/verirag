"""
Corpus Cleaning Audit Script
============================
Performs a deep-dive audit on verirag/data/processed/ildc_cleaned.parquet:

1. Identifies and displays 15-20 concrete examples where a legitimate 'number'
   phrase (e.g. 'number of X', 'case number', 'large number') matched in RAW text
   but was altered in CLEANED text, showing side-by-side sentences.
2. Scans text_cleaned for any remaining words starting with 'company' that are not
   in the legitimate set {company, companies, company's, accompany, accompanying,
   accompanied, accompaniment, intercompany}.
3. Scans text_cleaned for standalone 'number' NOT preceded by standard
   articles/quantifiers and NOT followed by 'of', printing 15 sample sentences
   with document IDs to check for missed corruptions vs genuine nouns.
"""

import os
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from datasets import load_dataset

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "ildc_cleaned_v4.parquet" if (PROCESSED_DIR / "ildc_cleaned_v4.parquet").exists() else (PROCESSED_DIR / "ildc_cleaned_v3.parquet" if (PROCESSED_DIR / "ildc_cleaned_v3.parquet").exists() else PROCESSED_DIR / "ildc_cleaned.parquet")

LEGIT_PATTERNS = [
    re.compile(r'\bnumber\s+of\b', re.I),
    re.compile(r'\bthe\s+number\b', re.I),
    re.compile(r'\ba\s+number\b', re.I),
    re.compile(r'\blarge\s+number\b', re.I),
    re.compile(r'\bsmall\s+number\b', re.I),
    re.compile(r'\btotal\s+number\b', re.I),
    re.compile(r'\bany\s+number\b', re.I),
    re.compile(r'\bcase\s+number\b', re.I),
    re.compile(r'\bserial\s+number\b', re.I),
    re.compile(r'\bregistration\s+number\b', re.I),
    re.compile(r'\border\s+number\b', re.I),
    re.compile(r'\brule\s+number\b', re.I),
    re.compile(r'\bsection\s+number\b', re.I),
    re.compile(r'\bitem\s+number\b', re.I),
    re.compile(r'\bpage\s+number\b', re.I),
    re.compile(r'\bequal\s+number\b', re.I),
    re.compile(r'\bgreater\s+number\b', re.I),
    re.compile(r'\blesser\s+number\b', re.I),
    re.compile(r'\bin\s+number\b', re.I),
]

LEGIT_COMPANY_SET = {
    "company",
    "companies",
    "company's",
    "companys",
    "accompany",
    "accompanying",
    "accompanied",
    "accompaniment",
    "accompaniments",
    "intercompany",
}

STANDALONE_NUMBER_EXCLUDE_BEFORE = re.compile(
    r'\b(?:a|the|large|small|total|any|equal|greater|lesser|in|case|serial|registration|order|rule|section|item|page)\s+$',
    re.I
)


def run_audit():
    print("=" * 80)
    print("CORPUS CLEANING AUDIT ON verirag/data/processed/ildc_cleaned.parquet")
    print("=" * 80)

    print("Loading cleaned Parquet dataframe...")
    df_clean = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df_clean):,} cleaned documents.")

    print("Loading raw ILDC train split for side-by-side comparison...")
    ds_raw = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE", split="train")
    print(f"Loaded {len(ds_raw):,} raw documents.\n")

    # -------------------------------------------------------------
    # PART 1: Altered Legitimate 'number' Phrases
    # -------------------------------------------------------------
    print("=" * 80)
    print("PART 1: SIDE-BY-SIDE EXAMPLES WHERE A 'LEGITIMATE NUMBER' PHRASE WAS ALTERED")
    print("=" * 80)
    print("Investigating cases where a match for legit_patterns existed in RAW text")
    print("but was modified in CLEANED text...\n")

    altered_examples = []
    
    # Iterate through corpus to find altered instances
    for idx in range(len(df_clean)):
        raw_text = ds_raw[idx]["text"]
        clean_text = df_clean.iloc[idx]["text_cleaned"]
        doc_id = df_clean.iloc[idx]["id"]

        # Fast check: did counts decrease?
        raw_legit_count = sum(len(p.findall(raw_text)) for p in LEGIT_PATTERNS)
        clean_legit_count = sum(len(p.findall(clean_text)) for p in LEGIT_PATTERNS)

        if raw_legit_count > clean_legit_count:
            # Locate matches in raw_text
            for p in LEGIT_PATTERNS:
                for match in p.finditer(raw_text):
                    matched_str = match.group(0)
                    m_start = match.start()
                    m_end = match.end()

                    # Surrounding snippet in raw
                    span_start = max(0, m_start - 70)
                    while span_start > 0 and raw_text[span_start] not in ' \t\n':
                        span_start -= 1
                    span_end = min(len(raw_text), m_end + 70)
                    while span_end < len(raw_text) and raw_text[span_end] not in ' \t\n':
                        span_end += 1

                    raw_snippet = raw_text[span_start:span_end].replace("\n", " ").strip()

                    # Check if this exact snippet is absent or altered in clean_text
                    if matched_str.lower() not in clean_text[max(0, m_start - 100):min(len(clean_text), m_end + 100)].lower():
                        # Find corresponding clean snippet
                        # Approximate matching region around m_start
                        c_start = max(0, m_start - 70)
                        while c_start > 0 and clean_text[c_start] not in ' \t\n':
                            c_start -= 1
                        c_end = min(len(clean_text), m_end + 70)
                        while c_end < len(clean_text) and clean_text[c_end] not in ' \t\n':
                            c_end += 1
                        clean_snippet = clean_text[c_start:c_end].replace("\n", " ").strip()

                        altered_examples.append((doc_id, matched_str, raw_snippet, clean_snippet))
                        break
                if len(altered_examples) >= 20:
                    break
        if len(altered_examples) >= 20:
            break

    print(f"Found {len(altered_examples)} representative altered examples:")
    print("-" * 80)
    for i, (doc_id, phrase, raw_s, clean_s) in enumerate(altered_examples, 1):
        print(f"[{i:2d}] Doc ID: {doc_id} (Target phrase: '{phrase}')")
        print(f"     RAW     : \"...{raw_s}...\"")
        print(f"     CLEANED : \"...{clean_s}...\"\n")

    # -------------------------------------------------------------
    # PART 2A: Full-Corpus Scan for Non-Legitimate 'company*' Words
    # -------------------------------------------------------------
    print("=" * 80)
    print("PART 2A: FULL-CORPUS SCAN FOR RESIDUAL OR UNEXPECTED 'company*' WORDS")
    print("=" * 80)
    print("Scanning all 38,904 cleaned documents for any word matching \\bcompany[a-z]+\\b")
    print(f"Excluding legitimate set: {sorted(list(LEGIT_COMPANY_SET))}\n")

    company_word_counts = Counter()

    for clean_text in df_clean["text_cleaned"]:
        matches = re.findall(r'\bcompany[a-z]+\b', clean_text, flags=re.I)
        for w in matches:
            w_lower = w.lower()
            if w_lower not in LEGIT_COMPANY_SET:
                company_word_counts[w_lower] += 1

    total_non_legit_company = sum(company_word_counts.values())
    unique_non_legit_company = len(company_word_counts)

    print(f"Total non-whitelisted 'company*' instances found : {total_non_legit_company:,}")
    print(f"Total unique non-whitelisted words               : {unique_non_legit_company:,}\n")

    if unique_non_legit_company > 0:
        print("Breakdown of unique words found (sorted by frequency):")
        for word, count in company_word_counts.most_common():
            print(f"  - '{word}' : {count:>5,} occurrences")
    else:
        print("-> Clean! Zero unexpected 'company*' words found.")

    # -------------------------------------------------------------
    # PART 2B: Full-Corpus Scan for Standalone 'number'
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2B: FULL-CORPUS SCAN FOR STANDALONE 'number' (NON-STANDARD CONTEXTS)")
    print("=" * 80)
    print("Scanning for standalone \\bnumber\\b NOT preceded by standard quantifiers/articles")
    print("and NOT followed by 'of'...\n")

    standalone_number_samples = []
    total_nonstandard_number_hits = 0

    for idx in range(len(df_clean)):
        clean_text = df_clean.iloc[idx]["text_cleaned"]
        doc_id = df_clean.iloc[idx]["id"]

        for m in re.finditer(r'\b(number)\b', clean_text, flags=re.I):
            # Check what's before
            before_text = clean_text[max(0, m.start() - 30):m.start()]
            after_text = clean_text[m.end():min(len(clean_text), m.end() + 25)]

            # Check if after is 'of'
            if re.match(r'^\s+of\b', after_text, flags=re.I):
                continue
            
            # Check if before ends with excluded quantifiers
            if STANDALONE_NUMBER_EXCLUDE_BEFORE.search(before_text):
                continue

            total_nonstandard_number_hits += 1

            if len(standalone_number_samples) < 20:
                s_start = max(0, m.start() - 60)
                while s_start > 0 and clean_text[s_start] not in ' \t\n':
                    s_start -= 1
                s_end = min(len(clean_text), m.end() + 70)
                while s_end < len(clean_text) and clean_text[s_end] not in ' \t\n':
                    s_end += 1
                snippet = clean_text[s_start:s_end].replace("\n", " ").strip()
                standalone_number_samples.append((doc_id, snippet))

    print(f"Total non-standard standalone 'number' instances found in corpus: {total_nonstandard_number_hits:,}")
    print(f"Sample of {len(standalone_number_samples)} sentences for manual verification:\n")
    print("-" * 80)
    for i, (doc_id, snippet) in enumerate(standalone_number_samples, 1):
        print(f"[{i:2d}] Doc ID: {doc_id}")
        print(f"     Snippet: \"...{snippet}...\"\n")

    print("=" * 80)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
