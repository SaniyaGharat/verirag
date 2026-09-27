"""
Corpus-Wide Text Cleaning and Parquet Persistence Pipeline (Version 3)
======================================================================
Applies the enhanced dictionary-driven reconstruction pipeline (clean_corpus.py v3)
with candidate expansion, space-splitting, and sanity guardrails across all
38,904 judgments in jayadityagandham9/ILDC_35k_COMPLETE (train split).

Outputs:
- verirag/data/processed/ildc_cleaned_v3.parquet
- low_confidence_resolutions.csv (in artifact directory)
- unresolved_words_v3.csv (in artifact directory)
- comprehensive report artifact
"""

import os
import re
import sys
import time
from pathlib import Path
import multiprocessing as mp
from collections import Counter
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Suppress Hugging Face Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root and data dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))

import clean_corpus

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET_V3 = PROCESSED_DIR / "ildc_cleaned_v3.parquet"
OUTPUT_PARQUET_V2 = PROCESSED_DIR / "ildc_cleaned_v2.parquet"
OUTPUT_PARQUET_V1 = PROCESSED_DIR / "ildc_cleaned.parquet"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(PROJECT_ROOT / "data" / "reports")))

# Audit constants from audit_cleaning.py
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


def process_batch_v3(batch_rows):
    """
    Cleans a batch of rows using v3 reconstruction and extracts statistics.
    """
    results = []
    local_reconstructed_company = Counter()
    local_reconstructed_number = Counter()
    local_space_split_company = Counter()
    local_space_split_number = Counter()
    local_unresolved_company = Counter()
    local_unresolved_number = Counter()
    local_low_confidence = []

    legit_before_count = 0
    legit_after_count = 0

    token_pat = clean_corpus.TOKEN_CORRUPTION_PATTERN

    for row_id, raw_text, label in batch_rows:
        raw_text_str = str(raw_text) if raw_text is not None else ""
        raw_len = len(raw_text_str)

        # Count legitimate number phrases before
        for p in LEGIT_PATTERNS:
            legit_before_count += len(p.findall(raw_text_str))

        # Inspect all tokens containing company or number
        for m in token_pat.finditer(raw_text_str):
            tok = m.group(0)
            tok_low = tok.lower()
            if 'company' in tok_low:
                if tok_low not in clean_corpus.LEGIT_COMPANY_WORDS:
                    res, status, meta = clean_corpus._reconstruct_token_generic(
                        tok, clean_corpus.COMPANY_CANDIDATES, is_company=True
                    )
                    if status in ("single", "two_word", "single_nltk"):
                        local_reconstructed_company[(tok_low, res.lower())] += 1
                        if status == "two_word":
                            local_space_split_company[(tok_low, res.lower())] += 1
                    elif status == "low_confidence":
                        local_low_confidence.append(meta)
                    elif status == "unresolved":
                        local_unresolved_company[tok_low] += 1
            elif 'number' in tok_low:
                if tok_low not in clean_corpus.LEGIT_NUMBER_WORDS:
                    res, status, meta = clean_corpus._reconstruct_token_generic(
                        tok, clean_corpus.NUMBER_CANDIDATES, is_company=False
                    )
                    if status in ("single", "two_word", "single_nltk"):
                        local_reconstructed_number[(tok_low, res.lower())] += 1
                        if status == "two_word":
                            local_space_split_number[(tok_low, res.lower())] += 1
                    elif status == "low_confidence":
                        local_low_confidence.append(meta)
                    elif status == "unresolved":
                        local_unresolved_number[tok_low] += 1

        # Perform actual text cleaning
        cleaned_str = clean_corpus.clean_text(raw_text_str)
        cleaned_len = len(cleaned_str)

        # Count legitimate number phrases after
        for p in LEGIT_PATTERNS:
            legit_after_count += len(p.findall(cleaned_str))

        results.append({
            "id": str(row_id),
            "text_cleaned": cleaned_str,
            "text_raw_length": raw_len,
            "text_cleaned_length": cleaned_len,
            "label": int(label) if label is not None else -1
        })

    return (results, local_reconstructed_company, local_reconstructed_number,
            local_space_split_company, local_space_split_number,
            local_unresolved_company, local_unresolved_number,
            local_low_confidence, legit_before_count, legit_after_count)


def main():
    print("=" * 80)
    print("CORPUS CLEANING PIPELINE V3: EXPANDED CANDIDATES + SPACE-SPLITTING + GUARDRAILS")
    print("=" * 80)

    start_time = time.time()

    # 1. Load raw dataset
    print("[1/6] Loading full ILDC train split from Hugging Face cache...")
    ds = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE", split="train")
    total_docs = len(ds)
    print(f"      Loaded {total_docs:,} judgments into memory.")

    # 2. Distribute batches for multiprocessing
    num_cpus = max(1, min(14, os.cpu_count() or 4))
    batch_size = 250
    print(f"[2/6] Preparing parallel processing with {num_cpus} workers (batch size: {batch_size})...")

    raw_data = [(ds[i]["id"], ds[i]["text"], ds[i]["label"]) for i in range(total_docs)]
    batches = [raw_data[i:i + batch_size] for i in range(0, total_docs, batch_size)]

    all_cleaned_rows = []
    total_reconstructed_company = Counter()
    total_reconstructed_number = Counter()
    total_space_split_company = Counter()
    total_space_split_number = Counter()
    total_unresolved_company = Counter()
    total_unresolved_number = Counter()
    all_low_confidence = []
    corpus_legit_before = 0
    corpus_legit_after = 0

    # 3. Clean corpus
    print("[3/6] Cleaning text using v3 reconstruction engine...")
    with mp.Pool(processes=num_cpus) as pool:
        for batch_res in tqdm(pool.imap(process_batch_v3, batches), total=len(batches), desc="Processing Batches"):
            (rows, r_comp, r_num, s_comp, s_num, u_comp, u_num,
             low_conf, leg_b, leg_a) = batch_res
            all_cleaned_rows.extend(rows)
            total_reconstructed_company.update(r_comp)
            total_reconstructed_number.update(r_num)
            total_space_split_company.update(s_comp)
            total_space_split_number.update(s_num)
            total_unresolved_company.update(u_comp)
            total_unresolved_number.update(u_num)
            all_low_confidence.extend(low_conf)
            corpus_legit_before += leg_b
            corpus_legit_after += leg_a

    clean_duration = time.time() - start_time
    print(f"      Completed text cleaning in {clean_duration:.1f}s ({total_docs / clean_duration:.1f} docs/sec).")

    # 4. Save to Parquet V3
    print(f"[4/6] Saving cleaned corpus to {OUTPUT_PARQUET_V3}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_v3 = pd.DataFrame(all_cleaned_rows)
    df_v3.to_parquet(OUTPUT_PARQUET_V3, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = OUTPUT_PARQUET_V3.stat().st_size / (1024 * 1024)
    print(f"      Saved {len(df_v3):,} rows ({file_size_mb:.2f} MB on disk).")

    # 5. Round-trip QA check
    print("[5/6] Verifying Parquet round-trip integrity...")
    df_read = pd.read_parquet(OUTPUT_PARQUET_V3)
    assert len(df_read) == total_docs, "Row count mismatch in read verification!"

    # 6. Audit & Residual scans on V3
    print("[6/6] Executing Part 1 and Part 2 comparative audits on V3...")

    company_v3_counts = Counter()
    for clean_text_str in df_read["text_cleaned"]:
        matches = re.findall(r'\bcompany[a-z]+\b', clean_text_str, flags=re.I)
        for w in matches:
            w_lower = w.lower()
            if w_lower not in LEGIT_COMPANY_SET:
                company_v3_counts[w_lower] += 1

    total_non_legit_comp_v3 = sum(company_v3_counts.values())
    unique_non_legit_comp_v3 = len(company_v3_counts)

    total_nonstandard_number_v3 = 0
    for idx in range(len(df_read)):
        c_text = df_read.iloc[idx]["text_cleaned"]
        for m in re.finditer(r'\b(number)\b', c_text, flags=re.I):
            before_text = c_text[max(0, m.start() - 30):m.start()]
            after_text = c_text[m.end():min(len(c_text), m.end() + 25)]
            if re.match(r'^\s+of\b', after_text, flags=re.I):
                continue
            if STANDALONE_NUMBER_EXCLUDE_BEFORE.search(before_text):
                continue
            total_nonstandard_number_v3 += 1

    # Output detailed summary to stdout
    print("\n" + "=" * 80)
    print("V3 RECONSTRUCTION & CLEANING SUMMARY METRICS")
    print("=" * 80)
    print(f"Total documents processed                   : {total_docs:,}")
    print(f"Total processing runtime                    : {time.time() - start_time:.2f} seconds")
    print(f"Parquet file size (ildc_cleaned_v3.parquet) : {file_size_mb:.2f} MB")
    print()

    # Reconstructed types
    unique_comp_reconstructed = len(total_reconstructed_company)
    total_comp_instances_fixed = sum(total_reconstructed_company.values())
    unique_num_reconstructed = len(total_reconstructed_number)
    total_num_instances_fixed = sum(total_reconstructed_number.values())

    print("1. AUTOMATICALLY RECONSTRUCTED CORRUPTIONS:")
    print(f"   - Company-family unique word types reconstructed : {unique_comp_reconstructed:,}")
    print(f"     Total company-family word occurrences fixed   : {total_comp_instances_fixed:,}")
    print(f"   - Number-family unique word types reconstructed  : {unique_num_reconstructed:,}")
    print(f"     Total number-family word occurrences fixed    : {total_num_instances_fixed:,}")
    print(f"   - COMBINED TOTAL CORRUPTED INSTANCES FIXED       : {total_comp_instances_fixed + total_num_instances_fixed:,}")
    print()

    # Space-splitting reconstructions
    print("2. SPACE-INSERTION RESOLUTIONS (COMPOUND NUMBER+NOUN WORDS):")
    unique_num_space = len(total_space_split_number)
    total_num_space = sum(total_space_split_number.values())
    unique_comp_space = len(total_space_split_company)
    total_comp_space = sum(total_space_split_company.values())
    print(f"   - Number compound words split into two words     : {unique_num_space:,} unique types ({total_num_space:,} total occurrences)")
    print(f"   - Company compound words split into two words    : {unique_comp_space:,} unique types ({total_comp_space:,} total occurrences)")
    print()

    # Unresolved words
    print("3. UNRESOLVED WORDS (ZERO VALID CANDIDATES - UNCHANGED & LOGGED):")
    print(f"   - Company-family unresolved unique word types    : {len(total_unresolved_company):,}")
    print(f"     Total company-family unresolved occurrences   : {sum(total_unresolved_company.values()):,}")
    print(f"   - Number-family unresolved unique word types     : {len(total_unresolved_number):,}")
    print(f"     Total number-family unresolved occurrences    : {sum(total_unresolved_number.values()):,}")
    print()

    # Low-confidence guardrail items
    # Deduplicate low-confidence items
    low_conf_unique = {}
    for item in all_low_confidence:
        k = (item["family"], item["original"].lower())
        if k not in low_conf_unique:
            low_conf_unique[k] = {
                "family": item["family"],
                "original": item["original"],
                "mode": item["mode"],
                "winner": item["winner"],
                "winner_freq": item["winner_freq"],
                "competing": item["competing"],
                "competing_freq": item["competing_freq"],
                "candidates": ", ".join(item["candidates"]),
                "count": 0
            }
        low_conf_unique[k]["count"] += 1

    print("4. LOW-CONFIDENCE GUARDRAIL TRIGGERED:")
    print(f"   - Unique word types flagged as low-confidence    : {len(low_conf_unique):,}")
    print(f"   - Total occurrences protected from auto-accept   : {sum(v['count'] for v in low_conf_unique.values()):,}")
    print()

    # Part 1: Legitimate number preservation
    pres_rate = (corpus_legit_after / corpus_legit_before * 100.0) if corpus_legit_before else 100.0
    print("5. PART 1 AUDIT: LEGITIMATE 'number' NOUN PRESERVATION (FULL CORPUS):")
    print(f"   - Legitimate 'number' count BEFORE cleaning      : {corpus_legit_before:,}")
    print(f"   - Legitimate 'number' count AFTER cleaning       : {corpus_legit_after:,}")
    print(f"   - Legitimate 'number' preservation rate          : {pres_rate:.2f}%")
    print()

    # Part 2: Residual scan comparison
    print("6. PART 2 AUDIT: FULL-CORPUS RESIDUAL SCANS:")
    print(f"   - Residual non-whitelisted 'company*' words      : {total_non_legit_comp_v3:,} instances ({unique_non_legit_comp_v3:,} unique types)")
    print(f"   - Non-standard standalone 'number' instances     : {total_nonstandard_number_v3:,}")
    print()

    # Save artifact files
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    low_conf_csv = ARTIFACTS_DIR / "low_confidence_resolutions.csv"
    unres_v3_csv = ARTIFACTS_DIR / "unresolved_words_v3.csv"

    # Export low-confidence items to CSV
    df_low = pd.DataFrame(list(low_conf_unique.values()))
    if not df_low.empty:
        df_low.sort_values(by="count", ascending=False, inplace=True)
    df_low.to_csv(low_conf_csv, index=False)
    print(f"Saved low-confidence resolutions to: {low_conf_csv}")

    # Export unresolved words to CSV
    unresolved_rows = []
    for w, c in total_unresolved_company.most_common():
        unresolved_rows.append({"family": "company", "word": w, "count": c})
    for w, c in total_unresolved_number.most_common():
        unresolved_rows.append({"family": "number", "word": w, "count": c})
    pd.DataFrame(unresolved_rows).to_csv(unres_v3_csv, index=False)
    print(f"Saved complete v3 unresolved word list to: {unres_v3_csv}")

    print("Pipeline run complete.")


if __name__ == "__main__":
    main()
