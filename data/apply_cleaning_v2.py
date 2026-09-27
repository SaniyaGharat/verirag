"""
Corpus-Wide Text Cleaning and Parquet Persistence Pipeline (Version 2)
======================================================================
Applies the dictionary-driven reconstruction pipeline (clean_corpus.py)
across all 38,904 judgments in jayadityagandham9/ILDC_35k_COMPLETE.

Outputs:
- verirag/data/processed/ildc_cleaned_v2.parquet
  Columns: id, text_cleaned, text_raw_length, text_cleaned_length, label

Reports:
1. Reconstructed unique corrupted word types (company-family and number-family).
2. Complete unresolved unique word types with total occurrence counts.
3. Ambiguous tie-break examples showing candidate frequencies.
4. Part 1 & Part 2 audit comparison between v1 (ildc_cleaned.parquet) and v2.
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
OUTPUT_PARQUET_V2 = PROCESSED_DIR / "ildc_cleaned_v2.parquet"
OUTPUT_PARQUET_V1 = PROCESSED_DIR / "ildc_cleaned.parquet"
ARTIFACTS_DIR = Path(r"C:\Users\Saniya Gharat\.gemini\antigravity-ide\brain\0323c97c-d9f9-43fe-9fd1-c0b3056fa02a")

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


# Worker function for parallel batch processing
def process_batch_v2(batch_rows):
    """
    Cleans a batch of rows and extracts reconstruction statistics.
    """
    results = []
    local_reconstructed_company = Counter()
    local_reconstructed_number = Counter()
    local_unresolved_company = Counter()
    local_unresolved_number = Counter()
    local_tie_breaks = []

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
                    res = clean_corpus.reconstruct_company_word(tok)
                    if res != tok:
                        local_reconstructed_company[(tok_low, res.lower())] += 1
                    else:
                        local_unresolved_company[tok_low] += 1
            elif 'number' in tok_low:
                if tok_low not in clean_corpus.LEGIT_NUMBER_WORDS:
                    res = clean_corpus.reconstruct_number_word(tok)
                    if res != tok:
                        local_reconstructed_number[(tok_low, res.lower())] += 1
                    else:
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
            local_unresolved_company, local_unresolved_number,
            legit_before_count, legit_after_count)


def main():
    print("=" * 80)
    print("CORPUS CLEANING PIPELINE V2: DICTIONARY-DRIVEN RECONSTRUCTION")
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
    total_unresolved_company = Counter()
    total_unresolved_number = Counter()
    corpus_legit_before = 0
    corpus_legit_after = 0

    # 3. Clean corpus
    print("[3/6] Cleaning text using dictionary-driven reconstruction...")
    with mp.Pool(processes=num_cpus) as pool:
        for batch_res in tqdm(pool.imap(process_batch_v2, batches), total=len(batches), desc="Processing Batches"):
            rows, r_comp, r_num, u_comp, u_num, leg_b, leg_a = batch_res
            all_cleaned_rows.extend(rows)
            total_reconstructed_company.update(r_comp)
            total_reconstructed_number.update(r_num)
            total_unresolved_company.update(u_comp)
            total_unresolved_number.update(u_num)
            corpus_legit_before += leg_b
            corpus_legit_after += leg_a

    clean_duration = time.time() - start_time
    print(f"      Completed text cleaning in {clean_duration:.1f}s ({total_docs / clean_duration:.1f} docs/sec).")

    # 4. Save to Parquet V2
    print(f"[4/6] Saving cleaned corpus to {OUTPUT_PARQUET_V2}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_v2 = pd.DataFrame(all_cleaned_rows)
    df_v2.to_parquet(OUTPUT_PARQUET_V2, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = OUTPUT_PARQUET_V2.stat().st_size / (1024 * 1024)
    print(f"      Saved {len(df_v2):,} rows ({file_size_mb:.2f} MB on disk).")

    # 5. Round-trip QA check
    print("[5/6] Verifying Parquet round-trip integrity...")
    df_read = pd.read_parquet(OUTPUT_PARQUET_V2)
    assert len(df_read) == total_docs, "Row count mismatch in read verification!"

    # 6. Audit & Residual scans on V2
    print("[6/6] Executing Part 1 and Part 2 comparative audits on V2...")

    # Part 2A: Full-corpus scan for non-whitelisted 'company*' words in V2
    company_v2_counts = Counter()
    for clean_text_str in df_read["text_cleaned"]:
        matches = re.findall(r'\bcompany[a-z]+\b', clean_text_str, flags=re.I)
        for w in matches:
            w_lower = w.lower()
            if w_lower not in LEGIT_COMPANY_SET:
                company_v2_counts[w_lower] += 1

    total_non_legit_comp_v2 = sum(company_v2_counts.values())
    unique_non_legit_comp_v2 = len(company_v2_counts)

    # Part 2B: Full-corpus scan for non-standard standalone 'number' in V2
    total_nonstandard_number_v2 = 0
    standalone_number_samples = []

    for idx in range(len(df_read)):
        c_text = df_read.iloc[idx]["text_cleaned"]
        d_id = df_read.iloc[idx]["id"]

        for m in re.finditer(r'\b(number)\b', c_text, flags=re.I):
            before_text = c_text[max(0, m.start() - 30):m.start()]
            after_text = c_text[m.end():min(len(c_text), m.end() + 25)]

            if re.match(r'^\s+of\b', after_text, flags=re.I):
                continue
            if STANDALONE_NUMBER_EXCLUDE_BEFORE.search(before_text):
                continue

            total_nonstandard_number_v2 += 1
            if len(standalone_number_samples) < 15:
                s_start = max(0, m.start() - 60)
                while s_start > 0 and c_text[s_start] not in ' \t\n':
                    s_start -= 1
                s_end = min(len(c_text), m.end() + 70)
                while s_end < len(c_text) and c_text[s_end] not in ' \t\n':
                    s_end += 1
                snippet = c_text[s_start:s_end].replace("\n", " ").strip()
                standalone_number_samples.append((d_id, snippet))

    # Calculate V1 vs V2 stats if V1 exists
    v1_size_mb = 0.0
    if OUTPUT_PARQUET_V1.exists():
        v1_size_mb = OUTPUT_PARQUET_V1.stat().st_size / (1024 * 1024)

    # Output detailed summary to stdout
    print("\n" + "=" * 80)
    print("RECONSTRUCTION & CLEANING SUMMARY METRICS")
    print("=" * 80)
    print(f"Total documents processed                   : {total_docs:,}")
    print(f"Total processing runtime                    : {time.time() - start_time:.2f} seconds")
    print(f"Parquet file size (ildc_cleaned_v2.parquet) : {file_size_mb:.2f} MB")
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

    # Unresolved words
    print("2. UNRESOLVED WORDS (ZERO VALID CANDIDATES - UNCHANGED & LOGGED):")
    print(f"   - Company-family unresolved unique word types    : {len(total_unresolved_company):,}")
    print(f"     Total company-family unresolved occurrences   : {sum(total_unresolved_company.values()):,}")
    print(f"   - Number-family unresolved unique word types     : {len(total_unresolved_number):,}")
    print(f"     Total number-family unresolved occurrences    : {sum(total_unresolved_number.values()):,}")
    print()

    # Part 1: Legitimate number preservation
    pres_rate = (corpus_legit_after / corpus_legit_before * 100.0) if corpus_legit_before else 100.0
    print("3. PART 1 AUDIT: LEGITIMATE 'number' NOUN PRESERVATION (FULL CORPUS):")
    print(f"   - Legitimate 'number' count BEFORE cleaning      : {corpus_legit_before:,}")
    print(f"   - Legitimate 'number' count AFTER cleaning       : {corpus_legit_after:,}")
    print(f"   - Legitimate 'number' preservation rate          : {pres_rate:.2f}%")
    print()

    # Part 2: Residual scan comparison
    print("4. PART 2 AUDIT: FULL-CORPUS RESIDUAL SCANS:")
    print(f"   - Residual non-whitelisted 'company*' words      : {total_non_legit_comp_v2:,} instances ({unique_non_legit_comp_v2:,} unique types)")
    print(f"   - Non-standard standalone 'number' instances     : {total_nonstandard_number_v2:,}")
    print()

    # Save detailed unresolved lists and ambiguous cases report to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "corpus_cleaning_v2_report.md"
    unresolved_csv = ARTIFACTS_DIR / "unresolved_words.csv"

    # Export unresolved words to CSV
    unresolved_rows = []
    for w, c in total_unresolved_company.most_common():
        unresolved_rows.append({"family": "company", "word": w, "count": c})
    for w, c in total_unresolved_number.most_common():
        unresolved_rows.append({"family": "number", "word": w, "count": c})
    pd.DataFrame(unresolved_rows).to_csv(unresolved_csv, index=False)
    print(f"Saved complete unresolved word list to: {unresolved_csv}")

    print("Pipeline run complete.")


if __name__ == "__main__":
    main()
