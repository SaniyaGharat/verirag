"""
Corpus-Wide Text Cleaning and Parquet Persistence Pipeline (Version 4 - Final)
==============================================================================
Applies the final dictionary-driven reconstruction pipeline (clean_corpus.py v4)
with full candidate sets (including 'cop' and 'cos'), space-splitting, and sanity
guardrails across all 38,904 judgments in jayadityagandham9/ILDC_35k_COMPLETE (train split).

Outputs:
- verirag/data/processed/ildc_cleaned_v4.parquet
- low_confidence_resolutions.csv (in artifact directory)
- unresolved_words_v4.csv (in artifact directory)
- Deletes v1 and v2 parquets as requested
"""

import os
import re
import sys
import time
from pathlib import Path
import multiprocessing as mp
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# Suppress Hugging Face Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root and data dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))

import clean_corpus

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET_V4 = PROCESSED_DIR / "ildc_cleaned_v4.parquet"
OUTPUT_PARQUET_V3 = PROCESSED_DIR / "ildc_cleaned_v3.parquet"
OUTPUT_PARQUET_V2 = PROCESSED_DIR / "ildc_cleaned_v2.parquet"
OUTPUT_PARQUET_V1 = PROCESSED_DIR / "ildc_cleaned.parquet"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(PROJECT_ROOT / "data" / "reports")))

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


def process_batch_v4(batch_rows):
    """
    Cleans a batch of rows using v4 reconstruction and extracts statistics.
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
    total_tokens_count = 0

    token_pat = clean_corpus.TOKEN_CORRUPTION_PATTERN

    for row_id, raw_text, label in batch_rows:
        raw_text_str = str(raw_text) if raw_text is not None else ""
        raw_len = len(raw_text_str)
        # Fast memory-efficient token count (avoid allocating large split lists)
        total_tokens_count += raw_text_str.count(" ") + 1

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
            local_low_confidence, legit_before_count, legit_after_count, total_tokens_count)


def main():
    print("=" * 80)
    print("CORPUS CLEANING PIPELINE V4: FINAL PASS WITH EXPANDED CANDIDATES")
    print("=" * 80)

    start_time = time.time()

    import argparse
    parser = argparse.ArgumentParser(description="Clean ILDC corpus with v4 reconstruction engine.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of documents to process")
    args = parser.parse_args()

    # 1. Load raw dataset
    print("[1/7] Loading full ILDC train split from Hugging Face cache...")
    ds = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE", split="train")
    total_docs = min(args.limit, len(ds)) if args.limit is not None else len(ds)
    print(f"      Loaded {total_docs:,} judgments into memory.")

    # 2. Distribute batches for multiprocessing
    num_cpus = min(4, os.cpu_count() or 4)
    batch_size = 100
    print(f"[2/7] Preparing parallel processing with {num_cpus} workers (batch size: {batch_size})...")

    num_batches = (total_docs + batch_size - 1) // batch_size

    def batch_generator():
        for i in range(0, total_docs, batch_size):
            end = min(i + batch_size, total_docs)
            yield [(ds[j]["id"], ds[j]["text"], ds[j]["label"]) for j in range(i, end)]

    total_reconstructed_company = Counter()
    total_reconstructed_number = Counter()
    total_space_split_company = Counter()
    total_space_split_number = Counter()
    total_unresolved_company = Counter()
    total_unresolved_number = Counter()
    all_low_confidence = []
    corpus_legit_before = 0
    corpus_legit_after = 0
    corpus_total_tokens = 0

    # 3. Clean corpus and stream directly to ParquetWriter
    print(f"[3/7] Cleaning text and streaming directly to {OUTPUT_PARQUET_V4}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    temp_output = OUTPUT_PARQUET_V4.with_suffix(".tmp.parquet")

    schema = pa.schema([
        ("id", pa.string()),
        ("text_cleaned", pa.string()),
        ("text_raw_length", pa.int64()),
        ("text_cleaned_length", pa.int64()),
        ("label", pa.int64()),
    ])
    writer = pq.ParquetWriter(temp_output, schema=schema, compression="snappy")

    with mp.Pool(processes=num_cpus) as pool:
        for batch_res in tqdm(pool.imap(process_batch_v4, batch_generator()), total=num_batches, desc="Processing Batches"):
            (rows, r_comp, r_num, s_comp, s_num, u_comp, u_num,
             low_conf, leg_b, leg_a, tok_cnt) = batch_res

            batch_table = pa.Table.from_pydict({
                "id": [r["id"] for r in rows],
                "text_cleaned": [r["text_cleaned"] for r in rows],
                "text_raw_length": [r["text_raw_length"] for r in rows],
                "text_cleaned_length": [r["text_cleaned_length"] for r in rows],
                "label": [r["label"] for r in rows]
            }, schema=schema)
            writer.write_table(batch_table)
            del rows, batch_table

            total_reconstructed_company.update(r_comp)
            total_reconstructed_number.update(r_num)
            total_space_split_company.update(s_comp)
            total_space_split_number.update(s_num)
            total_unresolved_company.update(u_comp)
            total_unresolved_number.update(u_num)
            all_low_confidence.extend(low_conf)
            corpus_legit_before += leg_b
            corpus_legit_after += leg_a
            corpus_total_tokens += tok_cnt

    writer.close()
    if OUTPUT_PARQUET_V4.exists():
        OUTPUT_PARQUET_V4.unlink()
    temp_output.rename(OUTPUT_PARQUET_V4)

    clean_duration = time.time() - start_time
    file_size_mb = OUTPUT_PARQUET_V4.stat().st_size / (1024 * 1024)
    print(f"      Completed text cleaning in {clean_duration:.1f}s ({total_docs / clean_duration:.1f} docs/sec).")
    print(f"      Saved {total_docs:,} rows to {OUTPUT_PARQUET_V4} ({file_size_mb:.2f} MB on disk).")

    # 4. Round-trip QA check
    print("[4/7] Verifying Parquet round-trip integrity...")
    pf = pq.ParquetFile(OUTPUT_PARQUET_V4)
    assert pf.metadata.num_rows == total_docs, f"Row count mismatch! {pf.metadata.num_rows} vs {total_docs}"

    # 5. Delete v1 and v2 to save disk space
    print("[5/7] Managing checkpoints: deleting v1 and v2, keeping v3 and v4...")
    if OUTPUT_PARQUET_V1.exists():
        OUTPUT_PARQUET_V1.unlink()
        print(f"      Deleted {OUTPUT_PARQUET_V1.name}")
    if OUTPUT_PARQUET_V2.exists():
        OUTPUT_PARQUET_V2.unlink()
        print(f"      Deleted {OUTPUT_PARQUET_V2.name}")

    # 6. Audit & Residual scans on V4 (streamed to avoid loading entire corpus into RAM)
    print("[6/7] Executing Part 1 and Part 2 comparative audits on V4...")

    company_v4_counts = Counter()
    total_nonstandard_number_v4 = 0

    for batch in pf.iter_batches(batch_size=1000, columns=["text_cleaned"]):
        for clean_text_str in batch.column(0).to_pylist():
            matches = re.findall(r'\bcompany[a-z]+\b', clean_text_str, flags=re.I)
            for w in matches:
                w_lower = w.lower()
                if w_lower not in LEGIT_COMPANY_SET:
                    company_v4_counts[w_lower] += 1

            for m in re.finditer(r'\b(number)\b', clean_text_str, flags=re.I):
                before_text = clean_text_str[max(0, m.start() - 30):m.start()]
                after_text = clean_text_str[m.end():min(len(clean_text_str), m.end() + 25)]
                if re.match(r'^\s+of\b', after_text, flags=re.I):
                    continue
                if STANDALONE_NUMBER_EXCLUDE_BEFORE.search(before_text):
                    continue
                total_nonstandard_number_v4 += 1

    total_non_legit_comp_v4 = sum(company_v4_counts.values())
    unique_non_legit_comp_v4 = len(company_v4_counts)

    # Output detailed summary to stdout
    print("\n" + "=" * 80)
    print("V4 FINAL RECONSTRUCTION & CLEANING SUMMARY METRICS")
    print("=" * 80)
    print(f"Total documents processed                   : {total_docs:,}")
    print(f"Total processing runtime                    : {time.time() - start_time:.2f} seconds")
    print(f"Total corpus tokens evaluated               : {corpus_total_tokens:,}")
    print(f"Parquet file size (ildc_cleaned_v4.parquet) : {file_size_mb:.2f} MB")
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
    print("2. SPACE-INSERTION RESOLUTIONS:")
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
    print(f"   - Residual non-whitelisted 'company*' words      : {total_non_legit_comp_v4:,} instances ({unique_non_legit_comp_v4:,} unique types)")
    print(f"   - Non-standard standalone 'number' instances     : {total_nonstandard_number_v4:,}")
    print()

    # Save artifact files
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    low_conf_csv = ARTIFACTS_DIR / "low_confidence_resolutions.csv"
    unres_v4_csv = ARTIFACTS_DIR / "unresolved_words_v4.csv"

    df_low = pd.DataFrame(list(low_conf_unique.values()))
    if not df_low.empty:
        df_low.sort_values(by="count", ascending=False, inplace=True)
    df_low.to_csv(low_conf_csv, index=False)
    print(f"Saved low-confidence resolutions to: {low_conf_csv}")

    unresolved_rows = []
    for w, c in total_unresolved_company.most_common():
        unresolved_rows.append({"family": "company", "word": w, "count": c})
    for w, c in total_unresolved_number.most_common():
        unresolved_rows.append({"family": "number", "word": w, "count": c})
    pd.DataFrame(unresolved_rows).to_csv(unres_v4_csv, index=False)
    print(f"Saved complete v4 unresolved word list to: {unres_v4_csv}")

    print("Pipeline run complete.")


if __name__ == "__main__":
    main()
