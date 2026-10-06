"""
ILDC Corpus Chunking Pipeline
=============================
Splits cleaned ILDC court judgments (ildc_cleaned_v4.parquet) into overlapping
passages respecting natural sentence boundaries, saving to chunks.parquet.

Note on Chunking Unit:
---------------------
chunk_size and overlap are measured in WORDS (not characters) as a first pass.
Token-based chunking with the actual InLegalBERT tokenizer would be more precise,
and we may revisit this in Phase 1 refinement if retrieval quality suffers.
"""

import os
import re
import sys
import time
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "ildc_cleaned_v4.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"

# Common legal and standard abbreviations to avoid false sentence breaks
ABBREVIATIONS = {
    'v', 'vs', 'no', 'nos', 'sec', 'secs', 'art', 'dr', 'mr', 'mrs', 'ms',
    'rs', 'hon', 'ltd', 'co', 'inc', 'al', 'ibid', 'id', 'sk', 'md', 'st',
    'u', 'w', 'c.j', 'cj'
}

# Regex to detect sentence boundaries:
# Group 1: Preceding word or closing bracket/quote
# Group 2: Sentence-ending punctuation (., !, ?) optionally followed by quotes/brackets
# Group 3: Whitespace separating sentences
# Positive lookahead ensures following character begins a sentence (capital letter, digit, bracket, quote)
SENTENCE_SPLIT_REGEX = re.compile(
    r'([A-Za-z0-9]+|[)\]"\'])([.!?][\"\'\)]*)(\s+)(?=[A-Z0-9\"\'\(\[])'
)


def split_into_sentence_units(text: str, max_unit_words: int = 100) -> list[dict]:
    """
    Splits text into non-overlapping sentence/clause units with character spans and word counts.
    Preserves exact character offsets into text_cleaned.

    To prevent runaway sentence lengths in unpunctuated statutory quotes or legal lists,
    any sentence longer than max_unit_words is subdivided at comma/clause boundaries
    or word boundaries, ensuring atomic units can be cleanly composed into ~350-word chunks.
    """
    if not text:
        return []

    splits = []
    for m in SENTENCE_SPLIT_REGEX.finditer(text):
        last_word = m.group(1).lower()
        if last_word in ABBREVIATIONS:
            continue
        splits.append(m.start(3))

    raw_spans = []
    prev = 0
    for split_idx in splits:
        span_text = text[prev:split_idx]
        if span_text.strip():
            rel_start = len(span_text) - len(span_text.lstrip())
            rel_end = len(span_text.rstrip())
            raw_spans.append((prev + rel_start, prev + rel_end))
        prev = split_idx

    # Final sentence
    if prev < len(text):
        span_text = text[prev:]
        if span_text.strip():
            rel_start = len(span_text) - len(span_text.lstrip())
            rel_end = len(span_text.rstrip())
            raw_spans.append((prev + rel_start, prev + rel_end))

    # Subdivide units that exceed max_unit_words
    units = []
    for s_start, s_end in raw_spans:
        s_text = text[s_start:s_end]
        words = list(re.finditer(r'\S+', s_text))
        w_cnt = len(words)
        if w_cnt <= max_unit_words:
            units.append({
                "start": s_start,
                "end": s_end,
                "words": w_cnt
            })
        else:
            sub_start = 0
            curr_words = 0
            for i, w in enumerate(words):
                curr_words += 1
                is_last_word = (i == w_cnt - 1)
                has_comma = w.group().endswith(',') or (w.end() < len(s_text) and s_text[w.end():w.end() + 1] == ',')
                if is_last_word or (curr_words >= 50 and has_comma) or (curr_words >= max_unit_words):
                    c_abs_start = s_start + words[sub_start].start()
                    c_abs_end = s_start + w.end()
                    units.append({
                        "start": c_abs_start,
                        "end": c_abs_end,
                        "words": curr_words
                    })
                    sub_start = i + 1
                    curr_words = 0

    return units


def chunk_document(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
    doc_id: str = "doc"
) -> list[dict]:
    """
    Splits text_cleaned into overlapping chunks adhering to sentence/paragraph boundaries.

    Rules:
    1. chunk_size and overlap are measured in WORDS (not characters) as a first pass.
       Token-based chunking with the actual InLegalBERT tokenizer would be more precise,
       and we may revisit this in Phase 1 refinement if retrieval quality suffers.
    2. Prefer breaking on paragraph or sentence boundaries near the target chunk_size
       rather than cutting mid-sentence.
    3. Each chunk contains:
       - chunk_id: f"{doc_id}_c{chunk_index}"
       - doc_id: str(doc_id)
       - chunk_index: int
       - text: str
       - word_count: int
       - char_start: int (character offset into text_cleaned)
       - char_end: int (character offset into text_cleaned)
    """
    if not text or not text.strip():
        return []

    units = split_into_sentence_units(text)
    if not units:
        return []

    chunks = []
    n_units = len(units)
    start_u = 0
    chunk_index = 0
    prev_end_u = -1

    while start_u < n_units:
        end_u = start_u
        accum_words = 0

        while end_u < n_units:
            u_words = units[end_u]["words"]
            if accum_words == 0:
                accum_words += u_words
                end_u += 1
            elif accum_words + u_words <= chunk_size:
                accum_words += u_words
                end_u += 1
            else:
                # Include unit if it brings chunk closer to chunk_size without excessive overshoot (<= 1.15x)
                diff_without = chunk_size - accum_words
                diff_with = (accum_words + u_words) - chunk_size
                if diff_with < diff_without and (accum_words + u_words) <= int(chunk_size * 1.15):
                    accum_words += u_words
                    end_u += 1
                break

        # Ensure strict forward advancement: end_u must exceed prev_end_u
        if end_u <= prev_end_u:
            end_u = min(prev_end_u + 1, n_units)

        c_start = units[start_u]["start"]
        c_end = units[end_u - 1]["end"]
        c_text = text[c_start:c_end]
        c_words = len(c_text.split())

        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_index}",
            "doc_id": str(doc_id),
            "chunk_index": chunk_index,
            "text": c_text,
            "word_count": c_words,
            "char_start": c_start,
            "char_end": c_end
        })
        chunk_index += 1
        prev_end_u = end_u

        if end_u >= n_units:
            break

        # Determine next start unit to achieve ~overlap words
        # Overlap comes from units in range [cand_u, end_u - 1]
        best_next_u = end_u
        best_diff = float("inf")
        overlap_words = 0

        for cand_u in range(end_u - 1, start_u, -1):
            overlap_words += units[cand_u]["words"]
            diff = abs(overlap_words - overlap)
            if diff < best_diff:
                best_diff = diff
                best_next_u = cand_u
            if overlap_words > overlap * 2:
                break

        # Enforce forward progress: start_u must strictly increase
        start_u = max(start_u + 1, best_next_u)

    return chunks


def _process_batch(batch_items: list[tuple[str, str]]) -> tuple[list[dict], list[int]]:
    """Worker function for multiprocessing."""
    batch_chunks = []
    chunk_counts_per_doc = []
    for doc_id, text in batch_items:
        doc_chunks = chunk_document(text, chunk_size=350, overlap=50, doc_id=doc_id)
        batch_chunks.extend(doc_chunks)
        chunk_counts_per_doc.append(len(doc_chunks))
    return batch_chunks, chunk_counts_per_doc


def main():
    print("=" * 80)
    print("VERIRAG CORPUS CHUNKING PIPELINE")
    print("=" * 80)

    start_time = time.time()

    # 1. Load input dataset
    print(f"[1/5] Loading cleaned dataset from {INPUT_PARQUET}...")
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_PARQUET}. Run data/apply_cleaning_v4.py first.")

    df_raw = pd.read_parquet(INPUT_PARQUET, columns=["id", "text_cleaned"])
    total_docs = len(df_raw)
    print(f"      Loaded {total_docs:,} judgments into memory in {time.time() - start_time:.2f}s.")

    # 2. Parallel chunking
    num_workers = min(8, mp.cpu_count())
    batch_size = 500
    print(f"[2/5] Chunking {total_docs:,} documents using {num_workers} parallel workers...")

    items = [(str(df_raw.iloc[i]["id"]), str(df_raw.iloc[i]["text_cleaned"])) for i in range(total_docs)]
    batches = [items[i:i + batch_size] for i in range(0, total_docs, batch_size)]

    all_chunks = []
    chunks_per_doc_list = []

    with mp.Pool(processes=num_workers) as pool:
        for batch_chunks, counts in tqdm(pool.imap(_process_batch, batches), total=len(batches), desc="Chunking"):
            all_chunks.extend(batch_chunks)
            chunks_per_doc_list.extend(counts)

    chunking_duration = time.time() - start_time
    total_chunks = len(all_chunks)
    print(f"      Completed chunking in {chunking_duration:.2f}s ({total_docs / chunking_duration:.1f} docs/sec).")

    # 3. Compute corpus-wide statistics
    print("[3/5] Computing corpus-wide chunk statistics...")
    s_doc_chunks = pd.Series(chunks_per_doc_list)
    word_counts = pd.Series([c["word_count"] for c in all_chunks])

    avg_chunks_per_doc = s_doc_chunks.mean()
    min_chunks_per_doc = int(s_doc_chunks.min())
    max_chunks_per_doc = int(s_doc_chunks.max())
    single_chunk_docs = int((s_doc_chunks == 1).sum())

    avg_words_per_chunk = word_counts.mean()
    min_words_per_chunk = int(word_counts.min())
    max_words_per_chunk = int(word_counts.max())

    # 4. Save to parquet
    print(f"[4/5] Saving {total_chunks:,} chunks to {OUTPUT_PARQUET}...")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_chunks = pd.DataFrame(all_chunks)
    df_chunks.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"      Saved {len(df_chunks):,} rows ({file_size_mb:.2f} MB on disk).")

    # Round-trip QA check
    print("      Verifying round-trip parquet integrity...")
    df_verify = pd.read_parquet(OUTPUT_PARQUET)
    assert len(df_verify) == total_chunks, "Row count mismatch in parquet verification!"
    expected_cols = ["chunk_id", "doc_id", "chunk_index", "text", "word_count", "char_start", "char_end"]
    assert list(df_verify.columns) == expected_cols, f"Column mismatch! Expected {expected_cols}, got {list(df_verify.columns)}"
    print("      Parquet round-trip verification passed.")

    # 5. Display metrics and sample multi-chunk document inspection
    print("\n" + "=" * 80)
    print("CORPUS CHUNKING SUMMARY METRICS")
    print("=" * 80)
    print(f"Total documents processed       : {total_docs:,}")
    print(f"Total chunks produced           : {total_chunks:,}")
    print(f"Chunks per document (avg)       : {avg_chunks_per_doc:.2f}")
    print(f"Chunks per document (min / max) : {min_chunks_per_doc:,} / {max_chunks_per_doc:,}")
    print(f"Single-chunk documents          : {single_chunk_docs:,} ({single_chunk_docs / total_docs * 100:.2f}%)")
    print(f"Words per chunk (avg)           : {avg_words_per_chunk:.2f}")
    print(f"Words per chunk (min / max)     : {min_words_per_chunk:,} / {max_words_per_chunk:,}")
    print(f"Output parquet file             : {OUTPUT_PARQUET}")
    print(f"Output parquet size             : {file_size_mb:.2f} MB")
    print("=" * 80)

    # Find a multi-chunk document to display 3 example chunks in order
    print("\n" + "=" * 80)
    print("VISUAL VERIFICATION: 3 CONSECUTIVE CHUNKS FROM A MULTI-CHUNK DOCUMENT")
    print("=" * 80)

    sample_doc_id = df_raw.iloc[0]["id"]
    doc_chunks = [c for c in all_chunks if c["doc_id"] == str(sample_doc_id)][:3]

    for idx, c in enumerate(doc_chunks):
        print(f"\n--- CHUNK {c['chunk_index']} ({c['chunk_id']}) ---")
        print(f"Word Count : {c['word_count']} words | Character Span : [{c['char_start']}:{c['char_end']}]")
        print(f"Full Text  :\n\"{c['text']}\"")
        if idx < len(doc_chunks) - 1:
            next_c = doc_chunks[idx + 1]
            overlap_start = next_c["char_start"]
            overlap_end = min(c["char_end"], next_c["char_end"])
            if overlap_end > overlap_start:
                overlap_text = df_raw.iloc[0]["text_cleaned"][overlap_start:overlap_end]
                overlap_words = len(overlap_text.split())
                print(f"\n>>> OVERLAP with Chunk {next_c['chunk_index']} ({overlap_words} words, chars [{overlap_start}:{overlap_end}]):")
                print(f">>> \"{overlap_text}\"")

    print("\nChunking pipeline complete.")


if __name__ == "__main__":
    main()
