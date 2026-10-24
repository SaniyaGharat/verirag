"""
Negation Repair Scoring Engine
==============================
Applies deterministic frozen rules and Masked Language Model (InLegalBERT)
scoring to standalone 'number' occurrences in legal text.

Usage:
    python negation_repair.py score --input number_occurrences.parquet \\
                                    --output number_scores.parquet \\
                                    --window 32 \\
                                    --batch-size 64 \\
                                    [--limit N]
"""

import os
import sys
import re
import math
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Deterministic Frozen Rules
# -----------------------------------------------------------------------------
ROMAN_NUMERALS_PATTERN = r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)"
RULE_KEEP_NUMBER_RE = re.compile(
    r"^\s*(?:\d+\b|" + ROMAN_NUMERALS_PATTERN + r"\b|\.\s*\d+)",
    re.IGNORECASE
)
RULE_NON_RE = re.compile(r"^\s*[-–—]\s*[a-zA-Z]")


def rule_label(left_ctx: str, right_ctx: str):
    """
    Evaluates frozen deterministic rules on context around standalone 'number'.
    
    Rules:
    - next token is a digit string, a roman numeral (i..x), or '.' followed by a digit -> 'KEEP_NUMBER'
    - next non-space character is a hyphen followed by a letter (e.g. 'number-teaching') -> 'non'
    - anything else -> None (routed to MLM)
    """
    if not right_ctx:
        return None
    # Check hyphen followed by letter
    if RULE_NON_RE.match(right_ctx):
        return "non"
    # Check digit, roman numeral, or . followed by digit
    if RULE_KEEP_NUMBER_RE.match(right_ctx):
        return "KEEP_NUMBER"
    return None


# -----------------------------------------------------------------------------
# 2. MLM Scoring Pipeline
# -----------------------------------------------------------------------------
CANDIDATES = ["not", "nor", "now", "non", "no", "number"]


def run_scoring(input_path: str, output_path: str, window: int = 32, batch_size: int = 64, limit: int = None, shard_size: int = 5000):
    start_total_time = time.time()
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading occurrences from {input_file}...")
    df = pd.read_parquet(input_file)
    total_raw = len(df)
    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()
        print(f"Limited run: processing {len(df):,} of {total_raw:,} occurrences.")
    else:
        print(f"Full run: processing {len(df):,} occurrences.")

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_fp16 = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)} (FP16 enabled)")
    else:
        device = torch.device("cpu")
        use_fp16 = False
        print("Using CPU device (FP32 mode)")

    model_name = "law-ai/InLegalBERT"
    print(f"Loading model & tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    if use_fp16:
        model = model.half()
    model = model.to(device)
    model.eval()

    cand_ids = [tokenizer.convert_tokens_to_ids(c) for c in CANDIDATES]
    cand_tensor = torch.tensor(cand_ids, dtype=torch.long, device=device)

    # Process by shards of shard_size (5,000 items)
    num_shards = math.ceil(len(df) / shard_size)
    print(f"Dataset partitioned into {num_shards} shard(s) of up to {shard_size:,} items.")

    completed_shards = []
    mlm_items_processed = 0
    mlm_time_elapsed = 0.0

    for shard_idx in range(num_shards):
        shard_path = shards_dir / f"shard_{shard_idx:05d}.parquet"
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(df))
        shard_df = df.iloc[start_idx:end_idx].copy()

        if shard_path.exists():
            print(f"Shard {shard_idx + 1}/{num_shards} ({shard_path.name}) already exists. Skipping.")
            completed_shards.append(pd.read_parquet(shard_path))
            continue

        print(f"\nProcessing Shard {shard_idx + 1}/{num_shards} [indices {start_idx:,}..{end_idx:,}]...")
        shard_t0 = time.time()

        # Step 1: Apply rule_label
        rules = []
        for _, row in shard_df.iterrows():
            rules.append(rule_label(row["left_ctx"], row["right_ctx"]))
        shard_df["rule"] = rules

        # Prepare storage arrays
        n_shard = len(shard_df)
        prob_matrix = np.zeros((n_shard, len(CANDIDATES)), dtype=np.float32)

        # Separate items by rule vs MLM
        mlm_needed = []
        for i, r in enumerate(rules):
            if r == "KEEP_NUMBER":
                # number gets 1.0, others 0.0
                prob_matrix[i, CANDIDATES.index("number")] = 1.0
            elif r == "non":
                # non gets 1.0, others 0.0
                prob_matrix[i, CANDIDATES.index("non")] = 1.0
            else:
                mlm_needed.append(i)

        print(f"  Rules resolved: {n_shard - len(mlm_needed):,} items ({((n_shard - len(mlm_needed))/n_shard)*100:.1f}%).")
        print(f"  MLM scoring required: {len(mlm_needed):,} items.")

        if mlm_needed:
            t_mlm_start = time.time()
            # Tokenize MLM candidates
            mlm_data = []
            for local_idx in mlm_needed:
                row = shard_df.iloc[local_idx]
                left_toks = tokenizer.tokenize(row["left_ctx"])[-window:]
                right_toks = tokenizer.tokenize(row["right_ctx"])[:window]
                toks = ["[CLS]"] + left_toks + ["[MASK]"] + right_toks + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(toks)
                mask_pos = len(left_toks) + 1
                mlm_data.append({
                    "local_idx": local_idx,
                    "input_ids": input_ids,
                    "mask_pos": mask_pos,
                    "length": len(input_ids)
                })

            # Sort by length to minimize padding
            mlm_data.sort(key=lambda x: x["length"])

            # Batch inference
            for b_start in range(0, len(mlm_data), batch_size):
                b_items = mlm_data[b_start:b_start + batch_size]
                b_max_len = max(x["length"] for x in b_items)

                padded_ids = [x["input_ids"] + [tokenizer.pad_token_id] * (b_max_len - x["length"]) for x in b_items]
                attn_masks = [[1] * x["length"] + [0] * (b_max_len - x["length"]) for x in b_items]
                mask_positions = [x["mask_pos"] for x in b_items]
                local_indices = [x["local_idx"] for x in b_items]

                inp = torch.tensor(padded_ids, dtype=torch.long, device=device)
                mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

                with torch.inference_mode():
                    # Forward pass through BERT backbone
                    bert_out = model.bert(inp, attention_mask=mask)
                    sequence_output = bert_out[0]  # [B, L, H]

                    # Extract mask vector only: [B, H]
                    batch_idx_tensor = torch.arange(len(b_items), device=device)
                    mask_pos_tensor = torch.tensor(mask_positions, dtype=torch.long, device=device)
                    mask_vectors = sequence_output[batch_idx_tensor, mask_pos_tensor]

                    # Compute MLM predictions head only at mask positions: [B, V]
                    pred_logits = model.cls.predictions(mask_vectors)

                    # Extract only the 6 candidate logits: [B, 6]
                    cand_logits = pred_logits[:, cand_tensor]
                    cand_probs = torch.softmax(cand_logits, dim=-1).cpu().numpy()

                for bi, loc_idx in enumerate(local_indices):
                    prob_matrix[loc_idx] = cand_probs[bi]

            t_mlm_end = time.time()
            elapsed_mlm = t_mlm_end - t_mlm_start
            mlm_items_processed += len(mlm_needed)
            mlm_time_elapsed += elapsed_mlm
            print(f"  MLM inference completed: {len(mlm_needed):,} items in {elapsed_mlm:.2f}s ({len(mlm_needed)/elapsed_mlm:.2f} it/s).")

        # Compile shard results
        top1_indices = np.argmax(prob_matrix, axis=1)
        top1_labels = [CANDIDATES[i] for i in top1_indices]
        top1_probs = prob_matrix[np.arange(n_shard), top1_indices]

        shard_out = pd.DataFrame({
            "occ_id": shard_df["occ_id"].values,
            "rule": shard_df["rule"].values,
            "p_not": prob_matrix[:, 0],
            "p_nor": prob_matrix[:, 1],
            "p_now": prob_matrix[:, 2],
            "p_non": prob_matrix[:, 3],
            "p_no": prob_matrix[:, 4],
            "p_number": prob_matrix[:, 5],
            "top1": top1_labels,
            "top1_p": top1_probs
        })

        shard_out.to_parquet(shard_path, index=False)
        completed_shards.append(shard_out)
        shard_time = time.time() - shard_t0
        print(f"Saved {shard_path} in {shard_time:.2f}s.")

    # Combine all shards
    print("\nConcatenating shards into final output...")
    final_df = pd.concat(completed_shards, ignore_index=True)
    final_df.sort_values(by="occ_id", inplace=True)
    final_df.to_parquet(output_file, index=False)
    total_elapsed = time.time() - start_total_time
    throughput_total = len(final_df) / total_elapsed

    print(f"Saved final scores to {output_file} ({len(final_df):,} rows).")
    print(f"Total processing time: {total_elapsed:.2f}s ({throughput_total:.2f} total items/sec).")

    if mlm_items_processed > 0:
        mlm_throughput = mlm_items_processed / mlm_time_elapsed
        print(f"MLM-only throughput: {mlm_throughput:.2f} items/sec.")

    return final_df, throughput_total


# -----------------------------------------------------------------------------
# 3. CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Negation Repair Scoring Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score occurrences using rules and MLM")
    score_parser.add_argument("--input", type=str, required=True, help="Path to number_occurrences.parquet")
    score_parser.add_argument("--output", type=str, required=True, help="Path to output number_scores.parquet")
    score_parser.add_argument("--window", type=int, default=32, help="Wordpiece window on each side (default: 32)")
    score_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for MLM (default: 64)")
    score_parser.add_argument("--limit", type=int, default=None, help="Limit number of occurrences to score")
    score_parser.add_argument("--shard-size", type=int, default=5000, help="Shard size (default: 5000)")

    args = parser.parse_args()

    if args.command == "score":
        run_scoring(
            input_path=args.input,
            output_path=args.output,
            window=args.window,
            batch_size=args.batch_size,
            limit=args.limit,
            shard_size=args.shard_size
        )


if __name__ == "__main__":
    main()
