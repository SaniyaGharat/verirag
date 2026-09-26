"""
Corpus Inspection and Dataset Role Verification Script
======================================================
1. RETRIEVAL CORPUS:
   - Identifier: "jayadityagandham9/ILDC_35k_COMPLETE"
   - Role: Main retrieval corpus for indexing and dense/sparse retrieval in VeriRAG (38,904 cases).

2. EVAL / COMPARISON SET:
   - Identifier: "anuragiiser/ILDC_expert"
   - Role: EVAL/COMPARISON SET ONLY — 54 examples, NOT the retrieval corpus.
     Contains human gold-standard explanations and decisions for validation.

3. PHASE 6 BENCHMARK (Deferred):
   - Identifier: "Exploration-Lab/IL-TUR" (task: "CJPE")
   - Role: Official gated benchmark required for Phase 6 evaluation.
"""

import os
import random
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

# Suppress Windows symlink warnings from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Find and load .env from project root or current working dir
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def check_hf_authentication():
    """Check if huggingface_hub authentication token is configured in .env."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    print("=" * 75)
    print("HUGGING FACE AUTHENTICATION CHECK (For Phase 6 Benchmark: Exploration-Lab/IL-TUR)")
    print("=" * 75)
    
    if hf_token and len(hf_token.strip()) > 0:
        masked_token = hf_token[:4] + "..." + hf_token[-4:] if len(hf_token) > 8 else "***"
        print(f"Status: AUTHENTICATED (HF_TOKEN detected: {masked_token})")
        print("Note: Access to 'Exploration-Lab/IL-TUR' is enabled for downstream evaluation.")
    else:
        print("Status: NOT AUTHENTICATED")
        print("\nINSTRUCTIONS TO AUTHENTICATE FOR PHASE 6 EVAL:")
        print("1. Go to https://huggingface.co/settings/tokens and generate an access token.")
        print("2. Create a .env file in the project root directory (verirag/.env) with:")
        print("     HF_TOKEN=your_token_here")
        print("3. Visit https://huggingface.co/datasets/Exploration-Lab/IL-TUR and accept")
        print("   the dataset terms of use on the Hugging Face webpage.")
        print("\n* Note: Authentication is NOT required right now for indexing or decomposition;")
        print("  it will be required later during Phase 6 evaluation benchmarks.")
    print("=" * 75)
    print()


def inspect_corpus():
    # Set seed for reproducible sampling
    random.seed(42)

    # ---------------------------------------------------------
    # PART 1: RETRIEVAL CORPUS (jayadityagandham9/ILDC_35k_COMPLETE)
    # ---------------------------------------------------------
    print("=" * 75)
    print("PART 1: LOADING MAIN RETRIEVAL CORPUS (jayadityagandham9/ILDC_35k_COMPLETE)")
    print("=" * 75)
    print("Loading full Indian Legal Documents Corpus for retriever indexing...")
    corpus_dataset = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE")

    print("\n--- Corpus Splits & Example Counts ---")
    for split_name, split_data in corpus_dataset.items():
        print(f"  Split: {split_name:<10} | Total Examples: {len(split_data):,}")

    corpus_split = corpus_dataset["train"]
    first_corpus_ex = corpus_split[0]

    print("\n--- Fields in First Example ---")
    for key, val in first_corpus_ex.items():
        val_type = type(val).__name__
        if isinstance(val, str):
            val_len = f"length {len(val):,} chars"
        elif isinstance(val, (list, dict)):
            val_len = f"length {len(val):,} items"
        else:
            val_len = str(val)
        print(f"  - {key:<12} : type={val_type:<8} | value/preview: {val_len}")

    # Random sample stats across 200 examples
    num_sample = min(200, len(corpus_split))
    sample_indices = random.sample(range(len(corpus_split)), num_sample)
    sampled_lengths = [len(str(corpus_split[i]["text"])) for i in sample_indices]

    avg_len = sum(sampled_lengths) / len(sampled_lengths)
    min_len = min(sampled_lengths)
    max_len = max(sampled_lengths)

    print("\n--- Basic Corpus Statistics (Sample of 200 Examples) ---")
    print(f"  Sample Size         : {num_sample} randomly drawn judgments")
    print(f"  Average Char Length : {avg_len:,.1f} characters")
    print(f"  Minimum Char Length : {min_len:,} characters")
    print(f"  Maximum Char Length : {max_len:,} characters")

    # Full text of one random example's main case-text field (not truncated)
    random_idx = random.choice(range(len(corpus_split)))
    random_example = corpus_split[random_idx]
    case_id = random_example.get("id", f"index_{random_idx}")
    case_label = random_example.get("label", "N/A")
    full_case_text = str(random_example["text"])

    print("\n" + "=" * 75)
    print(f"FULL TEXT OF RANDOM CORPUS EXAMPLE (Index: {random_idx}, ID: {case_id}, Label: {case_label})")
    print(f"Total Character Count: {len(full_case_text):,} characters")
    print("=" * 75)
    print(full_case_text)
    print("=" * 75)
    print("END OF FULL TEXT PREVIEW")
    print("=" * 75)

    # ---------------------------------------------------------
    # PART 2: EVAL/COMPARISON SET (anuragiiser/ILDC_expert)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("PART 2: LOADING EVAL/COMPARISON SET (anuragiiser/ILDC_expert)")
    print("=" * 75)
    print(">>> EVAL/COMPARISON SET ONLY -- 54 examples, NOT the retrieval corpus. <<<")
    print("This subset provides legal expert gold-standard reasoning and ground-truth decisions")
    print("used strictly for verifier benchmarking and qualitative comparison.\n")

    eval_dataset = load_dataset("anuragiiser/ILDC_expert")
    print("--- Eval Set Splits & Example Counts ---")
    for split_name, split_data in eval_dataset.items():
        print(f"  Split: {split_name:<10} | Total Examples: {len(split_data):,}")

    eval_split = eval_dataset["train"]
    first_eval_ex = eval_split[0]
    print("\n--- Fields in First Eval Example ---")
    for key, val in first_eval_ex.items():
        val_type = type(val).__name__
        val_preview = f"length {len(val):,} chars" if isinstance(val, str) else str(val)
        print(f"  - {key:<25} : type={val_type:<8} | value/preview: {val_preview}")

    # ---------------------------------------------------------
    # PART 3: FINAL SUMMARY BLOCK
    # ---------------------------------------------------------
    print("\n" + "#" * 75)
    print("FINAL SUMMARY: DATASET ROLES IN VERIRAG")
    print("#" * 75)
    print(f"1. RETRIEVAL CORPUS (For Indexing & Chunk Retrieval):")
    print(f"   - Dataset ID : jayadityagandham9/ILDC_35k_COMPLETE")
    print(f"   - Role       : CORPUS (indexed via FAISS + Dense Embeddings)")
    print(f"   - Count      : {len(corpus_split):,} judgments")
    print()
    print(f"2. EVALUATION & BENCHMARKING SET (For Validation & Quality Control):")
    print(f"   - Dataset ID : anuragiiser/ILDC_expert")
    print(f"   - Role       : EVAL/COMPARISON SET ONLY (NOT for indexing)")
    print(f"   - Count      : {len(eval_split):,} expert-annotated judgments")
    print()
    print(f"3. CANONICAL BENCHMARK (Phase 6 Final Evaluation):")
    print(f"   - Dataset ID : Exploration-Lab/IL-TUR [task: CJPE]")
    print(f"   - Role       : Phase 6 Benchmark Evaluation (Requires HF_TOKEN + Terms)")
    print("#" * 75)
    print()


if __name__ == "__main__":
    check_hf_authentication()
    inspect_corpus()
