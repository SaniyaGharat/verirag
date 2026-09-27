"""
ILDC (Indian Legal Documents Corpus) Dataset Loader
====================================================
This module handles loading both the primary retrieval corpus and the evaluation sets
for the VeriRAG pipeline.

Dataset Roles:
1. PRIMARY RETRIEVAL CORPUS:
   - Identifier: "jayadityagandham9/ILDC_35k_COMPLETE"
   - Role: CORPUS for retriever chunking and indexing (~38.9k Supreme Court judgments).

2. EVAL / COMPARISON BENCHMARK SET:
   - Identifier: "anuragiiser/ILDC_expert"
   - Role: EVAL/COMPARISON SET ONLY — 54 examples, NOT the retrieval corpus.
     Used strictly for verifier evaluation against gold-standard human legal explanations.

3. CANONICAL BENCHMARK (Phase 6):
   - Identifier: "Exploration-Lab/IL-TUR" (task: "CJPE")
   - Role: Canonical gated benchmark for final Phase 6 evaluation.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

# Suppress Windows symlink warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables (.env in project root or current working dir)
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")


def check_hf_authentication():
    """Verify Hugging Face authentication for gated benchmark datasets."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token or len(token.strip()) == 0:
        print("\n" + "!" * 70)
        print("NOTICE: Hugging Face Authentication Token Not Detected in .env")
        print("!" * 70)
        print("To enable access to the canonical benchmark 'Exploration-Lab/IL-TUR':")
        print("1. Visit https://huggingface.co/settings/tokens and generate an access token.")
        print("2. Add HF_TOKEN=<your_token> to a '.env' file in the project root.")
        print("3. Accept dataset terms at: https://huggingface.co/datasets/Exploration-Lab/IL-TUR")
        print("Note: This is required for Phase 6 evaluation, not current development.")
        print("!" * 70 + "\n")
    else:
        print("[Auth] Hugging Face token detected in environment.")


def load_retrieval_corpus():
    """Loads the main ILDC retrieval corpus for indexing."""
    print("=" * 70)
    print("LOADING PRIMARY RETRIEVAL CORPUS (jayadityagandham9/ILDC_35k_COMPLETE)")
    print("=" * 70)
    corpus = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE")
    print(f"Total indexed judgments available: {len(corpus['train']):,}")
    return corpus


def load_cleaned_corpus(version: str = "v4"):
    """Loads the preprocessed and cleaned ILDC corpus from parquet."""
    import pandas as pd
    processed_dir = project_root / "data" / "processed"
    parquet_path = processed_dir / f"ildc_cleaned_{version}.parquet"
    if not parquet_path.exists():
        parquet_path = processed_dir / "ildc_cleaned_v4.parquet"
    if not parquet_path.exists():
        parquet_path = processed_dir / "ildc_cleaned_v3.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cleaned corpus not found in {processed_dir}. Run apply_cleaning_v4.py first.")
    print("=" * 70)
    print(f"LOADING CLEANED RETRIEVAL CORPUS ({parquet_path.name})")
    print("=" * 70)
    df = pd.read_parquet(parquet_path)
    print(f"Total cleaned judgments available: {len(df):,}")
    return df


def load_eval_set():
    """Loads the 54-case expert evaluation subset."""
    print("=" * 70)
    print("LOADING EVAL/COMPARISON SET (anuragiiser/ILDC_expert)")
    print("=" * 70)
    print(">>> EVAL/COMPARISON SET ONLY -- 54 examples, NOT the retrieval corpus. <<<")
    eval_set = load_dataset("anuragiiser/ILDC_expert")
    print(f"Total expert-annotated evaluation cases: {len(eval_set['train']):,}")
    return eval_set


if __name__ == "__main__":
    check_hf_authentication()
    corpus = load_retrieval_corpus()
    eval_ds = load_eval_set()
    print("\n" + "=" * 70)
    print("SUMMARY OF DATASETS LOADED:")
    print(f" - Primary Retrieval Corpus : {len(corpus['train']):,} examples")
    print(f" - Expert Evaluation Set    : {len(eval_ds['train']):,} examples")
    print("=" * 70)
