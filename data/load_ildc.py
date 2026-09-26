"""
ILDC (Indian Legal Documents Corpus) Dataset Loader
====================================================
Exact Hugging Face Dataset Identifiers:
1. Canonical Benchmark Repository:
   - Identifier: "Exploration-Lab/IL-TUR"
   - Configuration / Subset: "CJPE" (Court Judgment Prediction and Explanation)
   - Status: Gated repository on Hugging Face (requires HF authentication / accepted terms)
   - Reference: "IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning" (Paul et al., 2022)
     and "ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation" (Malik et al., 2021)

2. Open Access Community Mirrors / Subsets:
   - Identifier: "anuragiiser/ILDC_expert" (Gold-standard expert evaluation subset of ILDC)
   - Identifier: "jayadityagandham9/ILDC_35k_COMPLETE" (Complete 35k case corpus of ILDC)

This script attempts the canonical 'Exploration-Lab/IL-TUR' ('CJPE') first. If access is restricted/gated,
it gracefully falls back to the open accessible ILDC dataset 'anuragiiser/ILDC_expert'.
"""

import os
import sys

# Suppress Windows symlink warning from huggingface_hub for clean output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables (e.g., HF_TOKEN if available)
load_dotenv()


def load_ildc_dataset():
    # Candidates ordered by priority:
    # 1. Canonical gated benchmark ("Exploration-Lab/IL-TUR", "CJPE")
    # 2. Canonical lowercase variant ("Exploration-Lab/IL-TUR", "cjpe")
    # 3. Direct name ("ildc")
    # 4. Open-access ILDC expert evaluation corpus ("anuragiiser/ILDC_expert")
    candidates = [
        ("Exploration-Lab/IL-TUR", "CJPE", "Canonical benchmark repo (gated)"),
        ("Exploration-Lab/IL-TUR", "cjpe", "Canonical benchmark repo lowercase config"),
        ("ildc", None, "Direct name lookup"),
        ("anuragiiser/ILDC_expert", None, "Open-access ILDC expert evaluation subset"),
    ]

    dataset = None
    chosen_meta = None

    print("-" * 65)
    print("Searching and loading ILDC (Indian Legal Documents Corpus)...")
    print("-" * 65)

    for repo_id, config_name, desc in candidates:
        try:
            print(f"Trying identifier: '{repo_id}' (config: {config_name}) [{desc}]...")
            if config_name:
                dataset = load_dataset(repo_id, config_name)
            else:
                dataset = load_dataset(repo_id)
            chosen_meta = (repo_id, config_name, desc)
            print(f"-> SUCCESS: Successfully loaded '{repo_id}' (config: {config_name})\n")
            break
        except Exception as err:
            err_msg = str(err).split("\n")[0]
            print(f"-> FAILED: {err_msg}\n")

    if dataset is None:
        print("ERROR: Failed to load ILDC dataset from any candidate identifier.")
        sys.exit(1)

    repo_id, config_name, desc = chosen_meta

    # 1. Print dataset splits and number of examples per split
    print("=" * 65)
    print("ILDC DATASET SPLITS & STATISTICS")
    print("=" * 65)
    print(f"Hugging Face Identifier : {repo_id}")
    if config_name:
        print(f"Configuration / Subset  : {config_name}")
    print(f"Description             : {desc}")
    print(f"Available Splits        : {list(dataset.keys())}")
    print("-" * 65)
    for split_name, split_data in dataset.items():
        print(f"  Split: {split_name:<10} | Number of Examples: {len(split_data):,}")
    print("=" * 65)

    # 2. Inspect first split (usually 'train' or first available)
    first_split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    first_split = dataset[first_split_name]
    first_example = first_split[0]

    print(f"\nFIRST EXAMPLE DETAILS (Split: '{first_split_name}')")
    print("=" * 65)
    print("Fields available in first example:")
    for key, val in first_example.items():
        val_type = type(val).__name__
        val_preview = f"length {len(val)}" if isinstance(val, (str, list, dict)) else str(val)
        print(f"  - {key:<25} : (type: {val_type:<6}, preview: {val_preview})")

    # 3. Print truncated preview of the primary text/case description field
    candidate_text_fields = [
        "Case Description",
        "text",
        "document",
        "case_text",
        "Official Reasoning",
        "premise",
        "input",
    ]
    text_field = None
    for field in candidate_text_fields:
        if field in first_example and first_example[field]:
            text_field = field
            break

    if text_field:
        full_text = str(first_example[text_field]).strip()
        preview_len = 500
        truncated_text = full_text[:preview_len] + ("..." if len(full_text) > preview_len else "")
        print("\n" + "-" * 65)
        print(f"PREVIEW OF PRIMARY TEXT FIELD: '{text_field}' (First {preview_len} chars)")
        print("-" * 65)
        print(truncated_text)
        print("-" * 65)
        print(f"Total character count: {len(full_text):,} characters")

    # Also show decision/label if present
    for label_field in ["Official Decision", "label", "decision"]:
        if label_field in first_example:
            print(f"Outcome Label ('{label_field}'): {first_example[label_field]}")

    print("=" * 65)
    print("ILDC dataset loaded and inspected successfully.")
    return dataset


if __name__ == "__main__":
    load_ildc_dataset()
