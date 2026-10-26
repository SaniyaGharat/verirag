"""
VeriRAG Chunk Embedding Pipeline
=================================
Generates dense vector embeddings for all passages in chunks.parquet using
the configured SentenceTransformer model (default: sentence-transformers/all-MiniLM-L6-v2).

Outputs:
- data/processed/chunk_embeddings.npy
- data/processed/chunk_metadata.parquet
"""

import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def embed_chunks(
    chunks_path: Path,
    output_embeddings_path: Path,
    output_metadata_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = None,
    sample_size: int = None
):
    start_time = time.time()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks parquet not found at: {chunks_path}. Run retriever/chunk_corpus.py first.")

    print(f"[1/4] Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    total_chunks = len(df)
    print(f"      Loaded {total_chunks:,} chunks.")

    if sample_size and sample_size < total_chunks:
        print(f"      Subsampling {sample_size} chunks for testing...")
        df = df.iloc[:sample_size].copy()
        total_chunks = len(df)

    # Resolve compute device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/4] Initializing SentenceTransformer model '{model_name}' on device: {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Extract text list
    texts = df["text"].tolist()

    print(f"[3/4] Generating normalized embeddings for {total_chunks:,} chunks (batch_size: {batch_size})...")
    embed_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    embed_duration = time.time() - embed_start
    print(f"      Embedding completed in {embed_duration:.2f}s ({total_chunks / embed_duration:.1f} chunks/sec).")
    print(f"      Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Integrity verification
    assert embeddings.shape[0] == total_chunks, "Embedding count mismatch!"
    assert not np.isnan(embeddings).any(), "NaN found in computed embeddings!"

    # Save embeddings and metadata
    print(f"[4/4] Saving embeddings and metadata...")
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings_path, embeddings)
    embed_size_mb = output_embeddings_path.stat().st_size / (1024 * 1024)
    print(f"      Saved embeddings to {output_embeddings_path} ({embed_size_mb:.2f} MB).")

    # Save lightweight metadata (without large full text if desired, or with full text for quick retrieval)
    metadata_cols = [c for c in ["chunk_id", "doc_id", "chunk_index", "word_count", "char_start", "char_end", "text"] if c in df.columns]
    df[metadata_cols].to_parquet(output_metadata_path, index=False, engine="pyarrow", compression="snappy")
    meta_size_mb = output_metadata_path.stat().st_size / (1024 * 1024)
    print(f"      Saved metadata to {output_metadata_path} ({meta_size_mb:.2f} MB).")

    total_duration = time.time() - start_time
    print(f"\nPipeline finished in {total_duration:.2f}s.")
    return embeddings, df[metadata_cols]


def main():
    parser = argparse.ArgumentParser(description="Embed chunks for VeriRAG retrieval index.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--chunks", type=str, default=None, help="Path to chunks.parquet (overrides config)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--test", action="store_true", help="Run test on first 100 chunks")
    parser.add_argument("--sample", type=int, default=None, help="Embed only first N chunks")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    chunks_path = Path(args.chunks) if args.chunks else PROJECT_ROOT / cfg["corpus"]["chunks_path"]
    model_name = cfg["retriever"]["embedding_model"]

    output_embeddings_path = PROJECT_ROOT / "data" / "processed" / "chunk_embeddings.npy"
    output_metadata_path = PROJECT_ROOT / cfg["retriever"]["metadata_path"]

    sample_size = 100 if args.test else args.sample

    embed_chunks(
        chunks_path=chunks_path,
        output_embeddings_path=output_embeddings_path,
        output_metadata_path=output_metadata_path,
        model_name=model_name,
        batch_size=args.batch_size,
        device=args.device,
        sample_size=sample_size
    )


if __name__ == "__main__":
    main()
