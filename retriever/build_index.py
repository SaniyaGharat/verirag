"""
VeriRAG Index Builder (FAISS Dense Index + BM25 Sparse Index)
=============================================================
Builds and saves:
1. FAISS Flat-IP index (Inner Product / Cosine Similarity on L2-normalized embeddings)
2. BM25Okapi sparse lexical index (for baseline comparison and hybrid search)

Outputs:
- data/processed/faiss_index.bin
- data/processed/bm25_index.pkl
- data/processed/chunk_metadata.parquet
"""

import sys
import time
import pickle
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple, fast whitespace-and-alphanumeric tokenizer for legal passage BM25."""
    return [w.lower() for w in text.split() if w.isalnum() or len(w) > 1]


def build_faiss_index(embeddings: np.ndarray, index_output_path: Path) -> faiss.IndexFlatIP:
    """Builds and writes a FAISS Flat-IP index from normalized embeddings."""
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dimension={dim}, entries={len(embeddings):,})...")
    index = faiss.IndexFlatIP(dim)
    
    # Ensure float32 contiguous array
    embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(embeddings_f32)
    
    assert index.ntotal == len(embeddings), f"Index count {index.ntotal} != embedding count {len(embeddings)}"
    
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_output_path))
    file_size_mb = index_output_path.stat().st_size / (1024 * 1024)
    print(f"Saved FAISS index to {index_output_path} ({file_size_mb:.2f} MB, total vectors: {index.ntotal:,}).")
    return index


def build_bm25_index(texts: list[str], chunk_ids: list[str], output_bm25_path: Path) -> BM25Okapi:
    """Builds and pickles BM25Okapi index."""
    print(f"Tokenizing {len(texts):,} passages for BM25...")
    tokenized_corpus = [tokenize_for_bm25(t) for t in tqdm(texts, desc="Tokenizing BM25")]
    
    print("Fitting BM25Okapi model...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "corpus_size": len(texts)
    }
    
    output_bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_bm25_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    file_size_mb = output_bm25_path.stat().st_size / (1024 * 1024)
    print(f"Saved BM25 index to {output_bm25_path} ({file_size_mb:.2f} MB).")
    return bm25


def main():
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indices for VeriRAG.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to chunk_embeddings.npy")
    parser.add_argument("--metadata", type=str, default=None, help="Path to chunk_metadata.parquet")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index building")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    embeddings_path = Path(args.embeddings) if args.embeddings else PROJECT_ROOT / "data" / "processed" / "chunk_embeddings.npy"
    metadata_path = Path(args.metadata) if args.metadata else PROJECT_ROOT / cfg["retriever"]["metadata_path"]
    faiss_output_path = PROJECT_ROOT / cfg["retriever"]["index_path"]
    bm25_output_path = PROJECT_ROOT / cfg["retriever"]["bm25_index_path"]

    print("=" * 80)
    print("VERIRAG INDEX BUILDER (FAISS + BM25)")
    print("=" * 80)

    # 1. Load embeddings and metadata
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}. Run retriever/embed_chunks.py first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run retriever/embed_chunks.py first.")

    print(f"[1/3] Loading embeddings and metadata...")
    embeddings = np.load(embeddings_path)
    df_meta = pd.read_parquet(metadata_path)
    assert len(embeddings) == len(df_meta), f"Mismatch between embeddings ({len(embeddings)}) and metadata ({len(df_meta)})"
    print(f"      Loaded {len(embeddings):,} embeddings (dimension: {embeddings.shape[1]}).")

    # 2. Build FAISS index
    print(f"[2/3] Building FAISS Dense Index...")
    t0 = time.time()
    faiss_index = build_faiss_index(embeddings, faiss_output_path)
    print(f"      FAISS index build completed in {time.time() - t0:.2f}s.")

    # 3. Build BM25 index
    if not args.skip_bm25:
        print(f"[3/3] Building BM25 Sparse Index...")
        t0 = time.time()
        build_bm25_index(df_meta["text"].tolist(), df_meta["chunk_id"].tolist(), bm25_output_path)
        print(f"      BM25 index build completed in {time.time() - t0:.2f}s.")
    else:
        print("[3/3] Skipping BM25 index as requested.")

    # 4. Quick verification test
    print("\n" + "=" * 80)
    print("SANITY CHECK: TOP-3 RETRIEVAL TEST")
    print("=" * 80)
    test_chunk = df_meta.iloc[0]
    test_query = " ".join(test_chunk["text"].split()[:20])
    print(f"Sample Query from chunk '{test_chunk['chunk_id']}':\n\"{test_query}...\"\n")

    # Test FAISS query with sample vector
    q_vec = np.ascontiguousarray(embeddings[0:1], dtype=np.float32)
    scores, indices = faiss_index.search(q_vec, 3)
    print("FAISS Top-3 Matches:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        matched_chunk = df_meta.iloc[idx]
        print(f"  [{rank}] chunk_id: {matched_chunk['chunk_id']} | score: {score:.4f} | doc_id: {matched_chunk['doc_id']}")
    assert indices[0][0] == 0, "FAISS top-1 must be the exact identical chunk!"
    print("ASSERTION PASSED: FAISS exact match verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
