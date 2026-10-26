"""
VeriRAG Search Interface (Dense FAISS Search + BM25 Sparse Search)
==================================================================
Provides unified retrieval functions for both dense semantic search (FAISS Flat-IP)
and sparse lexical search (BM25Okapi) across the cleaned, chunked ILDC corpus.

Usage:
    from retriever.search import Retriever
    retriever = Retriever()
    dense_results = retriever.search("validity of preventive detention order", top_k=5)
    sparse_results = retriever.bm25_search("Article 21 Constitution", top_k=5)
"""

import sys
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace-and-alphanumeric tokenizer matching build_index.py."""
    return [w.lower() for w in text.split() if w.isalnum() or len(w) > 1]


class Retriever:
    """Unified retrieval engine supporting both dense (FAISS) and sparse (BM25) search."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        load_bm25: bool = True
    ):
        self.config = load_config(config_path)
        self.retriever_cfg = self.config["retriever"]
        
        self.model_name = self.retriever_cfg["embedding_model"]
        self.top_k = self.retriever_cfg.get("top_k", 5)
        
        self.faiss_index_path = PROJECT_ROOT / self.retriever_cfg["index_path"]
        self.metadata_path = PROJECT_ROOT / self.retriever_cfg["metadata_path"]
        self.bm25_index_path = PROJECT_ROOT / self.retriever_cfg["bm25_index_path"]
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. Load metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}. Build index first.")
        self.df_metadata = pd.read_parquet(self.metadata_path)
        
        # 2. Load FAISS index
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.faiss_index_path}. Build index first.")
        self.faiss_index = faiss.read_index(str(self.faiss_index_path))
        
        # 3. Load SentenceTransformer
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        
        # 4. Optional BM25
        self.bm25 = None
        self.bm25_chunk_ids = None
        if load_bm25 and self.bm25_index_path.exists():
            with open(self.bm25_index_path, "rb") as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data["bm25"]
                self.bm25_chunk_ids = bm25_data["chunk_ids"]

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Dense semantic search using FAISS Flat-IP on normalized embeddings.
        Returns top-k passages with scores, chunk IDs, and text spans.
        """
        k = top_k or self.top_k
        if not query.strip():
            return []

        # Encode query to normalized 1D vector
        q_emb = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        # Search FAISS
        scores, indices = self.faiss_index.search(q_emb, k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:
                continue
            row = self.df_metadata.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": int(row["chunk_index"]) if "chunk_index" in row else 0,
                "score": float(score),
                "text": str(row["text"]),
                "char_start": int(row["char_start"]) if "char_start" in row else 0,
                "char_end": int(row["char_end"]) if "char_end" in row else 0,
                "word_count": int(row["word_count"]) if "word_count" in row else len(str(row["text"]).split()),
                "method": "dense_faiss"
            })
        return results

    def bm25_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sparse lexical search using BM25Okapi baseline.
        """
        k = top_k or self.top_k
        if not self.bm25:
            raise RuntimeError(f"BM25 index not loaded or not found at {self.bm25_index_path}")
        if not query.strip():
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            row = self.df_metadata.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": int(row["chunk_index"]) if "chunk_index" in row else 0,
                "score": score,
                "text": str(row["text"]),
                "char_start": int(row["char_start"]) if "char_start" in row else 0,
                "char_end": int(row["char_end"]) if "char_end" in row else 0,
                "word_count": int(row["word_count"]) if "word_count" in row else len(str(row["text"]).split()),
                "method": "sparse_bm25"
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Query VeriRAG retrieval index.")
    parser.add_argument("query", type=str, nargs="?", default=None, help="Search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--method", choices=["dense", "bm25", "both"], default="both", help="Search method")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    args = parser.parse_args()

    query = args.query or "The appellant was convicted under Section 302 of the Indian Penal Code."
    print("=" * 80)
    print("VERIRAG SEARCH TEST")
    print(f"Query : \"{query}\"")
    print(f"Top-K : {args.top_k}")
    print("=" * 80)

    retriever = Retriever(config_path=Path(args.config))

    if args.method in ("dense", "both"):
        t0 = time.time()
        dense_results = retriever.search(query, top_k=args.top_k)
        dt = (time.time() - t0) * 1000
        print(f"\n--- Dense FAISS Search ({dt:.1f}ms) ---")
        for r in dense_results:
            preview = r["text"][:120].replace("\n", " ")
            print(f"[{r['rank']}] score: {r['score']:.4f} | doc: {r['doc_id']} | chunk: {r['chunk_id']}")
            print(f"    \"{preview}...\"\n")

    if args.method in ("bm25", "both") and retriever.bm25 is not None:
        t0 = time.time()
        bm25_results = retriever.bm25_search(query, top_k=args.top_k)
        dt = (time.time() - t0) * 1000
        print(f"\n--- Sparse BM25 Search ({dt:.1f}ms) ---")
        for r in bm25_results:
            preview = r["text"][:120].replace("\n", " ")
            print(f"[{r['rank']}] score: {r['score']:.4f} | doc: {r['doc_id']} | chunk: {r['chunk_id']}")
            print(f"    \"{preview}...\"\n")


if __name__ == "__main__":
    main()
