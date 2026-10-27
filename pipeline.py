"""
VeriRAG End-to-End Verification Pipeline
=========================================
Integrated system executing:
1. Decomposer: Splits complex legal text into atomic, self-contained claims.
2. Retriever: Dense FAISS (or BM25) passage retrieval across ILDC corpus.
3. Verifier: NLI Cross-Encoder + deterministic legal consistency validation.
4. Aggregator: Trust score calculation, citation linking, and audit report generation.

Usage:
    from pipeline import VeriRAG
    rag = VeriRAG()
    result = rag.verify(
        text="The appellant was convicted under Section 302 IPC. The High Court acquitted him.",
        query="What was the outcome of the appeal?"
    )
    print(result["report_markdown"])
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from decomposer.decompose import decompose
from retriever.search import Retriever
from verifier.nli_verify import NLIVerifier
from verifier.verify_pipeline import VerificationPipeline
from aggregator.aggregate import aggregate

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VeriRAG:
    """VeriRAG end-to-end legal verification framework."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        retriever: Optional[Retriever] = None,
        verifier: Optional[VerificationPipeline] = None
    ):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        # Initialize components
        print("[VeriRAG] Initializing retrieval engine...")
        self.retriever = retriever or Retriever(config_path=self.config_path)
        
        print("[VeriRAG] Initializing verification pipeline...")
        self.verifier = verifier or VerificationPipeline(config_path=self.config_path)

    def verify(
        self,
        text: str,
        query: str = "",
        top_k: int = 5,
        decompose_method: str = "auto",
        source_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end verification of legal text.
        """
        t0 = time.time()
        print(f"\n[VeriRAG] Processing input text ({len(text)} chars)...")

        # 1. Decompose into atomic claims
        t_dec = time.time()
        claims = decompose(text, method=decompose_method, config_path=self.config_path)
        dt_dec = (time.time() - t_dec) * 1000
        print(f"[VeriRAG] Extracted {len(claims)} atomic claims ({dt_dec:.1f}ms).")

        # 2. Retrieve & Verify per claim
        claim_verdicts = []
        for c in claims:
            c_text = c["claim_text"]
            t_ret = time.time()
            if source_doc_id is None:
                passages = self.retriever.search(c_text, top_k=top_k)
            else:
                passages = self.retriever.search(
                    c_text, top_k=top_k, source_doc_id=source_doc_id
                )
            dt_ret = (time.time() - t_ret) * 1000

            t_ver = time.time()
            verdict_res = self.verifier.verify_claim(
                c_text, passages, source_doc_id=source_doc_id
            )
            dt_ver = (time.time() - t_ver) * 1000

            verdict_res["claim_index"] = c["claim_index"]
            verdict_res["source_sentence"] = c.get("source_sentence", "")
            verdict_res["retrieval_time_ms"] = dt_ret
            verdict_res["verification_time_ms"] = dt_ver

            claim_verdicts.append(verdict_res)

        # 3. Aggregate Trust Score and Citations
        t_agg = time.time()
        aggregated = aggregate(
            claim_verdicts=claim_verdicts,
            query=query,
            original_text=text
        )
        dt_agg = (time.time() - t_agg) * 1000

        total_time_ms = (time.time() - t0) * 1000

        return {
            "query": query,
            "original_text": text,
            "trust_score": aggregated["trust_score"],
            "trust_summary": aggregated["trust_summary"],
            "claim_verdicts": claim_verdicts,
            "citations": aggregated["citations"],
            "report_markdown": aggregated["report_markdown"],
            "latency_ms": {
                "decomposition_ms": dt_dec,
                "total_time_ms": total_time_ms
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Run VeriRAG end-to-end verification pipeline.")
    parser.add_argument("text", type=str, nargs="?", default=None, help="Input legal text to verify")
    parser.add_argument("--query", type=str, default="", help="Optional legal query context")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved passages per claim")
    parser.add_argument("--source-doc-id", type=str, default=None, help="Optional judgment ID to restrict evidence to one docket")
    parser.add_argument("--decompose-method", choices=["auto", "llm", "rule_based"], default="auto")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    sample_text = (
        args.text or
        "The appellant, convicted under Section 302 of the Indian Penal Code by the Sessions Court, "
        "was sentenced to life imprisonment, but was subsequently acquitted by the High Court due to "
        "material discrepancies in witness testimony."
    )

    rag = VeriRAG(config_path=Path(args.config))
    result = rag.verify(
        text=sample_text,
        query=args.query,
        top_k=args.top_k,
        decompose_method=args.decompose_method,
        source_doc_id=args.source_doc_id
    )

    print("\n" + "=" * 80)
    print(result["report_markdown"])
    print("=" * 80)


if __name__ == "__main__":
    main()
