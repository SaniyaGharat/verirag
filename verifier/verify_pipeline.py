"""
VeriRAG Claim Verification Pipeline
===================================
Coordinates NLI verification and deterministic legal-reference checks across
retrieved evidence passages for an atomic legal claim.

Outputs verifiable, provenance-tracked claim verdicts:
- SUPPORTED
- CONTRADICTED
- UNSUPPORTED
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verifier.nli_verify import NLIVerifier
from verifier.legal_checks import verify_legal_references

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VerificationPipeline:
    """Verifies atomic claims against multiple retrieved evidence passages."""

    def __init__(
        self,
        verifier: Optional[NLIVerifier] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
        enable_legal_checks: bool = True
    ):
        self.config = load_config(config_path)
        v_cfg = self.config.get("verifier", {})
        
        self.enable_legal_checks = enable_legal_checks and v_cfg.get("enable_legal_checks", True)
        self.verifier = verifier or NLIVerifier(config_path=config_path)

    def verify_claim(
        self,
        claim_text: str,
        retrieved_passages: List[Dict[str, Any]],
        source_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies a single atomic claim against a list of retrieved evidence passages.
        """
        if source_doc_id is None and retrieved_passages:
            # Search results are ranked globally across the corpus. In the
            # absence of an explicit docket, use the top-ranked result's
            # judgment as the candidate source and never combine verdicts from
            # other judgments.
            source_doc_id = next(
                (p.get("doc_id") for p in retrieved_passages if p.get("doc_id") is not None),
                None
            )

        if source_doc_id is not None:
            retrieved_passages = [
                p for p in retrieved_passages if str(p.get("doc_id")) == str(source_doc_id)
            ]

        if not retrieved_passages:
            return {
                "claim_text": claim_text,
                "verdict": "UNSUPPORTED",
                "confidence": 1.0,
                "reason": "No evidence passages retrieved for the requested source document",
                "best_evidence": None,
                "source_doc_id": source_doc_id,
                "legal_flags": [],
                "evaluations": []
            }

        # 1. Batch NLI inference on all (claim, passage) pairs
        pairs = [(claim_text, p["text"]) for p in retrieved_passages]
        nli_results = self.verifier.verify_batch(pairs)

        evaluations = []
        has_contradiction = False
        con_evidence = None
        con_confidence = 0.0

        has_support = False
        best_sup_evidence = None
        best_sup_confidence = 0.0

        all_legal_flags = []

        for passage, nli_res in zip(retrieved_passages, nli_results):
            p_text = passage["text"]
            legal_res = {"flags": [], "deterministic_contradiction": False}
            if self.enable_legal_checks:
                legal_res = verify_legal_references(claim_text, p_text)
                all_legal_flags.extend(legal_res["flags"])

            eval_entry = {
                "chunk_id": passage.get("chunk_id"),
                "doc_id": passage.get("doc_id"),
                "retrieval_score": passage.get("score", 0.0),
                "nli_verdict": nli_res["verdict"],
                "nli_confidence": nli_res["confidence"],
                "probabilities": nli_res["probabilities"],
                "legal_flags": legal_res["flags"],
                "text_snippet": p_text[:200] + "..." if len(p_text) > 200 else p_text,
                "char_start": passage.get("char_start", 0),
                "char_end": passage.get("char_end", 0)
            }
            evaluations.append(eval_entry)

            # Check for hard contradiction (either deterministic or NLI)
            if legal_res.get("deterministic_contradiction") or nli_res["verdict"] == "CONTRADICTED":
                has_contradiction = True
                if nli_res["confidence"] > con_confidence:
                    con_confidence = nli_res["confidence"]
                    con_evidence = passage

            # Check for support
            if nli_res["verdict"] == "SUPPORTED":
                if nli_res["confidence"] > best_sup_confidence:
                    has_support = True
                    best_sup_confidence = nli_res["confidence"]
                    best_sup_evidence = passage

        # Synthesis Decision Logic:
        # Contradiction takes precedence if evidenced
        if has_contradiction:
            final_verdict = "CONTRADICTED"
            final_conf = max(con_confidence, 0.85 if any(f.get("severity") == "HIGH" for f in all_legal_flags) else 0.70)
            best_evidence = con_evidence
        elif has_support:
            final_verdict = "SUPPORTED"
            final_conf = best_sup_confidence
            best_evidence = best_sup_evidence
        else:
            final_verdict = "UNSUPPORTED"
            # Confidence is highest neutral probability
            final_conf = max(e["probabilities"]["unsupported"] for e in evaluations) if evaluations else 0.5
            best_evidence = retrieved_passages[0] if retrieved_passages else None

        return {
            "claim_text": claim_text,
            "verdict": final_verdict,
            "confidence": float(final_conf),
            "source_doc_id": source_doc_id,
            "best_evidence": {
                "chunk_id": best_evidence.get("chunk_id") if best_evidence else None,
                "doc_id": best_evidence.get("doc_id") if best_evidence else None,
                "char_start": best_evidence.get("char_start", 0) if best_evidence else 0,
                "char_end": best_evidence.get("char_end", 0) if best_evidence else 0,
                "text": best_evidence.get("text", "") if best_evidence else "",
                "score": best_evidence.get("score", 0.0) if best_evidence else 0.0
            } if best_evidence else None,
            "legal_flags": all_legal_flags,
            "evaluations": evaluations
        }


if __name__ == "__main__":
    sample_claim = "The High Court acquitted the appellant."
    sample_passages = [
        {
            "chunk_id": "test_chunk_1",
            "doc_id": "test_doc_1",
            "text": "The High Court dismissed the appeal and confirmed the death sentence.",
            "score": 0.88,
            "char_start": 100,
            "char_end": 170
        },
        {
            "chunk_id": "test_chunk_2",
            "doc_id": "test_doc_1",
            "text": "The appellant filed an appeal before the High Court challenging the trial court judgment.",
            "score": 0.75,
            "char_start": 0,
            "char_end": 88
        }
    ]

    pipeline = VerificationPipeline()
    result = pipeline.verify_claim(sample_claim, sample_passages)
    print("Verification Result:")
    print(f"Verdict: {result['verdict']} (Confidence: {result['confidence']:.4f})")
    print(f"Flags: {result['legal_flags']}")
