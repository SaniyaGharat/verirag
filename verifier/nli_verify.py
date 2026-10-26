"""
VeriRAG NLI Cross-Encoder Verifier
==================================
Performs claim-level natural language inference (NLI) between an atomic claim
(hypothesis) and a retrieved legal passage (premise) using a fine-tuned cross-encoder.

Default model: cross-encoder/nli-deberta-v3-base
Label space:
- 0: Contradiction  -> CONTRADICTED
- 1: Entailment     -> SUPPORTED
- 2: Neutral        -> UNSUPPORTED
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
import torch
import numpy as np
from sentence_transformers import CrossEncoder

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NLIVerifier:
    """NLI Cross-Encoder wrapper with calibrated thresholding for VeriRAG."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold_supported: float = 0.70,
        threshold_contradicted: float = 0.70,
        device: Optional[str] = None,
        config_path: Path = DEFAULT_CONFIG_PATH
    ):
        cfg = load_config(config_path)
        v_cfg = cfg.get("verifier", {})

        self.model_name = model_name or v_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base")
        self.thresh_sup = threshold_supported or v_cfg.get("threshold_supported", 0.70)
        self.thresh_con = threshold_contradicted or v_cfg.get("threshold_contradicted", 0.70)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading NLIVerifier '{self.model_name}' on {self.device}...")
        self.model = CrossEncoder(self.model_name, device=self.device)

        # Map model label IDs to names
        # Most standard NLI models: 0=contradiction, 1=entailment, 2=neutral
        # Or read from model.config.id2label if available
        self.id2label = getattr(self.model.model.config, "id2label", {
            0: "contradiction",
            1: "entailment",
            2: "neutral"
        })

    def verify_pair(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Verifies a single (claim, evidence) pair.
        In NLI standard terminology:
        - Premise: Evidence passage
        - Hypothesis: Atomic Claim
        """
        results = self.verify_batch([(claim, evidence)])
        return results[0]

    def verify_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Verifies a batch of (claim, evidence) pairs.
        Each pair is (claim_hypothesis, evidence_premise).
        """
        if not pairs:
            return []

        # SentenceTransformers CrossEncoder expects [(premise, hypothesis)]
        cross_pairs = [(ev, cl) for (cl, ev) in pairs]

        logits = self.model.predict(cross_pairs, batch_size=batch_size, convert_to_numpy=True)
        
        # Apply softmax to get normalized probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        results = []
        for prob_vec in probs:
            # Map probabilities based on model config id2label
            p_dict = {}
            for idx, prob in enumerate(prob_vec):
                label_name = self.id2label.get(idx, str(idx)).lower()
                p_dict[label_name] = float(prob)

            p_ent = p_dict.get("entailment", 0.0)
            p_con = p_dict.get("contradiction", 0.0)
            p_neu = p_dict.get("neutral", 0.0)

            # Operational decision rule with calibrated thresholds
            if p_ent >= self.thresh_sup:
                verdict = "SUPPORTED"
                confidence = p_ent
            elif p_con >= self.thresh_con:
                verdict = "CONTRADICTED"
                confidence = p_con
            else:
                verdict = "UNSUPPORTED"
                confidence = p_neu

            results.append({
                "verdict": verdict,
                "confidence": confidence,
                "probabilities": {
                    "supported": p_ent,
                    "contradicted": p_con,
                    "unsupported": p_neu
                }
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Test NLIVerifier on sample legal claim & evidence.")
    parser.add_argument("--claim", type=str, default=None)
    parser.add_argument("--evidence", type=str, default=None)
    args = parser.parse_args()

    sample_claim = args.claim or "The High Court acquitted the appellant."
    sample_evidence = (
        args.evidence or
        "In the result, the conviction recorded by the trial court is set aside and the appellant is acquitted of all charges."
    )

    print("=" * 80)
    print("VERIRAG NLI CROSS-ENCODER VERIFIER TEST")
    print(f"Claim    : \"{sample_claim}\"")
    print(f"Evidence : \"{sample_evidence}\"")
    print("=" * 80)

    verifier = NLIVerifier()
    t0 = time.time()
    res = verifier.verify_pair(sample_claim, sample_evidence)
    dt = (time.time() - t0) * 1000

    print(f"\nResult ({dt:.1f}ms):")
    print(f"  Verdict    : {res['verdict']}")
    print(f"  Confidence : {res['confidence']:.4f}")
    print(f"  Probs      : {res['probabilities']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
