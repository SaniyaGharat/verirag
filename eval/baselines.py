"""
VeriRAG Evaluation Baselines (Experiment A)
===========================================
Implements non-NLI verification baselines as specified in research plan Deliverable 5:
1. LexicalROUGEVerifier: Surface lexical overlap (ROUGE-L F1 score thresholding)
2. BM25Verifier: Lexical term-weighting score thresholding
3. CosineEmbeddingVerifier: Bi-encoder dense embedding cosine similarity thresholding
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LexicalROUGEVerifier:
    """Baseline 1: ROUGE-L F1 overlap thresholding."""

    def __init__(self, threshold_supported: float = 0.50):
        self.threshold = threshold_supported
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        scores = self.scorer.score(evidence, claim)
        rouge_l_f1 = float(scores["rougeL"].fmeasure)

        # Lexical overlap cannot distinguish contradiction from lack of support
        if rouge_l_f1 >= self.threshold:
            verdict = "SUPPORTED"
        else:
            verdict = "UNSUPPORTED"

        return {
            "verdict": verdict,
            "confidence": rouge_l_f1,
            "method": "rouge_l",
            "score": rouge_l_f1
        }

    def verify_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.verify(cl, ev) for cl, ev in pairs]


class BM25Verifier:
    """Baseline 2: BM25 score thresholding between claim and evidence."""

    def __init__(self, threshold_score: float = 8.0):
        self.threshold = threshold_score

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if w.isalnum() or len(w) > 1]

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        evidence_tokens = self._tokenize(evidence)
        claim_tokens = self._tokenize(claim)

        if not evidence_tokens or not claim_tokens:
            return {"verdict": "UNSUPPORTED", "confidence": 0.0, "score": 0.0}

        # BM25 model fitted on the single evidence passage as corpus
        bm25 = BM25Okapi([evidence_tokens])
        scores = bm25.get_scores(claim_tokens)
        bm25_score = float(scores[0]) if len(scores) > 0 else 0.0

        if bm25_score >= self.threshold:
            verdict = "SUPPORTED"
        else:
            verdict = "UNSUPPORTED"

        return {
            "verdict": verdict,
            "confidence": min(1.0, bm25_score / 20.0),
            "method": "bm25",
            "score": bm25_score
        }

    def verify_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        return [self.verify(cl, ev) for cl, ev in pairs]


class CosineEmbeddingVerifier:
    """Baseline 3: Sentence embedding cosine similarity thresholding."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold_supported: float = 0.70,
        device: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        self.threshold = threshold_supported

    def verify(self, claim: str, evidence: str) -> Dict[str, Any]:
        results = self.verify_batch([(claim, evidence)])
        return results[0]

    def verify_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[Dict[str, Any]]:
        if not pairs:
            return []

        claims = [cl for cl, ev in pairs]
        evidences = [ev for cl, ev in pairs]

        cl_embs = self.model.encode(claims, batch_size=batch_size, normalize_embeddings=True)
        ev_embs = self.model.encode(evidences, batch_size=batch_size, normalize_embeddings=True)

        cos_sims = np.sum(cl_embs * ev_embs, axis=1)

        results = []
        for sim in cos_sims:
            sim_val = float(sim)
            if sim_val >= self.threshold:
                verdict = "SUPPORTED"
            else:
                verdict = "UNSUPPORTED"

            results.append({
                "verdict": verdict,
                "confidence": sim_val,
                "method": "cosine_embedding",
                "score": sim_val
            })

        return results
