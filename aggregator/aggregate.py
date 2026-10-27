"""
VeriRAG Aggregator & Citation Linking
=====================================
Aggregates atomic claim verdicts into an overall response trust score,
compiles grounded evidence citations, and generates verified markdown reports.

Outputs:
- overall_trust_score: float in [0.0, 1.0]
- verdict_distribution: count and percentage of SUPPORTED, CONTRADICTED, UNSUPPORTED
- citations: traceable evidence mappings (doc_id, chunk_id, char_span, score)
- formatted_report: structured markdown response with inline claim verdicts
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_trust_score(
    claim_verdicts: List[Dict[str, Any]],
    contradiction_penalty: float = 1.0
) -> Dict[str, Any]:
    """
    Computes calibrated trust score across all atomic claims.
    
    Formula:
    - Base score: proportion of verified supported claims weighted by confidence.
    - Contradiction penalty: heavily penalizes direct factual contradictions.
    """
    total_claims = len(claim_verdicts)
    if total_claims == 0:
        return {
            "trust_score": 1.0,
            "counts": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 0},
            "percentages": {"SUPPORTED": 0.0, "CONTRADICTED": 0.0, "UNSUPPORTED": 0.0}
        }

    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 0}
    weighted_sum = 0.0

    for c in claim_verdicts:
        v = c["verdict"]
        conf = c.get("confidence", 1.0)
        counts[v] = counts.get(v, 0) + 1

        if v == "SUPPORTED":
            weighted_sum += conf
        elif v == "CONTRADICTED":
            weighted_sum -= (contradiction_penalty * conf)
        elif v == "UNSUPPORTED":
            weighted_sum += 0.0

    raw_trust = weighted_sum / total_claims
    bounded_trust = max(0.0, min(1.0, raw_trust))

    percentages = {
        k: (v / total_claims) * 100.0 for k, v in counts.items()
    }

    return {
        "trust_score": round(bounded_trust, 4),
        "total_claims": total_claims,
        "counts": counts,
        "percentages": {k: round(v, 2) for k, v in percentages.items()}
    }


def link_citations(claim_verdicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts and deduplicates traceable evidence citations for all supported/contradicted claims.
    """
    citations = []
    seen_chunks = set()

    for idx, c in enumerate(claim_verdicts, 1):
        best_ev = c.get("best_evidence")
        if best_ev and best_ev.get("chunk_id"):
            chunk_id = best_ev["chunk_id"]
            citations.append({
                "citation_index": idx,
                "claim_index": c.get("claim_index", idx),
                "claim_text": c.get("claim_text"),
                "verdict": c.get("verdict"),
                "doc_id": best_ev.get("doc_id"),
                "chunk_id": chunk_id,
                "char_span": [best_ev.get("char_start", 0), best_ev.get("char_end", 0)],
                "evidence_text": best_ev.get("text", ""),
                "retrieval_score": best_ev.get("score", 0.0)
            })
            seen_chunks.add(chunk_id)

    return citations


def format_verified_report(
    original_query: str,
    original_response: str,
    claim_verdicts: List[Dict[str, Any]],
    trust_summary: Dict[str, Any],
    citations: List[Dict[str, Any]]
) -> str:
    """Formats a complete human-readable and audit-ready Markdown report."""
    score = trust_summary["trust_score"]
    total = trust_summary["total_claims"]
    c_counts = trust_summary["counts"]

    # Status badge
    if score >= 0.85:
        badge = "HIGH TRUST"
    elif score >= 0.60:
        badge = "MODERATE TRUST"
    else:
        badge = "LOW TRUST / CONTRADICTED"

    lines = [
        "# VeriRAG Verification Report",
        "",
        f"**Status**: [{badge}] (Trust Score: `{score:.2%}`)",
        f"**Claim Summary**: {c_counts.get('SUPPORTED', 0)} Supported | {c_counts.get('CONTRADICTED', 0)} Contradicted | {c_counts.get('UNSUPPORTED', 0)} Unsupported (Total: {total})",
        "",
        "---",
        "",
        "### 1. Query & Original Response",
        f"**Query**: {original_query}",
        "",
        "**Generated Response**:",
        f"> {original_response}",
        "",
        "---",
        "",
        "### 2. Decomposed Claims & Verification Verdicts",
        ""
    ]

    for idx, c in enumerate(claim_verdicts, 1):
        v = c["verdict"]
        conf = c.get("confidence", 0.0)
        lines.append(f"{idx}. **[{v}]** `{c['claim_text']}` *(Confidence: {conf:.2f})*")

        reason = c.get("reason")
        if reason:
            lines.append(f"   - **Reason**: {reason}")
        
        # Add flags if present
        flags = c.get("legal_flags", [])
        if flags:
            for f in flags:
                lines.append(f"   - [FLAG: {f.get('type')}]: {f.get('description')}")

        best_ev = c.get("best_evidence")
        if best_ev and best_ev.get("chunk_id"):
            ev_snippet = best_ev.get("text", "")[:150].replace("\n", " ")
            lines.append(f"   - *Evidence [{best_ev.get('doc_id')}:{best_ev.get('chunk_id')}]*: \"{ev_snippet}...\"")
        lines.append("")

    lines.extend([
        "---",
        "",
        "### 3. Traceable Evidence Citations",
        ""
    ])

    if citations:
        for cit in citations:
            lines.append(
                f"- **Citation [{cit['citation_index']}]** (Doc `{cit['doc_id']}`, Chunk `{cit['chunk_id']}`, Span `{cit['char_span']}`):"
            )
            lines.append(f"  > \"{cit['evidence_text'][:300]}...\"")
            lines.append("")
    else:
        lines.append("*No supporting citations verified.*")

    return "\n".join(lines)


def aggregate(
    claim_verdicts: List[Dict[str, Any]],
    query: str = "",
    original_text: str = ""
) -> Dict[str, Any]:
    """Unified entrypoint for verification aggregation."""
    trust_summary = compute_trust_score(claim_verdicts)
    citations = link_citations(claim_verdicts)
    report = format_verified_report(
        original_query=query,
        original_response=original_text,
        claim_verdicts=claim_verdicts,
        trust_summary=trust_summary,
        citations=citations
    )

    return {
        "trust_score": trust_summary["trust_score"],
        "trust_summary": trust_summary,
        "citations": citations,
        "report_markdown": report
    }


if __name__ == "__main__":
    sample_verdicts = [
        {
            "claim_index": 1,
            "claim_text": "The appellant was convicted under Section 302 IPC.",
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "best_evidence": {
                "chunk_id": "chunk_001",
                "doc_id": "doc_1998_45",
                "char_start": 10,
                "char_end": 120,
                "text": "The trial court convicted the appellant under Section 302 IPC and sentenced him to life imprisonment.",
                "score": 0.91
            }
        },
        {
            "claim_index": 2,
            "claim_text": "The High Court confirmed the death sentence.",
            "verdict": "CONTRADICTED",
            "confidence": 0.92,
            "best_evidence": {
                "chunk_id": "chunk_002",
                "doc_id": "doc_1998_45",
                "char_start": 300,
                "char_end": 420,
                "text": "The High Court acquitted the appellant and set aside the conviction.",
                "score": 0.89
            }
        }
    ]

    res = aggregate(sample_verdicts, query="What was the outcome of the appeal?", original_text="The appellant was convicted under Section 302 IPC and the High Court confirmed the death sentence.")
    print(res["report_markdown"])
