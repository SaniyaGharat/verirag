"""
VeriRAG Legal Reference & Consistency Checks
============================================
Implements deterministic regex-based legal reference validation (Hypothesis H2):
- Statutory sections & articles (e.g., Section 302 IPC, Article 21 Constitution)
- Court docket & case citations (e.g., AIR 1980 SC 1234, (2020) 4 SCC 567, SLP (Crl.) No. 123/2021)
- Temporal / date extraction and consistency validation
"""

import re
from typing import List, Dict, Any, Optional, Set

# Regex patterns for Indian legal identifiers and statutory provisions
STATUTE_PATTERNS = [
    re.compile(r'\b(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]?(?:\(\d+\))*)', re.IGNORECASE),
    re.compile(r'\b(?:Article|Art\.?)\s*(\d+[A-Za-z]?(?:\(\d+\))*)', re.IGNORECASE),
    re.compile(r'\bOrder\s+([IVXLCDM\d]+)\s+Rule\s+(\d+)', re.IGNORECASE),
    re.compile(r'\b(?:Clause|Cl\.?)\s*(\d+[A-Za-z]?)', re.IGNORECASE),
]

ACT_PATTERNS = [
    re.compile(r'\b(?:Indian\s+Penal\s+Code|IPC)\b', re.IGNORECASE),
    re.compile(r'\b(?:Code\s+of\s+Criminal\s+Procedure|Cr\.?P\.?C\.?)\b', re.IGNORECASE),
    re.compile(r'\b(?:Code\s+of\s+Civil\s+Procedure|C\.?P\.?C\.?)\b', re.IGNORECASE),
    re.compile(r'\b(?:Constitution\s+of\s+India|Constitution)\b', re.IGNORECASE),
    re.compile(r'\b(?:Arbitration\s+and\s+Conciliation\s+Act)\b', re.IGNORECASE),
    re.compile(r'\b(?:Evidence\s+Act|Indian\s+Evidence\s+Act)\b', re.IGNORECASE),
    re.compile(r'\b(?:Companies\s+Act)\b', re.IGNORECASE),
]

CASE_CITATION_PATTERNS = [
    re.compile(r'\bAIR\s+(\d{4})\s+(?:SC|Supreme\s+Court|[A-Z][a-z]+)\s+(\d+)', re.IGNORECASE),
    re.compile(r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)', re.IGNORECASE),
    re.compile(r'\[(\d{4})\]\s+(\d+)\s+SCR\s+(\d+)', re.IGNORECASE),
    re.compile(r'\b(\d{4})\s+SCALE\s+(\d+)', re.IGNORECASE),
    re.compile(r'\b(?:Criminal|Civil)\s+Appeal\s+No\.?\s*(\d+)\s*(?:of|/)\s*(\d{4})', re.IGNORECASE),
    re.compile(r'\bSLP\s*(?:\([A-Za-z]+\))?\s*No\.?\s*(\d+)\s*(?:of|/)\s*(\d{4})', re.IGNORECASE),
    re.compile(r'\bWrit\s+Petition\s*(?:\([A-Za-z]+\))?\s*No\.?\s*(\d+)\s*(?:of|/)\s*(\d{4})', re.IGNORECASE),
]

DATE_PATTERNS = [
    re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b', re.IGNORECASE),
    re.compile(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b'),
]


def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    """Extracts statutory provisions, acts, citations, and dates from text."""
    provisions = set()
    for pat in STATUTE_PATTERNS:
        for m in pat.finditer(text):
            provisions.add(m.group(0).strip())

    acts = set()
    for pat in ACT_PATTERNS:
        for m in pat.finditer(text):
            acts.add(m.group(0).strip())

    citations = set()
    for pat in CASE_CITATION_PATTERNS:
        for m in pat.finditer(text):
            citations.add(m.group(0).strip())

    dates = set()
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            dates.add(m.group(0).strip())

    return {
        "provisions": sorted(list(provisions)),
        "acts": sorted(list(acts)),
        "citations": sorted(list(citations)),
        "dates": sorted(list(dates)),
    }


def verify_legal_references(claim: str, evidence: str) -> Dict[str, Any]:
    """
    Compares legal entities extracted from claim against evidence.
    Flags contradictions, misattributions, and unverified identifiers.
    """
    claim_entities = extract_legal_entities(claim)
    evidence_entities = extract_legal_entities(evidence)

    flags = []
    has_contradiction = False

    # Check provisions (e.g., Section 302)
    claim_provs = set(claim_entities["provisions"])
    ev_provs = set(evidence_entities["provisions"])
    
    if claim_provs and ev_provs:
        # If claim asserts a provision that is completely absent while evidence mentions a competing provision
        unmatched = claim_provs - ev_provs
        if unmatched and ev_provs:
            flags.append({
                "type": "PROVISION_MISMATCH",
                "severity": "HIGH",
                "claim_items": list(unmatched),
                "evidence_items": list(ev_provs),
                "description": f"Claim mentions provision(s) {unmatched} not found in evidence provisions {ev_provs}"
            })
            # Potential contradiction if claim cites one specific section and evidence cites another under same act
            if len(claim_provs) == 1 and len(ev_provs) == 1:
                has_contradiction = True

    # Check citations (docket / report volumes)
    claim_cits = set(claim_entities["citations"])
    ev_cits = set(evidence_entities["citations"])
    if claim_cits and ev_cits:
        unmatched_cits = claim_cits - ev_cits
        if unmatched_cits:
            flags.append({
                "type": "CITATION_MISMATCH",
                "severity": "HIGH",
                "claim_items": list(unmatched_cits),
                "evidence_items": list(ev_cits),
                "description": f"Case citation mismatch: {unmatched_cits} vs {ev_cits}"
            })
            has_contradiction = True

    # Check dates
    claim_dates = set(claim_entities["dates"])
    ev_dates = set(evidence_entities["dates"])
    if claim_dates and ev_dates:
        unmatched_dates = claim_dates - ev_dates
        if unmatched_dates:
            flags.append({
                "type": "DATE_MISMATCH",
                "severity": "MEDIUM",
                "claim_items": list(unmatched_dates),
                "evidence_items": list(ev_dates),
                "description": f"Date mismatch: Claim mentions {unmatched_dates}, Evidence mentions {ev_dates}"
            })

    has_references = any(len(v) > 0 for v in claim_entities.values())

    return {
        "has_legal_references": has_references,
        "claim_entities": claim_entities,
        "evidence_entities": evidence_entities,
        "flags": flags,
        "deterministic_contradiction": has_contradiction
    }


if __name__ == "__main__":
    test_claim = "The appellant was convicted under Section 302 of the Indian Penal Code on 15th January 2020."
    test_evidence_match = "The trial court found the accused guilty under Section 302 of the Indian Penal Code on 15th January 2020."
    test_evidence_mismatch = "The High Court altered the conviction to Section 304 Part II of the Indian Penal Code."

    print("Test 1 (Matching):")
    res1 = verify_legal_references(test_claim, test_evidence_match)
    print(res1)

    print("\nTest 2 (Section Mismatch Contradiction):")
    res2 = verify_legal_references(test_claim, test_evidence_mismatch)
    print(res2)
