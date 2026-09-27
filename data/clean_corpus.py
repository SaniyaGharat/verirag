"""
ILDC Corpus Corruption Diagnosis and Dictionary-Driven Text Cleaner (Version 3)
===============================================================================
This module diagnoses and reverses systematic string-replacement artifacts
present in the Indian Legal Documents Corpus (ILDC / jayadityagandham9/ILDC_35k_COMPLETE)
using an expanded dictionary-driven candidate reconstruction engine with space-insertion
and sanity guardrails.

Enhancements in Version 3:
--------------------------
1. Expanded Candidate Sets:
   - Company family: added 'cov', 'cog' -> ['co','com','con','cor','col','cou','cov','cog']
     Resolves: 'companyered' -> 'covered', 'companynizance' -> 'cognizance', 'companyering' -> 'covering'.
   - Number family: added 'nom', 'nov' -> ['no','not','nor','non','nom','nov']
     Resolves: 'numberination' -> 'nomination', 'numberember' -> 'november', 'numberic' -> 'nomic'.

2. Space-Insertion Strategy:
   - For tokens where single-word substitution fails, tests space-inserted two-word
     candidates (e.g. 'numberdoubt' -> 'no doubt', 'numbersuch' -> 'no such',
     'numberself' -> 'no self', 'companyaccused' -> 'co accused').
   - Both words in the candidate must pass dictionary validation.
   - Resolves the ~168k-occurrence glued compound 'number + noun' family.

3. Low-Confidence Sanity Guardrail:
   - If the winning candidate has word_frequency < 1e-6 AND there is a competing
     candidate within 1 order of magnitude (f_comp >= f_win / 10.0), the resolution
     is flagged as 'low-confidence'.
   - Low-confidence resolutions are NOT auto-accepted (original token is left unchanged)
     and logged to low_confidence_resolutions.csv for manual human review.
"""

import os
import re
import functools
from collections import Counter
import wordfreq
import nltk
from nltk.corpus import words

# Suppress Windows symlink warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =====================================================================
# 1. DICTIONARY & CORPUS INITIALIZATION
# =====================================================================

try:
    NLTK_WORDS = set(w.lower() for w in words.words())
except LookupError:
    nltk.download("words", quiet=True)
    NLTK_WORDS = set(w.lower() for w in words.words())

LEGIT_COMPANY_WORDS = {
    "company",
    "companies",
    "company's",
    "companys",
    "companies'",
    "accompany",
    "accompanying",
    "accompanied",
    "accompaniment",
    "accompaniments",
    "intercompany",
}

LEGIT_NUMBER_WORDS = {
    "number",
    "numbers",
    "number's",
    "numbers'",
}

# Expanded candidate prefix sets (final list with cop and cos)
COMPANY_CANDIDATES = ['co', 'com', 'con', 'cor', 'col', 'cou', 'cov', 'cog', 'cop', 'cos']
NUMBER_CANDIDATES = ['no', 'not', 'nor', 'non', 'nom', 'nov']

MIN_FREQ_THRESHOLD = 1e-7

# Module-level tracking logs
UNRESOLVED_COMPANY_LOG = Counter()
UNRESOLVED_NUMBER_LOG = Counter()
LOW_CONFIDENCE_LOG = []
RECONSTRUCTED_COMPANY_LOG = Counter()
RECONSTRUCTED_NUMBER_LOG = Counter()
AMBIGUOUS_TIE_BREAK_LOG = []


# =====================================================================
# 2. VALIDATION & CASING HELPERS
# =====================================================================

def is_valid_candidate(cand: str, min_freq: float = MIN_FREQ_THRESHOLD) -> tuple[bool, float, str]:
    """
    Validates a word candidate using wordfreq and NLTK words fallback.
    Returns: (is_valid, frequency_score, validator_source)
    """
    c_lower = cand.lower()
    stem = c_lower[:-2] if c_lower.endswith(("'s", "’s")) else c_lower

    cand_freq = wordfreq.word_frequency(c_lower, 'en')
    stem_freq = wordfreq.word_frequency(stem, 'en') if stem != c_lower else cand_freq
    max_freq = max(cand_freq, stem_freq)

    if max_freq >= min_freq:
        return True, max_freq, "wordfreq"
    
    if c_lower in NLTK_WORDS or stem in NLTK_WORDS:
        return True, max_freq, "nltk"
        
    return False, max_freq, "invalid"


def _restore_casing(original: str, replacement: str) -> str:
    """Restores the casing style of the original token, supporting multi-word replacements."""
    if " " in replacement:
        parts = replacement.split(" ")
        if original.isupper():
            return " ".join(p.upper() for p in parts)
        elif original.istitle() or (len(original) > 0 and original[0].isupper()):
            return parts[0].capitalize() + " " + parts[1]
        return replacement.lower()

    if original.isupper():
        return replacement.upper()
    elif original.istitle() or (len(original) > 0 and original[0].isupper() and original[1:].islower()):
        return replacement.capitalize()
    return replacement.lower()


# =====================================================================
# 3. RECONSTRUCTION ENGINE WITH SPACE-SPLITTING & GUARDRAILS
# =====================================================================

def _reconstruct_token_generic(word: str, candidates: list[str], is_company: bool) -> tuple[str, str, dict]:
    """
    Core engine that handles candidate substitutions, space-splitting,
    tie-breaking, and low-confidence guardrail checking.
    
    Returns: (reconstructed_word, status, metadata)
    status can be: 'unchanged', 'single', 'two_word', 'low_confidence', 'unresolved'
    """
    if not word or not isinstance(word, str):
        return word, "unchanged", {}

    w_lower = word.lower()
    target = 'company' if is_company else 'number'
    family_name = 'company' if is_company else 'number'

    # Check standalone legitimate words
    legit_set = LEGIT_COMPANY_WORDS if is_company else LEGIT_NUMBER_WORDS
    if w_lower in legit_set:
        return word, "unchanged", {}

    # -------------------------------------------------------------
    # Strategy 1: Single-Word Substitution
    # -------------------------------------------------------------
    single_cands = []
    for c in candidates:
        cand_lower = re.sub(target, c, w_lower)
        is_valid, freq, source = is_valid_candidate(cand_lower)
        if is_valid:
            single_cands.append((cand_lower, c, freq, source))

    # Evaluate single-word candidates that have wordfreq > 0
    pos_single = [c for c in single_cands if c[2] > 0]
    if pos_single:
        pos_single.sort(key=lambda x: (x[2], -candidates.index(x[1])), reverse=True)
        winner = pos_single[0]
        f_win = winner[2]

        # Guardrail check for single-word
        if f_win < 1e-6 and len(pos_single) > 1:
            comp = pos_single[1]
            f_comp = comp[2]
            if f_comp > 0 and (f_win / f_comp) <= 10.0:
                meta = {
                    "family": family_name,
                    "original": word,
                    "mode": "single_word",
                    "winner": winner[0],
                    "winner_freq": f_win,
                    "competing": comp[0],
                    "competing_freq": f_comp,
                    "candidates": [c[0] for c in pos_single]
                }
                LOW_CONFIDENCE_LOG.append(meta)
                return word, "low_confidence", meta

        res_str = _restore_casing(word, winner[0])
        meta = {
            "family": family_name,
            "original": word,
            "winner": res_str,
            "candidates": [(c[0], c[2]) for c in pos_single]
        }
        if len(pos_single) > 1:
            AMBIGUOUS_TIE_BREAK_LOG.append(meta)
        return res_str, "single", meta

    # -------------------------------------------------------------
    # Strategy 2: Space-Insertion Substitution (Two-Word Candidate)
    # -------------------------------------------------------------
    two_word_cands = []
    for c in candidates:
        cand_str = re.sub(target, f"{c} ", w_lower)
        parts = cand_str.split(" ")
        if len(parts) == 2:
            v1, f1, s1 = is_valid_candidate(parts[0])
            v2, f2, s2 = is_valid_candidate(parts[1])
            if v1 and v2:
                # Score is min(f1, f2)
                two_word_cands.append((cand_str, c, min(f1, f2), f1, f2))

    if two_word_cands:
        # Sort: highest min-frequency, then candidate priority order
        two_word_cands.sort(key=lambda x: (x[2], -candidates.index(x[1])), reverse=True)
        winner = two_word_cands[0]
        f_win = winner[2]

        # Guardrail check for two-word
        # If the second word is a rare word (f2 < 1e-6) and there are competing candidates
        if f_win < 1e-6 and len(two_word_cands) > 1:
            comp = two_word_cands[1]
            f_comp = comp[2]
            if f_comp > 0 and (f_win / f_comp) <= 10.0:
                # If second word is rare, flag low-confidence
                if winner[4] < 1e-6:
                    meta = {
                        "family": family_name,
                        "original": word,
                        "mode": "space_insertion",
                        "winner": winner[0],
                        "winner_freq": f_win,
                        "competing": comp[0],
                        "competing_freq": f_comp,
                        "candidates": [c[0] for c in two_word_cands]
                    }
                    LOW_CONFIDENCE_LOG.append(meta)
                    return word, "low_confidence", meta

        res_str = _restore_casing(word, winner[0])
        meta = {
            "family": family_name,
            "original": word,
            "winner": res_str,
            "candidates": [(c[0], c[2]) for c in two_word_cands]
        }
        if len(two_word_cands) > 1:
            AMBIGUOUS_TIE_BREAK_LOG.append(meta)
        return res_str, "two_word", meta

    # -------------------------------------------------------------
    # Fallback: Zero-frequency single-word candidates (NLTK-only)
    # -------------------------------------------------------------
    if single_cands:
        single_cands.sort(key=lambda x: -candidates.index(x[1]))
        winner = single_cands[0]
        res_str = _restore_casing(word, winner[0])
        meta = {
            "family": family_name,
            "original": word,
            "winner": res_str,
            "candidates": [(c[0], c[2]) for c in single_cands]
        }
        return res_str, "single_nltk", meta

    # -------------------------------------------------------------
    # Zero candidates valid: Log Unresolved
    # -------------------------------------------------------------
    if is_company:
        UNRESOLVED_COMPANY_LOG[word] += 1
    else:
        UNRESOLVED_NUMBER_LOG[word] += 1

    return word, "unresolved", {}


@functools.lru_cache(maxsize=65536)
def reconstruct_company_word(word: str) -> str:
    """Reconstructs words containing 'company' as a substring."""
    res, status, _ = _reconstruct_token_generic(word, COMPANY_CANDIDATES, is_company=True)
    if status in ("single", "two_word", "single_nltk"):
        RECONSTRUCTED_COMPANY_LOG[word] += 1
    return res


@functools.lru_cache(maxsize=65536)
def reconstruct_number_word(word: str) -> str:
    """Reconstructs words containing 'number' as a substring."""
    res, status, _ = _reconstruct_token_generic(word, NUMBER_CANDIDATES, is_company=False)
    if status in ("single", "two_word", "single_nltk"):
        RECONSTRUCTED_NUMBER_LOG[word] += 1
    return res


TOKEN_CORRUPTION_PATTERN = re.compile(
    r"\b[A-Za-z]*(?:company|number)[A-Za-z]*(?:['’][A-Za-z]+)?\b",
    re.IGNORECASE
)


# =====================================================================
# 4. CONTEXTUAL STANDALONE 'NUMBER' CLEANER
# =====================================================================

def _clean_standalone_number(text: str) -> str:
    """Preserves legitimate noun phrases while repairing auxiliary/existential negations."""
    text = re.sub(
        r'\b(shall|will|would|could|should|can|may|might|must|do|does|did|is|are|was|were|has|have|had|having|need|dare|ought)\s+number\b(?!\s+(?:of|\d+|one|two|three|four|five)\b)',
        r'\1 not',
        text,
        flags=re.I
    )
    text = re.sub(
        r'\b(there\s+(?:is|was|were|are|being))\s+number\b(?!\s+(?:of|\d+)\b)',
        r'\1 no',
        text,
        flags=re.I
    )
    text = re.sub(
        r'\b(whether\s+or|or|if|why)\s+number\b(?!\s+of\b)',
        r'\1 not',
        text,
        flags=re.I
    )
    text = re.sub(
        r'\bnumber\s+(only|exceeding|less\s+than|more\s+than|earlier\s+than|later\s+than|below|above|to\s+apply|to\s+be|to\s+have|to\s+interfere|being|having|found)\b',
        r'not \1',
        text,
        flags=re.I
    )
    text = re.sub(
        r'\bnumber\s+(available|maintainable|sustainable|applicable|entitled|eligible|permissible|feasible|possible|acceptable|justified|satisfied|proved|proven|guilty|valid|binding|correct|true|clear|disputed|challenged)\b',
        r'not \1',
        text,
        flags=re.I
    )
    return text


# =====================================================================
# 5. MAIN TEXT CLEANING PIPELINE
# =====================================================================

def clean_text(text: str) -> str:
    """Cleans ILDC text using dictionary reconstruction, space-insertion, and guardrails."""
    if not text or not isinstance(text, str):
        return text

    def _replace_token(match: re.Match) -> str:
        tok = match.group(0)
        tok_low = tok.lower()
        if 'company' in tok_low:
            return reconstruct_company_word(tok)
        elif 'number' in tok_low:
            if tok_low in LEGIT_NUMBER_WORDS:
                return tok
            return reconstruct_number_word(tok)
        return tok

    text = TOKEN_CORRUPTION_PATTERN.sub(_replace_token, text)
    text = _clean_standalone_number(text)
    return text


def reset_logs():
    UNRESOLVED_COMPANY_LOG.clear()
    UNRESOLVED_NUMBER_LOG.clear()
    LOW_CONFIDENCE_LOG.clear()
    RECONSTRUCTED_COMPANY_LOG.clear()
    RECONSTRUCTED_NUMBER_LOG.clear()
    AMBIGUOUS_TIE_BREAK_LOG.clear()
    reconstruct_company_word.cache_clear()
    reconstruct_number_word.cache_clear()


if __name__ == "__main__":
    sample_text = (
        "There is numberdoubt that the companyered vehicle was numberreason to delay. "
        "The High Companyrt and learned companynsel companyld numberice that "
        "numberination was filed in numberember. The case number 42 had numberself-respecting "
        "officers who companyered the proceedings."
    )
    print("BEFORE:\n", sample_text)
    print("\nAFTER:\n", clean_text(sample_text))
