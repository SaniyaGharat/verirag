"""
Corpus-Wide Text Cleaning and Parquet Persistence Pipeline
==========================================================
Applies the verified clean_text() pipeline across all 38,904 judgments in
jayadityagandham9/ILDC_35k_COMPLETE.

Outputs:
- verirag/data/processed/ildc_cleaned.parquet
  Columns: id, text_cleaned, text_raw_length, text_cleaned_length, label

QA Metrics Logged:
- Total documents processed
- Total corruption instances fixed (company + number patterns)
- Exact before/after legitimate 'number' noun counts across all 38,904 documents
- Scan for any missed/unresolved occurrences with example sentences and document IDs
- File size, row count, and round-trip read verification
"""

import os
import re
import sys
import time
from pathlib import Path
import multiprocessing
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Suppress Hugging Face Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "ildc_cleaned.parquet"


# =====================================================================
# PRECOMPILED REGEX RULES FOR CLEANING & QA
# =====================================================================

# 1. Company replacements
COMPANY_REPLACEMENTS = [
    # Core court terms
    (re.compile(r"\bcompanyrt(s)?\b"), r"court\1"),
    (re.compile(r"\bCompanyrt(s)?\b"), r"Court\1"),
    (re.compile(r"\bcompanyrt\'s\b"), "court's"),
    (re.compile(r"\bCompanyrt\'s\b"), "Court's"),
    (re.compile(r"\bcompanynsel(s)?\b"), r"counsel\1"),
    (re.compile(r"\bCompanynsel(s)?\b"), r"Counsel\1"),
    (re.compile(r"\bcompanynsel\'s\b"), "counsel's"),
    (re.compile(r"\bCompanynsel\'s\b"), "Counsel's"),
    (re.compile(r"\bcompanyld\b"), "could"),
    (re.compile(r"\bCompanyld\b"), "Could"),
    (re.compile(r"\bcompanyldn\'t\b"), "couldn't"),
    (re.compile(r"\bcompanyntr(y|ies)\b"), r"countr\1"),
    (re.compile(r"\bCompanyntr(y|ies)\b"), r"Countr\1"),
    (re.compile(r"\bcompanyncil(s)?\b"), r"council\1"),
    (re.compile(r"\bCompanyncil(s)?\b"), r"Council\1"),
    (re.compile(r"\bcompanyrier(s)?\b"), r"courier\1"),
    # Latin / Co- prefixes
    (re.compile(r"\bcompanyabit(ation)?\b"), r"cohabit\1"),
    (re.compile(r"\bcompanyarcen(ary|er|ers)\b"), r"coparcen\1"),
    (re.compile(r"\bcompanyaccused\b"), "co-accused"),
    (re.compile(r"\bcompanyal(s)?\b"), r"coal\1"),
    (re.compile(r"\bcompanyb(s)?\b"), r"comb\1"),
    (re.compile(r"\bcompanybat\b"), "combat"),
    (re.compile(r"\bcompanybine(d|s|ing)?\b"), r"combine\1"),
    (re.compile(r"\bcompanybination(s)?\b"), r"combination\1"),
    (re.compile(r"\bcompanybust(ible|ion)?\b"), r"combust\1"),
    (re.compile(r"\bcompanyceal(ed|ing|ment|s)?\b"), r"conceal\1"),
    (re.compile(r"\bcompanycede(d|s|ing)?\b"), r"concede\1"),
    (re.compile(r"\bcompanyceiv(e|ed|ing|able)?\b"), r"conceiv\1"),
    # con- prefix stems
    (re.compile(r"\bcompanycept(s|ion|ions|ual)?\b"), r"concept\1"),
    (re.compile(r"\bcompanycern(ed|ing|s)?\b"), r"concern\1"),
    (re.compile(r"\bcompanycert(ed)?\b"), r"concert\1"),
    (re.compile(r"\bcompanycession(s)?\b"), r"concession\1"),
    (re.compile(r"\bcompanyclude(d|s|ing)?\b"), r"conclude\1"),
    (re.compile(r"\bcompanyclusi(on|ons|ve|vely)\b"), r"conclusi\1"),
    (re.compile(r"\bcompanycord\b"), "concord"),
    (re.compile(r"\bcompanycrete\b"), "concrete"),
    (re.compile(r"\bcompanycur(red|ring|rence|rent|rently)?\b"), r"concur\1"),
    (re.compile(r"\bcompanyduct(ed|ing|or|s)?\b"), r"conduct\1"),
    (re.compile(r"\bcompanyfer(red|ring|ment)?\b"), r"confer\1"),
    (re.compile(r"\bcompanyfess(ed|ing|ion|ional)?\b"), r"confess\1"),
    (re.compile(r"\bcompanyfiden(ce|t|tial)?\b"), r"confiden\1"),
    (re.compile(r"\bcompanyfine(d|s|ing|ment)?\b"), r"confine\1"),
    (re.compile(r"\bcompanyfirm(ed|ing|ation|atory)?\b"), r"confirm\1"),
    (re.compile(r"\bcompanyflict(ed|ing|s)?\b"), r"conflict\1"),
    (re.compile(r"\bcompanyform(ed|ing|ity|ance)?\b"), r"conform\1"),
    (re.compile(r"\bcompanyfound\b"), "confound"),
    (re.compile(r"\bcompanyfus(e|ed|ing|ion)?\b"), r"confus\1"),
    (re.compile(r"\bcompanygru(ous|ity)?\b"), r"congru\1"),
    (re.compile(r"\bcompanyject(ure)?\b"), r"conject\1"),
    (re.compile(r"\bcompanynect(ed|ing|ion|ions)?\b"), r"connect\1"),
    (re.compile(r"\bcompanyniv(e|ed|ing|ance)?\b"), r"conniv\1"),
    (re.compile(r"\bcompanynotation(s)?\b"), r"connotation\1"),
    (re.compile(r"\bcompanyquer(ed|ing|or)?\b"), r"conquer\1"),
    (re.compile(r"\bcompanyscious(ness)?\b"), r"conscious\1"),
    (re.compile(r"\bcompanysecut(ive|ively)?\b"), r"consecut\1"),
    (re.compile(r"\bcompanysent(ed|ing)?\b"), r"consent\1"),
    (re.compile(r"\bcompanysequen(ce|ces|t|tly)?\b"), r"consequen\1"),
    (re.compile(r"\bcompanyserv(e|ed|ing|ation)?\b"), r"conserv\1"),
    (re.compile(r"\bcompanysider(\w*)\b"), r"consider\1"),
    (re.compile(r"\bCompanysider(\w*)\b"), r"Consider\1"),
    (re.compile(r"\bcompanysist(ed|ing|s|ence|ent)?\b"), r"consist\1"),
    (re.compile(r"\bcompanysol(e|ed|ing|idation|idated)?\b"), r"consol\1"),
    (re.compile(r"\bcompanysonan(ce)?\b"), r"consonan\1"),
    (re.compile(r"\bcompanyspicuous\b"), "conspicuous"),
    (re.compile(r"\bcompanystitut(e|ed|es|ing|ion|ional|ionally)?\b"), r"constitut\1"),
    (re.compile(r"\bcompanystruct(ed|ing|ion)?\b"), r"construct\1"),
    (re.compile(r"\bcompanystru(e|ed|ing)?\b"), r"constru\1"),
    (re.compile(r"\bcompanysult(ed|ing|ation)?\b"), r"consult\1"),
    (re.compile(r"\bcompanysum(e|ed|ing|er|ers|ption)?\b"), r"consum\1"),
    (re.compile(r"\bcompanytain(ed|ing|s)?\b"), r"contain\1"),
    (re.compile(r"\bcompanytest(ed|ing|s)?\b"), r"contest\1"),
    (re.compile(r"\bcompanytext(s|ual)?\b"), r"context\1"),
    (re.compile(r"\bcompanytiguous\b"), "contiguous"),
    (re.compile(r"\bcompanytinu(e|ed|es|ing|ation|ous|ity)?\b"), r"continu\1"),
    (re.compile(r"\bcompanytort(ed|ion)?\b"), r"contort\1"),
    (re.compile(r"\bcompanytract(ed|ing|s|or|ors|ual)?\b"), r"contract\1"),
    (re.compile(r"\bcompanytradict(ed|ing|s|ion|ory)?\b"), r"contradict\1"),
    (re.compile(r"\bcompanytraven(e|ed|es|ing|tion)?\b"), r"contraven\1"),
    (re.compile(r"\bcompanytribut(e|ed|es|ing|ion|ions|ory)?\b"), r"contribut\1"),
    (re.compile(r"\bcompanytrol(led|ling|ler|lers|s)?\b"), r"control\1"),
    (re.compile(r"\bcompanytrovers(y|ies|ial)?\b"), r"controvers\1"),
    (re.compile(r"\bcompanyvene(d|s|ing)?\b"), r"convene\1"),
    (re.compile(r"\bcompanyvenien(ce|t|tly)?\b"), r"convenien\1"),
    (re.compile(r"\bcompanyvent(ion|ional|ions)?\b"), r"convent\1"),
    (re.compile(r"\bcompanyvers(ation|e|ed|ing)?\b"), r"convers\1"),
    (re.compile(r"\bcompanyvert(ed|ing|s|ion)?\b"), r"convert\1"),
    (re.compile(r"\bcompanyvey(ed|ing|s|ance)?\b"), r"convey\1"),
    (re.compile(r"\bcompanyvict(ed|ing|s|ion|ions)?\b"), r"convict\1"),
    (re.compile(r"\bcompanyvinc(e|ed|ing)?\b"), r"convinc\1"),
    (re.compile(r"\bcompanyvok(e|ed|ing)?\b"), r"convok\1"),
    # com- prefix stems
    (re.compile(r"\bcompanymon(ly|wealth)?\b"), r"common\1"),
    (re.compile(r"\bcompanymend(ed|ing|ation)?\b"), r"commend\1"),
    (re.compile(r"\bcompanyment(ed|ing|ary|s)?\b"), r"comment\1"),
    (re.compile(r"\bcompanymerc(e|ial|ially)?\b"), r"commerc\1"),
    (re.compile(r"\bcompanymiss(ion|ioner|ioned)?\b"), r"commiss\1"),
    (re.compile(r"\bcompanymit(ted|ting|tee|tees|ment)?\b"), r"commit\1"),
    (re.compile(r"\bcompanymod(ity|ities)?\b"), r"commod\1"),
    (re.compile(r"\bcompanymunic(ate|ated|ating|ation|ations)?\b"), r"communic\1"),
    (re.compile(r"\bcompanymut(e|ed|ing|ation)?\b"), r"commut\1"),
    (re.compile(r"\bcompanypact\b"), "compact"),
    (re.compile(r"\bcompanypar(e|ed|es|ing|ison|isons|ative|atively)?\b"), r"compar\1"),
    (re.compile(r"\bcompanypass(ion|ionate)?\b"), r"compass\1"),
    (re.compile(r"\bcompanypat(ible|ibility)?\b"), r"compat\1"),
    (re.compile(r"\bcompanypel(led|ling)?\b"), r"compel\1"),
    (re.compile(r"\bcompanypens(e|ed|ation|ations|atory)?\b"), r"compens\1"),
    (re.compile(r"\bcompanypet(e|ed|es|ing|ent|ently|ence|ency|ition|itions|itor|itors)?\b"), r"compet\1"),
    (re.compile(r"\bcompanypil(e|ed|ing|ation)?\b"), r"compil\1"),
    (re.compile(r"\bcompanyplain(ed|ing|t|ts|ant|ants)?\b"), r"complain\1"),
    (re.compile(r"\bcompanyple(te|ted|tely|tion|x|xity)?\b"), r"comple\1"),
    (re.compile(r"\bcompanypli(ed|es|ance|ant)?\b"), r"compli\1"),
    (re.compile(r"\bcompanyply\b"), "comply"),
    (re.compile(r"\bcompanypos(e|ed|ing|ition|ite)?\b"), r"compos\1"),
    (re.compile(r"\bcompanypound(ed|ing|s)?\b"), r"compound\1"),
    (re.compile(r"\bcompanyprehen(d|ded|ding|sion|sive)?\b"), r"comprehen\1"),
    (re.compile(r"\bcompanypris(e|ed|es|ing)?\b"), r"compris\1"),
    (re.compile(r"\bcompanypromis(e|ed|es|ing)?\b"), r"compromis\1"),
    (re.compile(r"\bcompanyput(e|ed|es|ing|ation)?\b"), r"comput\1"),
]

# 2. Number replacements
NUMBER_REPLACEMENTS = [
    # Suffixes corrupted from not-
    (re.compile(r"\bnumberification(s)?\b"), r"notification\1"),
    (re.compile(r"\bNumberification(s)?\b"), r"Notification\1"),
    (re.compile(r"\bnumberified\b"), "notified"),
    (re.compile(r"\bnumberif(y|ies|ying)\b"), r"notif\1"),
    (re.compile(r"\bnumberwithstanding\b"), "notwithstanding"),
    (re.compile(r"\bNumberwithstanding\b"), "Notwithstanding"),
    (re.compile(r"\bnumberice(s)?\b"), r"notice\1"),
    (re.compile(r"\bNumberice(s)?\b"), r"Notice\1"),
    (re.compile(r"\bnumbericeable\b"), "noticeable"),
    (re.compile(r"\bnumberable\b"), "notable"),
    (re.compile(r"\bnumberably\b"), "notably"),
    (re.compile(r"\bnumberion(s)?\b"), r"notion\1"),
    (re.compile(r"\bnumberional\b"), "notional"),
    (re.compile(r"\bnumberoriety\b"), "notoriety"),
    (re.compile(r"\bnumberorious(ly)?\b"), r"notorious\1"),
    (re.compile(r"\bnumberary\b"), "notary"),
    (re.compile(r"\bnumberebook(s)?\b"), r"notebook\1"),
    # Glued phrases
    (re.compile(r"\bnumbersuch\b"), "no such"),
    (re.compile(r"\bnumber(candidate|person|witness|appeal|application|evidence|ground|reason|case|order|appointment|suit|doubt|fault|right|claim|justification)(s)?\b"), r"no \1\2"),
    (re.compile(r"\bnumber(compliance|payment|joinder|appearance|delivery|disclosure|performance|resident|user|violence|applicant|party|agricultural|bailable|cognizable|interference|obstruction)\b"), r"non-\1"),
    (re.compile(r"\bnumber([A-Z][a-z]+)\b"), r"No \1"),
    # Auxiliary verbs + negation
    (re.compile(r"\b(shall|will|would|could|should|can|may|might|must|do|does|did|is|are|was|were|has|have|had|having|need|dare|ought)\s+number\b(?!\s+(?:of|\d+|one|two|three|four|five)\b)", re.I), r"\1 not"),
    (re.compile(r"\b(there\s+(?:is|was|were|are|being))\s+number\b(?!\s+(?:of|\d+)\b)", re.I), r"\1 no"),
    (re.compile(r"\b(whether\s+or|or|if|why)\s+number\b(?!\s+of\b)", re.I), r"\1 not"),
    (re.compile(r"\bnumber\s+(only|exceeding|less\s+than|more\s+than|earlier\s+than|later\s+than|below|above|to\s+apply|to\s+be|to\s+have|to\s+interfere|being|having|found)\b", re.I), r"not \1"),
    (re.compile(r"\bnumber\s+(available|maintainable|sustainable|applicable|entitled|eligible|permissible|feasible|possible|acceptable|justified|satisfied|proved|proven|guilty|valid|binding|correct|true|clear|disputed|challenged)\b", re.I), r"not \1"),
]

# Patterns for counting before/after
COMPANY_SCAN_PATTERNS = [
    re.compile(r'\bcompanyrt(s)?\b', re.I),
    re.compile(r'\bcompanynsel(s)?\b', re.I),
    re.compile(r'\bcompanyld\b', re.I),
    re.compile(r'\bcompanyntr(y|ies)\b', re.I),
    re.compile(r'\bcompanyncil(s)?\b', re.I),
    re.compile(r'\bcompanyrier(s)?\b', re.I),
    re.compile(r'\bcompanysider\w*\b', re.I),
    re.compile(r'\bcompanyclude\w*\b', re.I),
    re.compile(r'\bcompanyclusi\w*\b', re.I),
    re.compile(r'\bcompanymon(ly|wealth)?\b', re.I),
    re.compile(r'\bcompanystitut\w*\b', re.I),
    re.compile(r'\bcompany(cern|cept|cession|cord|crete|cur|duct|fer|fess|fiden|fine|firm|flict|form|found|fuse|gru|ject|nect|niv|notation|quer|scious|secut|sent|sequen|serv|sist|sol|sonan|spicuous|struct|stru|sult|sum|tain|test|text|tiguous|tinu|tort|tract|tradict|traven|tribut|trol|trovers|vene|venien|vent|vers|vert|vey|vict|vinc|vok)\w*\b', re.I),
    re.compile(r'\bcompany(mend|ment|merc|miss|mit|mod|munic|mut|pact|par|pass|pat|pel|pens|pet|pil|plain|ple|pli|ply|pos|pound|prehen|pris|promis|put)\w*\b', re.I),
    re.compile(r'\bcompany(abit|arcen|accused|al|als|b|bs|bat|bine|bination|bust|ceal|cede|ceiv)\w*\b', re.I),
]

NUMBER_SCAN_PATTERNS = [
    re.compile(r'\bnumber(ification|ified|if|withstanding|ice|iceable|able|ably|ion|ional|oriety|orious|ary|ebook)\w*\b', re.I),
    re.compile(r'\bnumbersuch\b', re.I),
    re.compile(r'\bnumber(candidate|person|witness|appeal|application|evidence|ground|reason|case|order|appointment|suit|doubt|fault|right|claim|justification)(s)?\b', re.I),
    re.compile(r'\bnumber(compliance|payment|joinder|appearance|delivery|disclosure|performance|resident|user|violence|applicant|party|agricultural|bailable|cognizable|interference|obstruction)\b', re.I),
    re.compile(r'\bnumber([A-Z][a-z]+)\b'),
    re.compile(r'\b(shall|will|would|could|should|can|may|might|must|do|does|did|is|are|was|were|has|have|had|having|need|dare|ought)\s+number\b(?!\s+(?:of|\d+|one|two|three|four|five)\b)', re.I),
    re.compile(r'\b(there\s+(?:is|was|were|are|being))\s+number\b(?!\s+(?:of|\d+)\b)', re.I),
    re.compile(r'\b(whether\s+or|or|if|why)\s+number\b(?!\s+of\b)', re.I),
    re.compile(r'\bnumber\s+(only|exceeding|less\s+than|more\s+than|earlier\s+than|later\s+than|below|above|to\s+apply|to\s+be|to\s+have|to\s+interfere|being|having|found)\b', re.I),
    re.compile(r'\bnumber\s+(available|maintainable|sustainable|applicable|entitled|eligible|permissible|feasible|possible|acceptable|justified|satisfied|proved|proven|guilty|valid|binding|correct|true|clear|disputed|challenged)\b', re.I),
]

LEGIT_PATTERNS = [
    re.compile(r'\bnumber\s+of\b', re.I),
    re.compile(r'\bthe\s+number\b', re.I),
    re.compile(r'\ba\s+number\b', re.I),
    re.compile(r'\blarge\s+number\b', re.I),
    re.compile(r'\bsmall\s+number\b', re.I),
    re.compile(r'\btotal\s+number\b', re.I),
    re.compile(r'\bany\s+number\b', re.I),
    re.compile(r'\bcase\s+number\b', re.I),
    re.compile(r'\bserial\s+number\b', re.I),
    re.compile(r'\bregistration\s+number\b', re.I),
    re.compile(r'\border\s+number\b', re.I),
    re.compile(r'\brule\s+number\b', re.I),
    re.compile(r'\bsection\s+number\b', re.I),
    re.compile(r'\bitem\s+number\b', re.I),
    re.compile(r'\bpage\s+number\b', re.I),
    re.compile(r'\bequal\s+number\b', re.I),
    re.compile(r'\bgreater\s+number\b', re.I),
    re.compile(r'\blesser\s+number\b', re.I),
    re.compile(r'\bin\s+number\b', re.I),
]

UNRESOLVED_CHECK_PATTERNS = [
    ("companyrt", re.compile(r'\bcompanyrt(s)?\b', re.I)),
    ("companynsel", re.compile(r'\bcompanynsel(s)?\b', re.I)),
    ("companysider", re.compile(r'\bcompanysider\w*\b', re.I)),
    ("companyld", re.compile(r'\bcompanyld\b', re.I)),
    ("unresolved_auxiliary_number", re.compile(r'\b(?:shall|will|would|could|should|can|may|might|must|do|does|did|is|are|was|were|has|have|had)\s+number\b(?!\s+(?:of|\d+|one|two|three|four|five)\b)', re.I)),
]


def clean_single_text(text: str) -> str:
    """Applies precompiled cleaning regexes."""
    if not text:
        return ""
    for pat, repl in COMPANY_REPLACEMENTS:
        text = pat.sub(repl, text)
    for pat, repl in NUMBER_REPLACEMENTS:
        text = pat.sub(repl, text)
    return text


def process_batch(batch_rows):
    """Worker function to process a batch of records in parallel."""
    results = []
    total_company_fixes = 0
    total_number_fixes = 0
    total_legit_before = 0
    total_legit_after = 0
    unresolved_items = []

    for row_id, raw_text, label in batch_rows:
        raw_text_str = str(raw_text) if raw_text is not None else ""
        raw_len = len(raw_text_str)

        # Count corruptions in raw
        for p in COMPANY_SCAN_PATTERNS:
            total_company_fixes += len(p.findall(raw_text_str))
        for p in NUMBER_SCAN_PATTERNS:
            total_number_fixes += len(p.findall(raw_text_str))

        # Count legitimate nouns in raw
        for p in LEGIT_PATTERNS:
            total_legit_before += len(p.findall(raw_text_str))

        # Perform cleaning
        cleaned_str = clean_single_text(raw_text_str)
        cleaned_len = len(cleaned_str)

        # Count legitimate nouns in cleaned
        for p in LEGIT_PATTERNS:
            total_legit_after += len(p.findall(cleaned_str))

        # Scan for any unresolved instances
        for label_name, p in UNRESOLVED_CHECK_PATTERNS:
            m = p.search(cleaned_str)
            if m:
                start = max(0, m.start() - 40)
                end = min(len(cleaned_str), m.end() + 60)
                snippet = cleaned_str[start:end].replace("\n", " ").strip()
                unresolved_items.append((str(row_id), label_name, snippet))
                break

        results.append({
            "id": str(row_id),
            "text_cleaned": cleaned_str,
            "text_raw_length": raw_len,
            "text_cleaned_length": cleaned_len,
            "label": int(label) if label is not None else -1
        })

    return (results, total_company_fixes, total_number_fixes,
            total_legit_before, total_legit_after, unresolved_items)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    print("=" * 80)
    print("CORPUS CLEANING PIPELINE: jayadityagandham9/ILDC_35k_COMPLETE (ALL 38,904 ROWS)")
    print("=" * 80)

    start_time = time.time()

    # 1. Load full train split
    print("[1/5] Loading full ILDC train split from Hugging Face cache...")
    ds = load_dataset("jayadityagandham9/ILDC_35k_COMPLETE", split="train")
    total_docs = len(ds)
    print(f"      Loaded {total_docs:,} judgments into memory.")

    # 2. Prepare batches for multiprocessing
    num_cpus = max(1, min(14, os.cpu_count() or 4))
    batch_size = 200
    print(f"[2/5] Distributing tasks across {num_cpus} parallel CPU worker processes (batch size: {batch_size})...")

    raw_data = [(ds[i]["id"], ds[i]["text"], ds[i]["label"]) for i in range(total_docs)]
    batches = [raw_data[i:i + batch_size] for i in range(0, total_docs, batch_size)]

    all_cleaned_rows = []
    total_company_fixes = 0
    total_number_fixes = 0
    corpus_legit_before = 0
    corpus_legit_after = 0
    all_unresolved = []

    # 3. Execute in parallel with progress bar
    print("[3/5] Cleaning text and extracting metrics across all documents...")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        for batch_res in tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing Batches"):
            rows, comp_cnt, num_cnt, legit_b, legit_a, unres = batch_res
            all_cleaned_rows.extend(rows)
            total_company_fixes += comp_cnt
            total_number_fixes += num_cnt
            corpus_legit_before += legit_b
            corpus_legit_after += legit_a
            all_unresolved.extend(unres)

    clean_duration = time.time() - start_time
    print(f"      Completed text cleaning in {clean_duration:.1f}s ({total_docs / clean_duration:.1f} docs/sec).")

    # 4. Save to Parquet
    print(f"[4/5] Saving cleaned corpus to {OUTPUT_PARQUET}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_cleaned_rows)
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    file_size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"      Successfully saved {len(df):,} rows ({file_size_mb:.2f} MB on disk).")

    # 5. Round-trip QA check
    print("[5/5] Performing round-trip read verification using Pandas...")
    df_read = pd.read_parquet(OUTPUT_PARQUET)
    
    print("\n" + "=" * 80)
    print("CORPUS PROCESSING AND QUALITY AUDIT SUMMARY")
    print("=" * 80)
    print(f"1. Total Documents Processed          : {len(df_read):,} / {total_docs:,}")
    print(f"2. Total Corruption Instances Fixed   : {total_company_fixes + total_number_fixes:,}")
    print(f"   - 'company' patterns fixed         : {total_company_fixes:,}")
    print(f"   - 'number' patterns fixed          : {total_number_fixes:,}")
    print()
    print("3. Legitimate 'number' Noun Preservation (FULL CORPUS):")
    print(f"   - Legitimate 'number' count BEFORE : {corpus_legit_before:,}")
    print(f"   - Legitimate 'number' count AFTER  : {corpus_legit_after:,}")
    pres_rate = (corpus_legit_after / corpus_legit_before * 100.0) if corpus_legit_before else 100.0
    print(f"   - Exact Preservation Rate          : {pres_rate:.2f}%")
    print()
    print(f"4. Unresolved Corruptions Scan:")
    print(f"   - Documents with unresolved tokens : {len(all_unresolved):,} ({len(all_unresolved)/total_docs*100:.2f}%)")
    if all_unresolved:
        print(f"\n   First {min(20, len(all_unresolved))} Unresolved Examples:")
        for idx, (doc_id, rule_type, snippet) in enumerate(all_unresolved[:20], 1):
            print(f"     [{idx:2d}] Doc ID: {doc_id} | Pattern: {rule_type}")
            print(f"          Snippet: \"...{snippet}...\"")
    else:
        print("   -> Clean! No unresolved raw corruption tokens found.")

    print()
    print("5. Storage & Round-Trip Verification:")
    print(f"   - Parquet File Path                : {OUTPUT_PARQUET}")
    print(f"   - Total Processing Time            : {time.time() - start_time:.2f} seconds")
    print(f"   - File Size on Disk                : {file_size_mb:.2f} MB")
    print(f"   - Saved Row Count                  : {len(df_read):,}")
    print("\nDataFrame Info:")
    df_read.info()

    print("\nFirst Row Preview (text_cleaned, first 500 chars):")
    print("-" * 80)
    first_cleaned_text = str(df_read.iloc[0]["text_cleaned"])
    print(first_cleaned_text[:500] + ("..." if len(first_cleaned_text) > 500 else ""))
    print("-" * 80)
    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
