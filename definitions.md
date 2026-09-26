# VeriRAG: Operational Definitions

This document defines the foundational standards for claim decomposition, verification labeling, and evidence validity within the VeriRAG framework.

---

## 1. What Counts as an "Atomic Claim"

An **atomic claim** is defined as an indivisible, self-contained declarative proposition that asserts a single, discrete factual, procedural, or legal statement. It cannot be subdivided into smaller propositions without losing coherent semantic meaning, and it must be independently verifiable or falsifiable against source text.

### Key Criteria for Atomicity:
1. **Single Predicate**: The proposition links a single subject to a single action, property, status, or legal outcome.
2. **Context Independence & Coreference Resolution**: All pronouns ("he", "she", "it"), deictic expressions ("this court", "the impugned order"), and relative titles ("the appellant", "the petitioner") must be resolved to their specific named entities, case parties, or docket references.
3. **Decompounding**: Compound sentences connected by conjunctions ("and", "or", "while", "furthermore") or subordinate clauses ("after which", "alleging that") must be split into separate individual claims.

### Concrete Examples from Legal Text:

#### Example 1: Procedural History & Outcome
* **Original Complex Sentence**:
  > *"The appellant, convicted under Section 302 of the Indian Penal Code by the Sessions Court, was sentenced to life imprisonment, but was subsequently acquitted by the High Court due to material discrepancies in witness testimony."*
* **Decomposed Atomic Claims**:
  1. `Claim 1.1`: The appellant was convicted by the Sessions Court under Section 302 of the Indian Penal Code.
  2. `Claim 1.2`: The Sessions Court sentenced the appellant to life imprisonment.
  3. `Claim 1.3`: The High Court acquitted the appellant.
  4. `Claim 1.4`: The High Court's acquittal was based on material discrepancies in witness testimony.

#### Example 2: Supreme Court Holding & Constitutional Interpretation
* **Original Complex Sentence**:
  > *"The Supreme Court dismissed the appeal and reaffirmed that the right to privacy is guaranteed under Article 21, subject to legitimate state interests."*
* **Decomposed Atomic Claims**:
  1. `Claim 2.1`: The Supreme Court dismissed the appeal.
  2. `Claim 2.2`: The Supreme Court reaffirmed that the right to privacy is guaranteed under Article 21 of the Constitution.
  3. `Claim 2.3`: The right to privacy under Article 21 is subject to legitimate state interests.

#### Example 3: Commercial & Arbitration Dispute
* **Original Complex Sentence**:
  > *"The sole arbitrator delivered an award of INR 4.5 crores against the respondent on 15th January 2020, prompting the respondent to file a petition under Section 34 of the Arbitration and Conciliation Act citing lack of proper notice."*
* **Decomposed Atomic Claims**:
  1. `Claim 3.1`: The sole arbitrator delivered an award of INR 4.5 crores against the respondent.
  2. `Claim 3.2`: The arbitral award was delivered on 15th January 2020.
  3. `Claim 3.3`: The respondent filed a petition under Section 34 of the Arbitration and Conciliation Act.
  4. `Claim 3.4`: The respondent's Section 34 petition cited lack of proper notice.

---

## 2. Operational Definitions of Verification Labels

Each atomic claim is evaluated against the set of retrieved evidence passages $E = \{e_1, e_2, \dots, e_k\}$ and assigned one of three mutually exclusive labels:

### 2.1 Supported (`SUPPORTED`)
* **Definition**: The retrieved evidence directly and unambiguously confirms the factual or legal truth of the atomic claim via strict textual entailment or direct identity.
* **Operational Rule**: A rational human judge or verified NLI model can infer the truth of the claim from $E$ without requiring external unstated assumptions or speculative leaps. Every essential entity, date, numeric value, and legal holding in the claim matches the evidence.
* **Example**:
  * *Claim*: "The High Court set aside the conviction of the appellant."
  * *Evidence*: "In the result, the conviction recorded by the trial court is set aside and the appellant is acquitted of all charges."
  * *Verdict*: `SUPPORTED`

### 2.2 Contradicted (`CONTRADICTED`)
* **Definition**: The retrieved evidence directly refutes or negates the factual or legal truth of the atomic claim, presenting a mutually exclusive reality.
* **Operational Rule**: If the claim asserts fact $X$, and the retrieved evidence explicitly asserts $\neg X$ or a mutually inconsistent state $Y$ (e.g., dates mismatch, outcome reversed, parties inverted).
* **Example**:
  * *Claim*: "The Supreme Court dismissed the special leave petition."
  * *Evidence*: "Leave granted. The appeal is allowed and the impugned judgment of the High Court is quashed."
  * *Verdict*: `CONTRADICTED`

### 2.3 Unsupported (`UNSUPPORTED`)
* **Definition**: The retrieved evidence is silent, indeterminate, or insufficient to prove or disprove the atomic claim.
* **Operational Rule**: Assigned when:
  1. The retrieved passages make no mention of the asserted entities or facts.
  2. The evidence contains partial overlap but lacks critical elements required to confirm the claim (e.g., mentions the appeal, but omits whether it was dismissed or allowed).
  3. The evidence is ambiguous or requires ungrounded extrapolation.
* **Example**:
  * *Claim*: "The appellant filed the review petition within the statutory period of 30 days."
  * *Evidence*: "The review petition was filed before the Registrar on 12th October 2019." *(Silent on date of initial order, making timeliness unverifiable)*
  * *Verdict*: `UNSUPPORTED`

---

## 3. What Counts as Valid Evidence

In VeriRAG, not all retrieved text qualifies as valid evidence. Valid evidence must meet four strict criteria:

1. **Grounded Provenance**:
   - Evidence must be extracted from verified source documents within the corpus (e.g., Indian Legal Documents Corpus / ILDC judgments, statutes, or official case briefs).
   - Every piece of evidence must retain explicit metadata: `document_id`, `chunk_id`, `court_name`, `decision_date`, and `source_url`/`citation`.

2. **Semantic Self-Sufficiency**:
   - The excerpted passage must contain enough local context to be intelligible on its own. Truncated spans, sentence fragments missing core referents, or corrupted OCR chunks cannot serve as valid evidence.

3. **Temporal & Docket Identity**:
   - The evidence must strictly belong to the same legal matter, docket, or bench proceeding referenced in the claim. A statement from an unrelated case sharing the same party name (e.g., "State of Maharashtra") is invalid evidence.

4. **Operative Distinction (Holding vs. Obiter / Submissions)**:
   - For claims concerning the *court's ruling or decision* (such as CJPE labels: accepted vs. rejected), the evidence must be drawn from the operative portion or *ratio decidendi* of the judgment.
   - Text summarizing the arguments of counsel (e.g., *"Learned counsel for the respondent contended that..."*) constitutes valid evidence **only** if the claim specifically asserts what counsel argued, not what the court held.
