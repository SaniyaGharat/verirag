# VeriRAG: Verified Retrieval-Augmented Generation for Legal NLP

VeriRAG is a framework for verified, verifiable retrieval-augmented generation applied to legal documents, focusing on the Indian Legal Documents Corpus (ILDC). It deconstructs generated legal answers into atomic claims, retrieves source citations, independently verifies each claim against retrieved evidence, and aggregates claim-level verifications into an overall document/judgment trust score.

## Directory Structure

```
verirag/
  data/         # Dataset loaders, preprocessors, and storage (e.g., ILDC)
  retriever/    # Dense/sparse retrieval and indexing (FAISS, Sentence-Transformers)
  generator/    # Answer and legal judgment generation modules
  decomposer/   # Claim extraction and atomic decomposition
  verifier/     # Entailment, contradiction, and factual verification models
  aggregator/   # Confidence aggregation, citation linking, and scoring
  eval/         # Evaluation benchmarks and validation scripts
  notebooks/    # Exploratory data analysis and prototyping
  configs/      # Configuration files (retriever, models, chunking)
  README.md     # Project overview and instructions
  definitions.md# Operational definitions of atomic claims, evidence, and verdicts
```

## Setup

1. Activate virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Test ILDC dataset loading:
   ```powershell
   python data/load_ildc.py
   ```

## Architecture Overview

VeriRAG executes a 4-stage pipeline for legal hallucination detection and provenance tracking:

1. **Decomposer (`decomposer/`)**: Deconstructs complex legal text into atomic, self-contained factual/legal propositions using LLM prompting or deterministic boundary heuristics.
2. **Retriever (`retriever/`)**: Dense passage retrieval (FAISS Flat-IP on `all-MiniLM-L6-v2`) and sparse lexical retrieval (BM25Okapi) across cleaned ILDC court judgments.
3. **Verifier (`verifier/`)**: Claim-level Natural Language Inference using fine-tuned Cross-Encoder (`cross-encoder/nli-deberta-v3-base`) augmented with deterministic legal-reference validation for statutory sections and court hierarchies.
4. **Aggregator (`aggregator/`)**: Calibrated document trust score calculation, contradiction penalty, citation span linking, and audit-ready Markdown report generation.

---

## Quickstart & Pipeline Usage

### 1. Run End-to-End Verification
Verify an arbitrary legal claim or model response against the ILDC corpus:
```powershell
python pipeline.py "The appellant, convicted under Section 302 IPC, was sentenced to life imprisonment, but was acquitted by the High Court."
```

Optional arguments:
- `--query`: Context query (e.g., `"What was the outcome of the appeal?"`)
- `--top-k`: Number of retrieved evidence passages per claim (default: 5)
- `--decompose-method`: `auto`, `llm`, or `rule_based`

### 2. Search Retrieval Index
Directly query the FAISS dense and BM25 sparse index:
```powershell
python retriever/search.py --query "quashing of FIR under Section 482 CrPC" --top-k 5
```

### 3. Verify Claim Pair Directly
```powershell
python verifier/nli_verify.py --claim "The High Court acquitted the appellant." --evidence "The High Court set aside the conviction and acquitted the appellant of all charges."
```

---

## VeriRAG-Bench Benchmark & Evaluation

### Benchmark Dataset (`data/benchmark/`)
`VeriRAG-Bench` contains 300 gold-standard legal claim–evidence triples drawn from Indian Supreme Court and High Court judgments, partitioned with **strict zero-leakage document isolation**:
- `data/benchmark/verirag_bench.jsonl`: Full benchmark (300 triples across 7 claim categories)
- `data/benchmark/dev.jsonl`: Dev split (107 triples, 27 distinct judgments)
- `data/benchmark/test.jsonl`: Test split (193 triples, 52 distinct judgments)

Label distribution: Balanced across `{SUPPORTED: 100, CONTRADICTED: 100, UNSUPPORTED: 100}`.

To regenerate the benchmark:
```powershell
python eval/build_benchmark.py
```

### Run Experimental Evaluation Suite
Executes Experiments A through D on the test split and compiles the published evaluation report:
```powershell
python eval/evaluate_verifier.py
```

Outputs generated in `eval/results/`:
- `experiment_a_baselines.json`: Cross-Encoder NLI vs. ROUGE-L, BM25, and Cosine Embedding
- `experiment_b_legal_checks.json`: Ablation of deterministic statutory reference checks
- `experiment_c_ablations.json`: 4-way pipeline component ablation
- `experiment_d_robustness.json`: Graceful degradation on distractor, truncated, and empty evidence
- `EVALUATION_REPORT.md`: Formatted consolidated experimental report

---

## Experimental Findings Summary

| System / Configuration | Macro-F1 | Accuracy | False-Supported Rate (Safety ↓) | Contradiction Recall ↑ |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline ROUGE-L** | 0.1872 | 30.57% | 2.42% | 0.00% |
| **Baseline BM25** | 0.1478 | 28.50% | 0.00% | 0.00% |
| **Baseline Cosine Similarity** | 0.3849 | 47.67% | 24.19% | 0.00% |
| **VeriRAG Cross-Encoder NLI** | **0.5343** | **53.89%** | **4.03%** | **60.87%** |

- **Safety Advantage**: While surface lexical and bi-encoder methods suffer from high false-support or zero contradiction detection, VeriRAG successfully captures **60.87% of legal contradictions** while maintaining a low False-Supported Rate (4.03%).
- **Adversarial Robustness**: Under unrelated distractor passages, VeriRAG resists false support with **95.7% rejection rate**, shifting cleanly to `UNSUPPORTED (NEI)`.
