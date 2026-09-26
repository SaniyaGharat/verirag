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
