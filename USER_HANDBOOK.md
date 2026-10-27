# VeriRAG User Handbook

This guide explains how to run VeriRAG, enter a legal response for checking, read its results, and try known benchmark examples.

## Start the app

Install the project dependencies, then start the local server from the project folder:

```powershell
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000` in your browser. The retrieval index and model files must already be available; see the project README for corpus preparation and index-building steps.

## Verify a legal response

In **Verification Studio**, fill in:

1. **Legal Context Query** — optional context for the report. The current pipeline retrieves passages from each decomposed claim, so this field does not change retrieval.
2. **Generated Legal Response / Claims** — the text you want checked. Use short, factual statements when possible. The system splits compound text into individual claims.
3. **Passages per Claim** — choose 3, 5, or 10 retrieved passages.
4. **Decomposition Mode** — Rule-Based is the predictable default. Auto uses LLM decomposition only when configured and available.

Click **Execute Legal Verification**. Results show each claim’s verdict, confidence, evidence passages, and the counts of supported, contradicted, and unsupported claims. The overall trust score has been removed from the UI.

### Ready-to-test supported example

**Legal Context Query**

> How much compensation did the High Court award per acre, and what solatium was granted?

**Generated Legal Response / Claims**

> The High Court enhanced the land compensation to Rs. 4,50,000 per acre with 30% solatium.

This is labeled `SUP` in the benchmark (`doc_id`: `2020_15`, `chunk_id`: `2020_15_c4`). Other supported examples are in `data/benchmark/test.jsonl`.

## Understanding verdicts

- **SUPPORTED** — retrieved evidence supports the claim.
- **CONTRADICTED** — evidence from the selected source judgment directly conflicts with the claim.
- **UNSUPPORTED** — evidence is missing, incomplete, or insufficient to decide.

The verifier does not combine evidence across different judgments. Without a specified source judgment, it uses the highest-ranked retrieved judgment as the candidate source. For a precise docket-specific check, pass `source_doc_id` through the API or command line.

## Search the corpus

Use the **Corpus Explorer** to search for evidence passages. Select **Dense (FAISS)**, **Sparse (BM25)**, or **Hybrid Search**. Search results include passage text, source judgment, and retrieval rank/score.

## API examples

### Docket-scoped verification

`POST /api/verify` accepts `text`, optional `query`, `top_k`, `decompose_method`, and optional `source_doc_id`:

```json
{
  "query": "How much compensation did the High Court award?",
  "text": "The High Court enhanced the land compensation to Rs. 4,50,000 per acre with 30% solatium.",
  "top_k": 5,
  "decompose_method": "rule_based",
  "source_doc_id": "2020_15"
}
```

### Corpus search

`POST /api/search` accepts `query`, optional `top_k`, and `mode` (`dense`, `sparse`, or `hybrid`).

## Troubleshooting unsupported results

- Check that the claim is stated clearly and contains one main fact.
- Use a query with the important entities, outcome, and statute when searching the Corpus Explorer.
- If you know the judgment ID, provide it as `source_doc_id` in the API or use `--source-doc-id` with `pipeline.py`.
- Review the returned evidence. An unsupported verdict can be correct when the retrieved passage does not establish the claim.

## Run the regression tests

From the project folder, run:

```powershell
python -m unittest discover -s tests -v
```

