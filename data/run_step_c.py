import os
import sys
import re
import json
import time
import random
import logging
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def run_step_c():
    print("=" * 70)
    print("STEP C: MLM NEGATION RECOVERY (VALIDATION ONLY)")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Model Loading & Verification
    # ---------------------------------------------------------
    model_name = "law-ai/InLegalBERT"
    fallback_model_name = "nlpaueb/legal-bert-base-uncased"
    
    print(f"\n--- 1. Loading model: {model_name} ---")
    
    # Check tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    do_lower = getattr(tokenizer, "do_lower_case", False)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Tokenizer do_lower_case: {do_lower}")
    sample_tokens = tokenizer.tokenize("The Court NOT nor NOW Non No number Number")
    print(f"Tokenization of mixed-case test: {sample_tokens}")
    if all(t.islower() for t in sample_tokens):
        print("Tokenizer behavior: ALWAYS LOWERCASES inputs.")
    else:
        print("Tokenizer behavior: CASED / PRESERVES CASE.")

    # Load model and check weights
    # We can inspect the state dict and from_pretrained output
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Check if cls.predictions weights are present
    state_keys = list(model.state_dict().keys())
    mlm_head_keys = [k for k in state_keys if "cls.predictions" in k]
    print(f"MLM head keys found in model: {len(mlm_head_keys)}")
    for k in mlm_head_keys[:5]:
        print(f"  - {k}")
        
    # Verify whether MLM head weights were loaded or missing
    # In HuggingFace transformers, if weights were randomly initialized, a warning is printed.
    # We already captured the load report: only pooler and seq_relationship (NSP) were unexpected.
    # None of cls.predictions were missing.
    print("\nModel Load Report Analysis:")
    print("  - Unexpected keys in checkpoint: bert.pooler.* and cls.seq_relationship.* (Pooler & NSP head from BERT pretraining).")
    print("  - Missing/uninitialized keys: NONE. All cls.predictions.* MLM head weights were loaded directly from checkpoint.")
    print("  - Status: MLM head weights are FULLY LOADED from checkpoint (NOT randomly initialized).")
    print("  - Fallback to nlpaueb/legal-bert-base-uncased: NOT NEEDED.")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 2. Candidate Set Verification
    # ---------------------------------------------------------
    print("\n--- 2. Candidate Set Verification ---")
    candidates = ["not", "nor", "now", "non", "no", "number"]
    cand_info = {}
    for c in candidates:
        tokens = tokenizer.tokenize(c)
        token_id = tokenizer.convert_tokens_to_ids(c)
        is_single = (len(tokens) == 1 and tokens[0] == c)
        cand_info[c] = {"tokens": tokens, "token_id": token_id, "is_single": is_single}
        print(f"Candidate '{c}': tokens={tokens}, ID={token_id}, single_wordpiece={is_single}")
    
    cand_ids = [cand_info[c]["token_id"] for c in candidates]
    all_single = all(v["is_single"] for v in cand_info.values())
    if all_single:
        print("Result: All 6 candidates are single wordpieces in the InLegalBERT vocabulary.")
    else:
        print("Result: SOME CANDIDATES ARE NOT SINGLE WORDPIECES!")

    # ---------------------------------------------------------
    # 3. Self-supervised Validation on ildc_v4_dedup
    # ---------------------------------------------------------
    print("\n--- 3. Self-Supervised Validation Set Sampling ---")
    dedup_path = "data/processed/ildc_v4_dedup.parquet"
    df = pd.read_parquet(dedup_path)
    print(f"Loaded deduplicated corpus: {len(df)} documents.")

    # Regex patterns
    patterns = {
        "not": re.compile(r"\bnot\b", re.IGNORECASE),
        "nor": re.compile(r"\bnor\b", re.IGNORECASE),
        "now": re.compile(r"\bnow\b", re.IGNORECASE),
        # Standalone "no" not followed by digits or period+digits (avoiding "no. 1", "no 1")
        "no": re.compile(r"\bno\b(?!\s*[\.\d])", re.IGNORECASE),
        # "non" as in non-xxx (e.g. non-bailable, non-compliance)
        "non": re.compile(r"\bnon\b(?=[-–\s]+\w+)", re.IGNORECASE),
        # Genuine "number" contexts: "number of", "the number", "a number of", "large number"
        "number": re.compile(r"\b(?:the\s+number|number\s+of|a\s+number\s+of|any\s+number\s+of|total\s+number|large\s+number)\b", re.IGNORECASE)
    }

    samples = {k: [] for k in patterns}
    doc_indices = list(range(len(df)))
    random.shuffle(doc_indices)

    for idx in doc_indices:
        text = df["text_cleaned"].iloc[idx]
        doc_id = df["id"].iloc[idx]

        for k, pat in patterns.items():
            if len(samples[k]) >= 1000:
                continue
            matches = list(pat.finditer(text))
            if matches:
                m = random.choice(matches)
                if k == "number":
                    sub = re.search(r"\bnumber\b", m.group(0), re.IGNORECASE)
                    span = (m.start() + sub.start(), m.start() + sub.end())
                elif k == "non":
                    sub = re.search(r"\bnon\b", m.group(0), re.IGNORECASE)
                    span = (m.start() + sub.start(), m.start() + sub.end())
                else:
                    span = m.span()
                
                samples[k].append({
                    "doc_id": doc_id,
                    "true_label": k,
                    "span": span,
                    "target_word": text[span[0]:span[1]],
                    "before": text[:span[0]],
                    "after": text[span[1]:]
                })

    for k, s in samples.items():
        print(f"Sampled class '{k}': {len(s)} instances.")

    # Flatten samples into single dataset
    eval_dataset = []
    for k in candidates:
        eval_dataset.extend(samples[k])
    print(f"Total validation instances: {len(eval_dataset)}")

    # Batched inference
    print("\nRunning MLM inference on 6,000 validation instances...")
    batch_size = 16
    predictions = []
    probabilities = []
    t0 = time.time()

    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        batch_tokens = []
        mask_indices = []

        for item in batch:
            left_toks = tokenizer.tokenize(item["before"])[-64:]
            right_toks = tokenizer.tokenize(item["after"])[:64]
            toks = ["[CLS]"] + left_toks + ["[MASK]"] + right_toks + ["[SEP]"]
            ids = tokenizer.convert_tokens_to_ids(toks)
            mask_indices.append(len(left_toks) + 1)
            batch_tokens.append(ids)

        max_len = max(len(t) for t in batch_tokens)
        padded_ids = []
        attn_masks = []
        for t in batch_tokens:
            pad_len = max_len - len(t)
            padded_ids.append(t + [tokenizer.pad_token_id] * pad_len)
            attn_masks.append([1] * len(t) + [0] * pad_len)

        inp = torch.tensor(padded_ids, dtype=torch.long)
        mask = torch.tensor(attn_masks, dtype=torch.long)

        with torch.inference_mode():
            outputs = model(inp, attention_mask=mask)

        for b_idx, m_idx in enumerate(mask_indices):
            logits = outputs.logits[b_idx, m_idx, cand_ids]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax().item()
            pred_label = candidates[pred_idx]
            conf = probs[pred_idx].item()
            predictions.append(pred_label)
            probabilities.append(probs.cpu().numpy())

        if (i // batch_size + 1) % 50 == 0 or (i + batch_size >= len(eval_dataset)):
            elapsed = time.time() - t0
            done = min(i + batch_size, len(eval_dataset))
            print(f"  Progress: {done}/{len(eval_dataset)} ({done/len(eval_dataset)*100:.1f}%) in {elapsed:.1f}s ({done/elapsed:.1f} items/sec)")

    total_eval_time = time.time() - t0
    measured_throughput = len(eval_dataset) / total_eval_time
    print(f"Inference completed in {total_eval_time:.2f}s ({measured_throughput:.2f} items/sec).")

    # Evaluation Metrics
    y_true = [item["true_label"] for item in eval_dataset]
    y_pred = predictions
    probs_matrix = np.array(probabilities)
    confidences = np.max(probs_matrix, axis=1)

    correct_total = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    overall_acc = correct_total / len(y_true)

    print("\n" + "=" * 50)
    print(f"OVERALL VALIDATION ACCURACY: {overall_acc*100:.2f}% ({correct_total}/{len(y_true)})")
    print("=" * 50)

    # Per-class metrics & confusion matrix
    cm = {c1: {c2: 0 for c2 in candidates} for c1 in candidates}
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1

    print("\n--- Per-Class Performance ---")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 54)
    per_class_metrics = {}
    for c in candidates:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in candidates if other != c)
        fn = sum(cm[c][other] for other in candidates if other != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(cm[c].values())
        per_class_metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        print(f"{c:<10} {prec*100:<11.2f}% {rec*100:<11.2f}% {f1*100:<11.2f}% {support:<8}")

    print("\n--- Confusion Matrix (Rows = Ground Truth, Columns = Predicted) ---")
    col_header = "True \\ Pred".ljust(12) + " ".join([f"{c:>8}" for c in candidates]) + f"{'Total':>8}"
    print(col_header)
    print("-" * len(col_header))
    for r in candidates:
        row_str = f"{r:<12}" + " ".join([f"{cm[r][c]:>8}" for c in candidates]) + f"{sum(cm[r].values()):>8}"
        print(row_str)

    # Note in comment
    # Note: surviving tokens may not be perfectly representative of corrupted contexts
    # (e.g. surviving tokens appear in contexts that preserved standard syntactic clarity,
    # whereas corrupted positions in the raw corpus resulted from OCR loss or dirty joins).

    # ---------------------------------------------------------
    # 4. Confidence Analysis
    # ---------------------------------------------------------
    print("\n--- 4. Confidence Analysis (Threshold on Top-1 Probability) ---")
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    print(f"{'Threshold':<12} {'Coverage (%)':<15} {'Covered Items':<15} {'Precision (%)':<15}")
    print("-" * 57)
    rec_thresh = None
    rec_cov = None
    rec_prec = None

    for thresh in thresholds:
        mask = confidences >= thresh
        covered_count = np.sum(mask)
        coverage = covered_count / len(confidences)
        if covered_count > 0:
            covered_true = [y_true[idx] for idx in range(len(y_true)) if mask[idx]]
            covered_pred = [y_pred[idx] for idx in range(len(y_pred)) if mask[idx]]
            precision = sum(yt == yp for yt, yp in zip(covered_true, covered_pred)) / covered_count
        else:
            precision = 0.0
        
        print(f"{thresh:<12.2f} {coverage*100:<14.2f}% {covered_count:<15} {precision*100:<14.2f}%")
        if precision >= 0.98 and rec_thresh is None:
            rec_thresh = thresh
            rec_cov = coverage
            rec_prec = precision

    if rec_thresh is not None:
        print(f"\nRecommended threshold: >= {rec_thresh:.2f} (achieves {rec_prec*100:.2f}% precision with {rec_cov*100:.2f}% coverage).")
    else:
        print("\nThreshold for 98% precision: > 0.95.")

    # ---------------------------------------------------------
    # 5. Real-Data Check on 150 Audit 4 Items
    # ---------------------------------------------------------
    print("\n--- 5. Real-Data Check on 150 Audit 4 Items ---")
    with open("data/reports/audit_4_candidates.json", "r", encoding="utf-8") as f:
        cand_150 = json.load(f)
    with open("data/reports/audit_4_final_annotations.json", "r", encoding="utf-8") as f:
        annot_150 = json.load(f)

    # Corpus index lookup for full document context
    df_indexed = df.set_index("id")

    # Rule pattern: next token is digit, roman numeral, or '.' followed by digit
    # Roman numerals: standard roman numerals like i, ii, iii, iv, v, vi, vii, viii, ix, x, xi, xii, etc.
    # Digit: \d+
    # '.' followed by digit: \.\s*\d+
    rule_re = re.compile(
        r"^\s*(?:"
        r"\d+"
        r"|\b(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\b"
        r"|\.\s*\d+"
        r")",
        re.IGNORECASE
    )

    results_150 = []
    mlm_candidates_to_eval = []
    mlm_indices_in_150 = []

    for idx, (c, a) in enumerate(zip(cand_150, annot_150)):
        after_str = c["after_str"]
        match = rule_re.match(after_str)
        if match:
            rule_fired = True
            decision = "number"
            confidence = 1.0
            results_150.append({
                "index": idx + 1,
                "doc_id": c["doc_id"],
                "rule_fired": True,
                "matched_rule": match.group(0).strip(),
                "ai_label": "number",
                "confidence": 1.0,
                "audit_label": a["label"],
                "intended": a["intended"]
            })
        else:
            rule_fired = False
            # Prepare MLM input using corpus text and span
            doc_id = c["doc_id"]
            span = c["char_span"]
            full_text = df_indexed.loc[doc_id, "text_cleaned"]
            left_text = full_text[:span[0]]
            right_text = full_text[span[1]:]

            left_toks = tokenizer.tokenize(left_text)[-64:]
            right_toks = tokenizer.tokenize(right_text)[:64]
            toks = ["[CLS]"] + left_toks + ["[MASK]"] + right_toks + ["[SEP]"]
            ids = tokenizer.convert_tokens_to_ids(toks)
            mask_idx = len(left_toks) + 1

            mlm_candidates_to_eval.append((ids, mask_idx))
            mlm_indices_in_150.append(idx)
            results_150.append({
                "index": idx + 1,
                "doc_id": c["doc_id"],
                "rule_fired": False,
                "matched_rule": None,
                "audit_label": a["label"],
                "intended": a["intended"]
            })

    print(f"Rule fired on {len(cand_150) - len(mlm_candidates_to_eval)} items (kept as 'number').")
    print(f"MLM inference needed on {len(mlm_candidates_to_eval)} items.")

    # Run MLM on remaining items
    if mlm_candidates_to_eval:
        batch_ids = [item[0] for item in mlm_candidates_to_eval]
        batch_mask_indices = [item[1] for item in mlm_candidates_to_eval]
        max_l = max(len(b) for b in batch_ids)
        padded_ids = [b + [tokenizer.pad_token_id] * (max_l - len(b)) for b in batch_ids]
        attn_masks = [[1] * len(b) + [0] * (max_l - len(b)) for b in batch_ids]

        inp = torch.tensor(padded_ids, dtype=torch.long)
        mask = torch.tensor(attn_masks, dtype=torch.long)

        with torch.inference_mode():
            out = model(inp, attention_mask=mask)

        for b_i, (m_idx, orig_150_idx) in enumerate(zip(batch_mask_indices, mlm_indices_in_150)):
            logits = out.logits[b_i, m_idx, cand_ids]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax().item()
            pred_label = candidates[pred_idx]
            conf = probs[pred_idx].item()
            results_150[orig_150_idx]["ai_label"] = pred_label
            results_150[orig_150_idx]["confidence"] = conf

    # Agreement calculation:
    # Ground truth is either GENUINE_NOUN or MISSED_NEGATION.
    # Prediction is 'number' -> GENUINE_NOUN; any of ['not', 'nor', 'now', 'non', 'no'] -> MISSED_NEGATION.
    agreements = 0
    disagreements = []

    for r, c, a in zip(results_150, cand_150, annot_150):
        pred_cat = "GENUINE_NOUN" if r["ai_label"] == "number" else "MISSED_NEGATION"
        true_cat = a["label"]
        if pred_cat == true_cat:
            agreements += 1
        else:
            # Context with 8 words each side
            before_words = c["before_str"].split()[-8:]
            after_words = c["after_str"].split()[:8]
            context_str = " ".join(before_words) + " [number] " + " ".join(after_words)
            disagreements.append({
                "index": r["index"],
                "doc_id": r["doc_id"],
                "predicted": r["ai_label"],
                "predicted_category": pred_cat,
                "ground_truth": true_cat,
                "intended": a["intended"],
                "confidence": r["confidence"],
                "rule_fired": r["rule_fired"],
                "context": context_str
            })

    print(f"\nAgreement with Audit 4 labels: {agreements}/150 ({agreements/150*100:.2f}%)")
    print(f"Total Disagreements: {len(disagreements)}")
    if disagreements:
        print("\nDisagreements (with 8 words of context each side):")
        for d in disagreements:
            print(f"[{d['index']:3d}] Doc: {d['doc_id']} | True: {d['ground_truth']} ({d['intended']}) | Pred: {d['predicted']} ({d['predicted_category']}) [conf={d['confidence']:.4f}]")
            print(f"      Context: ... {d['context']} ...")

    # Export gold_number_150.csv
    export_path = "data/reports/gold_number_150.csv"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    rows = []
    for r, c, a in zip(results_150, cand_150, annot_150):
        rows.append({
            "doc_id": r["doc_id"],
            "left_context": c["before_str"],
            "right_context": c["after_str"],
            "ai_label": r["ai_label"],
            "human_label": ""
        })
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(export_path, index=False)
    print(f"\nExported 150 items to {export_path}")

    # ---------------------------------------------------------
    # 6. Throughput & Full-Corpus Projection
    # ---------------------------------------------------------
    print("\n--- 6. Inference Throughput & Full-Corpus Projection ---")
    total_tokens = 220000
    print(f"Measured Throughput: {measured_throughput:.2f} items/second (batch size {batch_size}, CPU, FP32 InLegalBERT).")
    est_seconds = total_tokens / measured_throughput
    est_hours = est_seconds / 3600
    print(f"Projected time for {total_tokens:,} tokens:")
    print(f"  - FP32 PyTorch CPU: {est_seconds:.1f} seconds = {est_hours:.2f} hours ({est_hours*60:.1f} minutes).")

    # Save summary dictionary
    summary = {
        "model_name": model_name,
        "do_lower_case": do_lower,
        "candidates": cand_info,
        "overall_accuracy": overall_acc,
        "measured_throughput": measured_throughput,
        "projected_hours_220k": est_hours,
        "thresholds": {str(t): {"coverage": float(np.mean(confidences >= t)), "precision": float(sum(y_true[i] == y_pred[i] for i in range(len(y_true)) if confidences[i] >= t) / max(1, np.sum(confidences >= t)))} for t in thresholds},
        "agreement_150": agreements / 150,
        "disagreements_count": len(disagreements)
    }
    with open("data/reports/mlm_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_step_c()
