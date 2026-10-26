"""
VeriRAG Evaluation Metrics
==========================
Computes standard NLI classification metrics and legal-domain safety metrics:
1. Macro-F1 across {SUP, CON, NEI}
2. Per-class Precision, Recall, F1-score
3. False-Supported Rate (FSR): Safety-critical metric measuring percentage of
   non-supported claims (CON or NEI) erroneously marked as SUPPORTED.
4. Confusion Matrix and classification report.
"""

from typing import List, Dict, Any, Union
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

LABEL_MAP = {
    "SUP": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "CON": "CONTRADICTED",
    "CONTRADICTED": "CONTRADICTED",
    "NEI": "UNSUPPORTED",
    "UNSUPPORTED": "UNSUPPORTED"
}

STANDARD_LABELS = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]


def normalize_label(label: str) -> str:
    """Normalizes label string to canonical {SUPPORTED, CONTRADICTED, UNSUPPORTED}."""
    return LABEL_MAP.get(str(label).strip().upper(), "UNSUPPORTED")


def compute_verification_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Any]:
    """
    Computes comprehensive verification metrics given lists of true and predicted labels.
    """
    y_t = [normalize_label(lbl) for lbl in y_true]
    y_p = [normalize_label(lbl) for lbl in y_pred]

    # Global accuracy
    acc = accuracy_score(y_t, y_p)

    # Per-class and Macro metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_t, y_p, labels=STANDARD_LABELS, zero_division=0
    )

    macro_p = float(np.mean(p_per))
    macro_r = float(np.mean(r_per))
    macro_f1 = float(np.mean(f1_per))

    # Confusion matrix: rows=True, cols=Pred
    cm = confusion_matrix(y_t, y_p, labels=STANDARD_LABELS).tolist()

    # Safety Metric: False-Supported Rate (FSR)
    # FSR = Number of non-supported claims falsely predicted as SUPPORTED / Total non-supported claims
    non_supported_count = sum(1 for t in y_t if t in ("CONTRADICTED", "UNSUPPORTED"))
    false_supported_count = sum(
        1 for t, p in zip(y_t, y_p)
        if t in ("CONTRADICTED", "UNSUPPORTED") and p == "SUPPORTED"
    )
    fsr = (false_supported_count / non_supported_count) if non_supported_count > 0 else 0.0

    # Contradiction Detection Recall (sensitivity to actual hallucinations/errors)
    con_total = sum(1 for t in y_t if t == "CONTRADICTED")
    con_caught = sum(1 for t, p in zip(y_t, y_p) if t == "CONTRADICTED" and p == "CONTRADICTED")
    con_recall = (con_caught / con_total) if con_total > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "false_supported_rate": round(float(fsr), 4),
        "contradiction_recall": round(float(con_recall), 4),
        "per_class": {
            lbl: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s)
            }
            for lbl, p, r, f, s in zip(STANDARD_LABELS, p_per, r_per, f1_per, sup_per)
        },
        "confusion_matrix": {
            "labels": STANDARD_LABELS,
            "matrix": cm
        },
        "sample_counts": {
            "total": len(y_t),
            "true_supported": sum(1 for t in y_t if t == "SUPPORTED"),
            "true_contradicted": sum(1 for t in y_t if t == "CONTRADICTED"),
            "true_unsupported": sum(1 for t in y_t if t == "UNSUPPORTED")
        }
    }


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Formats metrics as a clean Markdown table."""
    lines = [
        f"### {title}",
        "",
        f"- **Accuracy**: `{metrics['accuracy']:.2%}`",
        f"- **Macro-F1**: `{metrics['macro_f1']:.4f}`",
        f"- **False-Supported Rate (Safety)**: `{metrics['false_supported_rate']:.2%}`",
        f"- **Contradiction Recall**: `{metrics['contradiction_recall']:.2%}`",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for lbl, vals in metrics["per_class"].items():
        lines.append(
            f"| **{lbl}** | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} | {vals['support']} |"
        )

    lines.extend([
        "",
        "**Confusion Matrix** (Rows: True, Cols: Pred):",
        "```",
        f"               Pred_SUP  Pred_CON  Pred_NEI",
        f"True_SUP        {metrics['confusion_matrix']['matrix'][0][0]:<9}{metrics['confusion_matrix']['matrix'][0][1]:<9}{metrics['confusion_matrix']['matrix'][0][2]:<9}",
        f"True_CON        {metrics['confusion_matrix']['matrix'][1][0]:<9}{metrics['confusion_matrix']['matrix'][1][1]:<9}{metrics['confusion_matrix']['matrix'][1][2]:<9}",
        f"True_NEI        {metrics['confusion_matrix']['matrix'][2][0]:<9}{metrics['confusion_matrix']['matrix'][2][1]:<9}{metrics['confusion_matrix']['matrix'][2][2]:<9}",
        "```"
    ])

    return "\n".join(lines)
