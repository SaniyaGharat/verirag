# Running Negation Repair on Google Colab (T4 GPU)

This guide provides step-by-step instructions for running the complete `negation_repair.py` scoring job across all 187,086 occurrences using a free or standard Google Colab T4 GPU instance.

---

### Step 1: Open Google Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook.
3. In the top menu, go to **Runtime** $\rightarrow$ **Change runtime type**.
4. Set **Hardware accelerator** to **T4 GPU** and click **Save**.

---

### Step 2: Install Minimal Dependencies
In the first Colab cell, install only the required packages:
```bash
!pip install -q transformers torch pandas numpy tqdm pyarrow
```

---

### Step 3: Upload Script and Extracted Occurrences
Upload the two required files from your local workspace:
- [`verirag/data/negation_repair.py`](file:///c:/Users/Saniya%20Gharat/Desktop/Projects/nlp/verirag/data/negation_repair.py) $\rightarrow$ `/content/negation_repair.py`
- [`verirag/data/processed/negation/number_occurrences.parquet`](file:///c:/Users/Saniya%20Gharat/Desktop/Projects/nlp/verirag/data/processed/negation/number_occurrences.parquet) $\rightarrow$ `/content/number_occurrences.parquet`

*Tip*: You can upload them using the Colab file explorer panel on the left or via Python:
```python
from google.colab import files
uploaded = files.upload()
```

---

### Step 4: Execute Full Scoring Command
Run the scoring script with window 32 and batch size 64:
```bash
!python negation_repair.py score \
    --input number_occurrences.parquet \
    --output number_scores.parquet \
    --window 32 \
    --batch-size 64
```

---

### Step 5: Resumability & Progress Monitoring
- The script automatically partitions all 187,086 rows into shards of **5,000 items** saved to `/content/shards/`.
- If the Colab session disconnects or preempts, simply rerun the same command: existing shards will be detected and skipped automatically.

---

### Step 6: Expected Runtime & Throughput
| Hardware Platform | Precision | Throughput | Expected Full Runtime (187k rows) |
| :--- | :--- | :--- | :--- |
| **Local 16-Core CPU** | FP32 | ~35 items/sec | ~1.5 hours |
| **Colab NVIDIA T4 GPU** | **FP16 (Autocast)** | **~350 - 450 items/sec** | **~7 - 9 minutes** |

*(Note: Frozen deterministic rules resolve 40,854 items (21.8%) instantly at zero GPU cost. The MLM only evaluates the remaining 146,232 items).*

---

### Step 7: Download Final Scores
Once execution finishes, download the resulting scored Parquet file back to your local repository:
- Destination: `/content/number_scores.parquet` $\rightarrow$ save locally to [`verirag/data/processed/negation/number_scores.parquet`](file:///c:/Users/Saniya%20Gharat/Desktop/Projects/nlp/verirag/data/processed/negation/number_scores.parquet)

```python
from google.colab import files
files.download("number_scores.parquet")
```
