import pandas as pd
import os

input_path = "data/processed/ildc_cleaned_v4.parquet"
output_path = "data/processed/ildc_v4_dedup.parquet"

print(f"Loading {input_path}...")
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} rows.")

# Verify duplicate assertions
grouped = df.groupby("id")
all_text_match = True
all_label_match = True
mismatches = []

for doc_id, group in grouped:
    if len(group) > 1:
        first_text = group["text_cleaned"].iloc[0]
        first_label = group["label"].iloc[0]
        for idx, row in group.iloc[1:].iterrows():
            if row["text_cleaned"] != first_text:
                all_text_match = False
                mismatches.append((doc_id, "text"))
            if row["label"] != first_label:
                all_label_match = False
                mismatches.append((doc_id, "label"))

assert all_text_match, f"Text mismatch found in dropped rows: {mismatches}"
assert all_label_match, f"Label mismatch found in dropped rows: {mismatches}"
print("ASSERTION PASSED: Every dropped row's text is byte-identical to the kept row, and labels matched.")

dedup_df = df.drop_duplicates(subset=["id"], keep="first")
assert len(dedup_df) == 33822, f"Expected 33,822 rows, got {len(dedup_df)}"
print(f"Deduplication complete: {len(dedup_df)} rows (exactly matches expected 33,822).")

print("\n--- Row counts ---")
print(f"Before: {len(df)}")
print(f"After:  {len(dedup_df)}")
print(f"Dropped duplicate rows: {len(df) - len(dedup_df)}")

print("\n--- Label distribution BEFORE ---")
before_dist = df["label"].value_counts()
for label, count in before_dist.items():
    print(f"Label {label}: {count} ({count/len(df)*100:.2f}%)")

print("\n--- Label distribution AFTER ---")
after_dist = dedup_df["label"].value_counts()
for label, count in after_dist.items():
    print(f"Label {label}: {count} ({count/len(dedup_df)*100:.2f}%)")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
dedup_df.to_parquet(output_path, index=False)
print(f"\nSaved successfully to {output_path}")
