# Gold Labeling Guide for Standalone `[number]` Occurrences

This guide defines the annotation protocol and valid categories for manually reviewing standalone `[number]` tokens in `gold_number_200_blind.csv`.

---

## Allowed Labels

| Label | Meaning | Description & Syntactic Criteria | Example |
| :--- | :--- | :--- | :--- |
| **`not`** | Adverb / Negation | Standard clausal negation modifying a verb, predicate, participle, or preposition. | *"The High Court was [number] right in dismissing the writ petition."* $\rightarrow$ **`not`** |
| **`nor`** | Coordinating Conjunction | Negative coordination, typically paired with *neither*, or continuing a negative clause. | *"He had neither food [number] water during the custody period."* $\rightarrow$ **`nor`** |
| **`now`** | Temporal Adverb | Discourse marker or adverb of time (*at this moment, presently*), often corrupted from OCR artifact. | *"Let us [number] consider the second submission raised by counsel."* $\rightarrow$ **`now`** |
| **`non`** | Negative Prefix | Negative prefix attached via hyphen or space to nouns, adjectives, or participles (*non-compliance, non-bailable*). | *"Suffers from the vice of [number] -application of mind."* $\rightarrow$ **`non`** |
| **`no`** | Determiner / Negative Quantifier | Quantifier modifying a noun phrase (*no reason, no evidence, there is no doubt*). | *"There was [number] doubt as to the jurisdiction of the Magistrate."* $\rightarrow$ **`no`** |
| **`number`** | Genuine Noun / Identifier | Legitimate cardinality, quantity, identifier, or index (*respondent number 4, serial number, number was small*). | *"Arguments were concluded on behalf of respondent [number] 4."* $\rightarrow$ **`number`** |
| **`?`** | Ambiguous / Indeterminate | Severe textual corruption where human annotators cannot reliably deduce the intended word. | *"and x71 q [number] z9 with respect to"* $\rightarrow$ **`?`** |

---

## Instructions for Human Reviewers

1. Open [`verirag/data/reports/gold_number_200_blind.csv`](file:///c:/Users/Saniya%20Gharat/Desktop/Projects/nlp/verirag/data/reports/gold_number_200_blind.csv).
2. Examine `left_context` (15 words), the target token position `[number]`, and `right_context` (15 words).
3. Fill the `human_label` column with one of the exact labels above:
   `not`, `nor`, `now`, `non`, `no`, `number`, or `?`.
4. Leave no predictions or guesses from automated scripts in this gold file.
