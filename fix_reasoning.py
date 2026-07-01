"""
fix_reasoning.py — Post-processes ranked_candidates.csv to:
1. Strip stale rank numbers from LLM reasoning (e.g. "Candidate ranks #9 due to")
2. Fix factual errors (e.g. "6.6 years fall short of 5-9 year requirement")
3. Re-validate submission
"""
import csv, re, os, sys, subprocess

CSV = r"C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\ranked_candidates.csv"
DATA_DIR = r"C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
VAL = os.path.join(DATA_DIR, "validate_submission.py")

def clean_reasoning(text, actual_rank):
    """Remove stale rank references and fix factual errors."""
    # 1. Strip "Candidate ranks #N due to/because" → just keep what follows
    text = re.sub(
        r"Candidate ranks #\d+\s+(?:due to|because|as|since)\s+",
        "", text, flags=re.IGNORECASE
    )
    # 2. Strip "Rank #N:" or "Rank #N because" at start of sentence
    text = re.sub(
        r"Rank #\d+[:\s]+(?:because\s+|due to\s+)?",
        "", text, flags=re.IGNORECASE
    )
    # 3. Fix YoE factual errors: "X years fall short of the 5-9 year"
    # where X is actually within 5-9
    def fix_yoe(m):
        yoe_str = m.group(1)
        try:
            yoe = float(yoe_str)
            if 5.0 <= yoe <= 9.0:
                return f"{yoe_str} years of experience"
            return m.group(0)
        except:
            return m.group(0)
    text = re.sub(
        r"(\d+\.?\d*)\s+years?\s+fall\s+short\s+of\s+the\s+5-9\s+year\s+requirement",
        fix_yoe, text, flags=re.IGNORECASE
    )
    # 4. Fix same pattern with "below the 5-9 year requirement" when YoE is ≥5
    def fix_below(m):
        yoe_str = m.group(1)
        try:
            yoe = float(yoe_str)
            if 5.0 <= yoe <= 9.0:
                return f"{yoe_str} years of experience at the lower end of the target range"
            return m.group(0)
        except:
            return m.group(0)
    text = re.sub(
        r"(\d+\.?\d*)\s+years?\s+(?:is\s+)?below\s+the\s+5-9\s+year\s+requirement",
        fix_below, text, flags=re.IGNORECASE
    )
    # 5. Clean up double spaces or leading commas from stripping
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[,;:\s]+", "", text)
    text = text.strip()
    # 6. Ensure sentence ends with period
    if text and not text.endswith((".","!","?")):
        text += "."
    return text

rows = []
with open(CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"Processing {len(rows)} rows...")
changes = 0
for row in rows:
    original = row["reasoning"]
    cleaned  = clean_reasoning(original, int(row["rank"]))
    if cleaned != original:
        print(f"\n  Rank #{row['rank']} ({row['candidate_id']}):")
        print(f"    BEFORE: {original[:120]}")
        print(f"    AFTER:  {cleaned[:120]}")
        row["reasoning"] = cleaned
        changes += 1

print(f"\n{changes} rows updated.")

with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate_id","rank","score","reasoning"])
    for row in rows:
        w.writerow([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])

print("\nValidating...")
res = subprocess.run([sys.executable, VAL, CSV], capture_output=True, text=True)
print((res.stdout + res.stderr).strip())
