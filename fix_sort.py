import csv

IN  = r'C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\ranked_candidates.csv'

rows = []
with open(IN, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Sort by score desc at 4dp precision, then candidate_id ascending for ties
rows.sort(key=lambda r: (-round(float(r['score']), 4), r['candidate_id']))

with open(IN, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for i, row in enumerate(rows, 1):
        writer.writerow([row['candidate_id'], i, f"{float(row['score']):.4f}", row['reasoning']])

print(f'Re-sorted {len(rows)} rows and written.')
print('Top 5:')
for r in rows[:5]:
    print(f"  #{r.get('rank','?')}  {r['candidate_id']}  {r['score']}")
