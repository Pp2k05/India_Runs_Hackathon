"""
diagnose.py — Investigate the 8 review issues before fixing
"""
import json, os, re, csv
from datetime import datetime, date

CANDIDATES_FILE = r"C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
CSV_FILE = r"C:\Users\parth\OneDrive\Documents\India-runs\ai_candidate_ranker\ranked_candidates.csv"

# ── Read current CSV ──
rows = {}
with open(CSV_FILE, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows[row["candidate_id"]] = row

# ── Load the problematic candidates ──
targets = {"CAND_0079387", "CAND_0018499", "CAND_0026942", "CAND_0043860",
           "CAND_0054100", "CAND_0002706", "CAND_0083852", "CAND_0089381", "CAND_0043957"}
found = {}

with open(CANDIDATES_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line.strip())
        cid = c.get("candidate_id")
        if cid in targets:
            found[cid] = c
        if len(found) == len(targets):
            break

print("="*70)
print("INVESTIGATION REPORT")
print("="*70)

def summarize(cid):
    c = found.get(cid, {})
    if not c:
        print(f"\n{cid}: NOT FOUND in loaded batch")
        return
    p = c.get("profile") or {}
    career = c.get("career_history") or []
    skills = c.get("skills") or []
    sigs = c.get("redrob_signals") or {}
    
    skill_names = [s.get("name","") for s in skills if s and s.get("name")]
    all_text = " ".join(skill_names + [p.get("headline",""), p.get("summary","")[:200]] +
                       [(j.get("title","") + " " + j.get("description","")[:100]) for j in career[:3]]).lower()
    
    # Core requirement checks
    PROD_EMB = {"faiss","pinecone","weaviate","qdrant","milvus","elasticsearch","opensearch",
                "vector search","dense retrieval","semantic search","embedding system","hybrid search"}
    EVAL_FW  = {"ndcg","mrr","map","a/b test","evaluation framework","learning to rank","offline eval"}
    VECT_DB  = {"faiss","pinecone","weaviate","qdrant","milvus","pgvector","chroma","vespa"}
    
    has_prod = any(k in all_text for k in PROD_EMB)
    has_eval = any(k in all_text for k in EVAL_FW)
    has_vdb  = any(k in all_text for k in VECT_DB)
    
    companies = [j.get("company","") for j in career[:5] if j.get("company")]
    job_titles = [j.get("title","") for j in career[:5] if j.get("title")]
    
    csv_row = rows.get(cid, {})
    rank = csv_row.get("rank","?")
    score = csv_row.get("score","?")
    reasoning = csv_row.get("reasoning","?")
    
    last_active = sigs.get("last_active_date","?")
    resp_rate = sigs.get("recruiter_response_rate","?")
    yoe = p.get("years_of_experience","?")
    notice = sigs.get("notice_period_days","?")
    
    print(f"\n{'─'*60}")
    print(f"  {cid} | Rank #{rank} | Score {score}")
    print(f"  Title: {p.get('current_title')} | YoE: {yoe} | Location: {p.get('location')}")
    print(f"  Companies: {', '.join(companies[:3])}")
    print(f"  Prev titles: {', '.join(job_titles[:3])}")
    print(f"  Skills: {', '.join(skill_names[:8])}")
    print(f"  Core checks: prod_embedding={has_prod} | eval_framework={has_eval} | vector_db={has_vdb}")
    print(f"  Signals: last_active={last_active} | resp_rate={resp_rate} | notice={notice}d")
    print(f"  REASONING: {reasoning[:200]}")

print("\n--- RANK 1 vs RANK 2 INVESTIGATION ---")
summarize("CAND_0079387")
summarize("CAND_0018499")

print("\n--- RANK 5 (should-be-disqualified) ---")
summarize("CAND_0026942")

print("\n--- OTHER FLAGGED CANDIDATES ---")
for cid in ["CAND_0043860", "CAND_0054100", "CAND_0002706", "CAND_0083852"]:
    summarize(cid)

print("\n--- TITLE MISMATCH CANDIDATES ---")
summarize("CAND_0089381")  # CV Engineer at rank 72
summarize("CAND_0043957")  # Data Analyst at rank 70

print("\n\n--- PHASE 1 DISQUALIFICATION SAMPLES ---")
print("Finding 5 disqualified candidates for evidence...")
CONSULTING = {"wipro","tata consultancy","tcs","infosys","accenture","cognizant",
              "capgemini","hcl","tech mahindra","mindtree","deloitte","pwc"}
NON_TECH = {"marketing manager","hr manager","content writer","graphic designer",
            "data analyst","business analyst","sales manager"}

disq_examples = []
with open(CANDIDATES_FILE, encoding="utf-8") as f:
    for line in f:
        if len(disq_examples) >= 5:
            break
        try:
            c = json.loads(line.strip())
        except:
            continue
        p = c.get("profile") or {}
        career = c.get("career_history") or []
        title = (p.get("current_title") or "").lower()
        
        # Check consulting-only
        if career:
            all_consulting = all(
                any(firm in (j.get("company","")).lower() for firm in CONSULTING)
                for j in career if j.get("company")
            )
            if all_consulting and career:
                disq_examples.append((c, "CONSULTING_ONLY", career))
                continue
        
        # Check non-tech title
        if any(nt in title for nt in NON_TECH):
            disq_examples.append((c, f"NON_TECH_TITLE: {p.get('current_title')}", []))
            continue

for c, reason, career in disq_examples:
    p = c.get("profile") or {}
    cos = [j.get("company","") for j in (c.get("career_history") or [])[:3]]
    print(f"\n  {c.get('candidate_id')} | {p.get('current_title')} | {p.get('years_of_experience')}yrs")
    print(f"  Disqualified: {reason}")
    if cos:
        print(f"  Companies: {', '.join(cos)}")

print("\n--- HYBRID CASE TEST (TCS early, then product startup) ---")
hybrid_found = []
with open(CANDIDATES_FILE, encoding="utf-8") as f:
    for line in f:
        if len(hybrid_found) >= 3:
            break
        try:
            c = json.loads(line.strip())
        except:
            continue
        career = c.get("career_history") or []
        if len(career) < 2:
            continue
        
        has_consulting_early = any(
            any(firm in (j.get("company","")).lower() for firm in CONSULTING)
            for j in career[-2:]  # last items = oldest jobs
        )
        has_product_recent = not any(
            any(firm in (j.get("company","")).lower() for firm in CONSULTING)
            for j in career[:2]  # first items = recent jobs
        )
        
        if has_consulting_early and has_product_recent:
            p = c.get("profile") or {}
            cos = [(j.get("company",""), j.get("title","")) for j in career[:4]]
            print(f"\n  {c.get('candidate_id')} | {p.get('current_title')} | {p.get('years_of_experience')}yrs")
            print(f"  Career (recent first): {cos}")
            print(f"  is_consulting_only() would return: False (has product company)")
            print(f"  → PASSES THROUGH correctly ✓")
            hybrid_found.append(c)

print("\n\nDone.")
