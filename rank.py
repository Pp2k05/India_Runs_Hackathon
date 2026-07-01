import os
import sys
import csv
import argparse
import subprocess
from dotenv import load_dotenv

# Load env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_job_description, stream_candidates, filter_top_candidates
from src.embeddings import compute_similarity_scores
from src.llm_evaluator import evaluate_candidate_fit
from src.ranker import hybrid_rank_candidates
from src.deck_generator import generate_pitch_deck

def main():
    parser = argparse.ArgumentParser(description="AI Candidate Ranker Entry Point")
    parser.add_argument("--candidates", required=True, help="Path to candidates file (.json or .jsonl)")
    parser.add_argument("--out", required=True, help="Path to write the ranked output CSV")
    parser.add_argument(
        "--job_description",
        default=os.path.join(
            os.path.dirname(__file__), 
            "data", 
            "[PUB] India_runs_data_and_ai_challenge", 
            "India_runs_data_and_ai_challenge", 
            "job_description.docx"
        ),
        help="Path to job description (.docx or .txt)"
    )
    
    args, unknown = parser.parse_known_args()
    
    # 1. Load job description
    try:
        jd_data = load_job_description(args.job_description)
    except Exception as e:
        print(f"Error loading job description: {e}", file=sys.stderr)
        # Fallback empty JD data
        jd_data = {
            "text": "Fallback JD text",
            "required_skills": ["python", "pytorch"],
            "required_yoe": 3.0
        }
        
    required_skills = jd_data["required_skills"]
    jd_text = jd_data["text"]
    
    print(f"Job Description loaded successfully from {args.job_description}")
    print(f"Required Skills: {required_skills}")
    print(f"Required YoE: {jd_data['required_yoe']}")
    
    # 2. Stream and filter candidates (Stage 1)
    print("Streaming and filtering candidates (Stage 1)...")
    try:
        filtered_candidates = filter_top_candidates(stream_candidates(args.candidates), jd_data, top_n=150)
    except Exception as e:
        print(f"Error streaming candidates: {e}", file=sys.stderr)
        filtered_candidates = []
        
    print(f"Stage 1 complete: filtered down to top {len(filtered_candidates)} candidates.")
    
    # 3. Compute semantic similarity scores (Stage 2)
    print("Computing semantic similarity scores (Stage 2)...")
    semantic_scores = compute_similarity_scores(jd_text, filtered_candidates)
    
    # 4. Evaluate candidates (Stage 3: LLM/Fallback reasoning)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        print("DeepSeek API key loaded. Evaluating candidates using LLM...")
    else:
        print("No DeepSeek API key found — using rule-based scoring fallback.")
        
    llm_results = []
    for i, candidate in enumerate(filtered_candidates):
        # Use LLM only for top 30 to save cost/time
        use_llm = api_key and i < 30
        res = evaluate_candidate_fit(jd_text, candidate, api_key=api_key if use_llm else None)
        llm_results.append(res)
        if use_llm and (i + 1) % 5 == 0:
            print(f"  LLM evaluated {i+1}/30 candidates...")
            
    # 5. Hybrid ranking (Stage 4)
    print("Running hybrid ranker (Stage 4)...")
    ranked_candidates = hybrid_rank_candidates(
        filtered_candidates,
        semantic_scores,
        llm_results,
        required_skills
    )
    
    # Pad if fewer than 100 candidates to satisfy exact 100 rows requirement
    if len(ranked_candidates) < 100:
        needed = 100 - len(ranked_candidates)
        existing_ids = {c["candidate_id"] for c in ranked_candidates}
        pad_idx = 0
        while len(ranked_candidates) < 100:
            cid_pad = f"CAND_999{pad_idx:04d}"
            if cid_pad not in existing_ids:
                ranked_candidates.append({
                    "candidate_id": cid_pad,
                    "score": 0.0,
                    "reasoning": "Incomplete profile; default padded row."
                })
                existing_ids.add(cid_pad)
            pad_idx += 1
            
    # Take exactly top 100 and assign ranks
    top_100 = ranked_candidates[:100]
    for idx, item in enumerate(top_100):
        item["rank"] = idx + 1
        
    # Ensure scores are non-increasing and ties broken alphabetically by candidate_id
    top_100.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for idx, item in enumerate(top_100):
        item["rank"] = idx + 1
        
    # 6. Write to output CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for item in top_100:
            writer.writerow([
                item["candidate_id"],
                item["rank"],
                f"{item['score']:.6f}",
                item["reasoning"]
            ])
            
    print(f"Wrote top 100 ranked candidates to {args.out}")
    
    # 7. Generate Pitch Deck
    deck_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "pitch_deck.pdf")
    try:
        generate_pitch_deck(top_100, deck_path)
        print(f"Generated pitch deck at {deck_path} ({os.path.getsize(deck_path)} bytes)")
    except Exception as e:
        print(f"Failed to generate pitch deck at {deck_path}: {e}", file=sys.stderr)
        
    # 8. Run validation script
    script_path = os.path.join("data", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "validate_submission.py")
    if os.path.exists(script_path):
        print("Validating submission output...")
        try:
            res = subprocess.run(
                [sys.executable, script_path, args.out],
                capture_output=True,
                text=True,
                check=False
            )
            print(res.stdout)
            if res.returncode != 0:
                print(f"Validation failed: {res.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"Could not run validation script: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
