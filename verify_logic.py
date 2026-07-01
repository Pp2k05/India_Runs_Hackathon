import csv
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ranker import hybrid_rank_candidates

def test_custom_verification_scenarios():
    print("Running custom verification unit tests...")
    
    # 1. Test profile_completeness_score tie breaking
    # Two candidates with identical qualifications but different profile completeness.
    # We will construct them so they have identical technical, career, and behavioral scores.
    c1 = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "AILab",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "Python dev"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30
        }
    }
    
    c2 = {
        "candidate_id": "CAND_0000002",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "AILab",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "Python dev"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {
            "profile_completeness_score": 80.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30
        }
    }
    
    # Run ranker
    ranked = hybrid_rank_candidates(
        candidates=[c1, c2],
        similarity_scores=[0.8, 0.8],
        llm_results=[{}, {}],
        required_skills=["python"]
    )
    
    # c1 has completeness 90, c2 has 80. Since they are identical otherwise, c1 must have a higher score
    assert ranked[0]["candidate_id"] == "CAND_0000001", "Completeness score tie breaking failed!"
    assert ranked[0]["score"] > ranked[1]["score"], "Completeness score did not increase final score!"
    print("  [PASS] Profile completeness tie-breaking verified.")

    # 2. Test alphabetical candidate_id tie breaking on equal scores
    # Two candidates with identical qualifications and identical profile completeness.
    # The alphabetical order of candidate_id must resolve the tie (CAND_0000003 before CAND_0000004).
    c3 = {
        "candidate_id": "CAND_0000004",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "AILab",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "Python dev"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30
        }
    }
    
    c4 = {
        "candidate_id": "CAND_0000003",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "AILab",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "Python dev"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30
        }
    }
    
    ranked_ties = hybrid_rank_candidates(
        candidates=[c3, c4],
        similarity_scores=[0.8, 0.8],
        llm_results=[{}, {}],
        required_skills=["python"]
    )
    
    # CAND_0000003 must be first since scores are identical and it's alphabetically smaller
    assert ranked_ties[0]["candidate_id"] == "CAND_0000003", "Alphabetical candidate_id tie breaking failed!"
    assert ranked_ties[0]["score"] == ranked_ties[1]["score"], "Scores should be identical!"
    print("  [PASS] Alphabetical tie-breaking verified.")

    # 3. Test Honeypot detection
    # A candidate with years_of_experience = 15.0 but career history total duration_months = 12 (1 year)
    c_honeypot = {
        "candidate_id": "CAND_HONEY",
        "profile": {
            "years_of_experience": 15.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "AILab",
                "title": "Software Engineer",
                "duration_months": 12,
                "description": "Python dev"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.8,
            "notice_period_days": 30
        }
    }
    
    ranked_hp = hybrid_rank_candidates(
        candidates=[c_honeypot],
        similarity_scores=[0.8],
        llm_results=[{}],
        required_skills=["python"]
    )
    
    assert ranked_hp[0]["score"] == 0.0, f"Honeypot did not get 0.0 score! Got: {ranked_hp[0]['score']}"
    assert "[HONEYPOT WARNING]" in ranked_hp[0]["reasoning"], "Honeypot reasoning mismatch!"
    print("  [PASS] Honeypot detection verified.")


def verify_output_csv():
    csv_path = "ranked_candidates.csv"
    print(f"\nVerifying output CSV at {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist!")
        sys.exit(1)
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # 1. Check header
        expected_header = ["candidate_id", "rank", "score", "reasoning"]
        assert header == expected_header, f"Header mismatch! Expected {expected_header}, got {header}"
        
        rows = list(reader)
        
    # 2. Check exactly 100 rows
    num_rows = len(rows)
    assert num_rows == 100, f"Expected exactly 100 data rows, got {num_rows}"
    print(f"  [PASS] Contains exactly 100 data rows.")
    
    prev_score = float('inf')
    prev_cid = ""
    
    for idx, row in enumerate(rows):
        cid, rank_s, score_s, reasoning = row
        rank = int(rank_s)
        score = float(score_s)
        
        # 3. Check contiguous ranks
        assert rank == idx + 1, f"Rank mismatch at row {idx+2}: expected {idx+1}, got {rank}"
        
        # 4. Check score non-increasing
        assert score <= prev_score, f"Score increased at rank {rank}: {prev_score} -> {score}"
        
        # 5. Check tie-breaker alphabetical
        # Note: if scores are equal (after rounding to 6 decimal places as written to CSV)
        if abs(score - prev_score) < 1e-9:
            assert cid > prev_cid, f"Alphabetical tie-break violation at rank {rank}: '{prev_cid}' and '{cid}' with score {score}"
            
        # 6. Check that no honeypots or disqualified candidates are in top 100
        assert not reasoning.startswith("[DISQUALIFIED]"), f"Disqualified candidate {cid} in top 100!"
        assert not reasoning.startswith("[HONEYPOT WARNING]"), f"Honeypot candidate {cid} in top 100!"
        
        prev_score = score
        prev_cid = cid
        
    print("  [PASS] Contiguous ranks verified.")
    print("  [PASS] Non-increasing scores verified.")
    print("  [PASS] Alphabetical tie-breaking verified on CSV content.")
    print("  [PASS] No disqualified or honeypot candidates found in top 100.")
    print("All CSV verification checks passed successfully!")


if __name__ == "__main__":
    test_custom_verification_scenarios()
    verify_output_csv()
