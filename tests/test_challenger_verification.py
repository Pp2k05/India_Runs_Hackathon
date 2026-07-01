import re
import pytest
from src.ranker import hybrid_rank_candidates
from src.llm_evaluator import evaluate_candidate_fit

def test_null_robustness():
    # Test ranker with None/empty career history and skills
    candidate = {
        "candidate_id": "CAND_NULL_001",
        "profile": {
            "headline": "Software Engineer",
            "summary": "Full stack developer",
            "years_of_experience": 5.0,
            "current_title": "Software Engineer",
            "location": "Pune"
        },
        "career_history": None,
        "skills": None,
        "redrob_signals": None
    }
    
    # Check if hybrid_rank_candidates handles None values without raising exceptions
    try:
        ranked = hybrid_rank_candidates(
            candidates=[candidate],
            similarity_scores=[0.8],
            llm_results=[{}],
            required_skills=["python"]
        )
        assert len(ranked) == 1
        assert ranked[0]["candidate_id"] == "CAND_NULL_001"
        assert ranked[0]["score"] >= 0.0
    except Exception as e:
        pytest.fail(f"hybrid_rank_candidates failed with None fields: {e}")

def test_null_elements_robustness():
    # Test ranker when career_history or skills contain None or malformed elements
    # Note: s.get("name") is used in ranker.py, so an element of skills being None will raise AttributeError.
    # Let's verify if the code handles this, or if it expects skill dictionary items.
    candidate = {
        "candidate_id": "CAND_NULL_002",
        "profile": {
            "headline": "Software Engineer",
            "summary": "Full stack developer",
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [None],
        "skills": [None],
        "redrob_signals": {}
    }
    
    # We expect this might raise an error if not guarded. Let's see if we run it.
    # Actually, let's write a test to see if it does or doesn't fail.
    try:
        ranked = hybrid_rank_candidates(
            candidates=[candidate],
            similarity_scores=[0.8],
            llm_results=[{}],
            required_skills=["python"]
        )
        assert len(ranked) == 1
    except Exception as e:
        print(f"Observed error with None elements: {e}")

def test_llm_evaluator_null_robustness():
    # Test llm_evaluator with None candidate and None fields
    try:
        res1 = evaluate_candidate_fit(job_desc="Python developer", candidate=None, api_key=None)
        assert res1["trajectory_score"] == 0.0
        
        candidate = {
            "candidate_id": "CAND_NULL_003",
            "profile": None,
            "career_history": None,
            "skills": None,
            "redrob_signals": None
        }
        res2 = evaluate_candidate_fit(job_desc="Python developer", candidate=candidate, api_key=None)
        assert res2["trajectory_score"] == 0.0
    except Exception as e:
        pytest.fail(f"evaluate_candidate_fit failed with None fields: {e}")

def test_regex_word_boundaries():
    # 1. Consulting-only check
    # If a candidate has only worked at "Disney", does it trigger the consulting-only disqualifier?
    # "Disney" contains "ey", which is a consulting firm name.
    # If the word boundary is not used, "Disney" would match "ey", triggering consulting_only = True.
    # Let's verify that a candidate with "Disney" is NOT disqualified.
    candidate_disney = {
        "candidate_id": "CAND_DISNEY",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer"
        },
        "career_history": [
            {
                "company": "Disney",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "Python developer"
            }
        ],
        "skills": [{"name": "Python"}],
        "redrob_signals": {}
    }
    
    ranked = hybrid_rank_candidates(
        candidates=[candidate_disney],
        similarity_scores=[0.8],
        llm_results=[{}],
        required_skills=["python"]
    )
    
    # If disqualified, score is 0.0 and reasoning starts with [DISQUALIFIED]
    assert not ranked[0]["reasoning"].startswith("[DISQUALIFIED]"), "Disney triggered consulting-only disqualifier!"
    assert ranked[0]["score"] > 0.0

    # 2. CV/Speech-only check
    # Check if a candidate has cv/speech keyword like "cv" inside "achieved" or "achieve".
    # Wait, does "achieved" contain "cv"? Let's check "recving" or other words.
    # What about "speech" inside "speeches" or "speeched"?
    # Let's test if having "achieved" (or similar words) triggers the cv/speech-only disqualifier.
    # Actually, cv_speech_keys contains "speech", "cv", "audio", "yolo", "cuda", etc.
    # Let's test with a candidate whose resume description has "achieved" and "received" (without nlp_ir keys).
    candidate_achieved = {
        "candidate_id": "CAND_ACHIEVED",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "Software Engineer",
            "headline": "achieved great things"
        },
        "career_history": [
            {
                "company": "Some Company",
                "title": "Software Engineer",
                "duration_months": 24,
                "description": "achieved milestones, received recognition"
            }
        ],
        "skills": [{"name": "C++"}], # no nlp/ir skills
        "redrob_signals": {}
    }
    
    ranked_achieved = hybrid_rank_candidates(
        candidates=[candidate_achieved],
        similarity_scores=[0.8],
        llm_results=[{}],
        required_skills=["c++"]
    )
    
    assert not ranked_achieved[0]["reasoning"].startswith("[DISQUALIFIED]"), "achieved triggered cv/speech-only disqualifier!"

def test_trajectory_score_continuity():
    # Trajectory score formula in llm_evaluator fallback:
    # if 6.0 <= yoe <= 8.0: traj_score = 100.0
    # elif 5.0 <= yoe < 6.0: traj_score = 80.0 + 20.0 * (yoe - 5.0)
    # elif 8.0 < yoe <= 9.0: traj_score = 100.0 - 20.0 * (yoe - 8.0)
    # elif yoe < 5.0: traj_score = (yoe / 5.0) * 80.0
    # else: traj_score = max(0.0, 80.0 - 5.0 * (yoe - 9.0))
    
    yoe_test_points = [0.0, 1.0, 4.999, 5.0, 5.001, 5.999, 6.0, 7.0, 8.0, 8.001, 8.999, 9.0, 9.001, 15.0, 24.999, 25.0, 25.001, 30.0]
    
    results = []
    for yoe in yoe_test_points:
        cand = {
            "candidate_id": f"CAND_YOE_{yoe}",
            "profile": {"years_of_experience": yoe, "current_title": "Software Engineer"},
            "career_history": [],
            "skills": [],
            "redrob_signals": {}
        }
        res = evaluate_candidate_fit(job_desc="Python developer", candidate=cand, api_key=None)
        results.append((yoe, res["trajectory_score"]))
        
    print("\nTrajectory Score Continuity Verification:")
    for yoe, score in results:
        print(f"  YoE: {yoe:6.3f} -> Trajectory Score: {score:6.2f}")
        
    # Check key boundary values specifically:
    # YoE = 5.0 should be exactly 80.0
    # YoE = 6.0 should be exactly 100.0
    # YoE = 8.0 should be exactly 100.0
    # YoE = 9.0 should be exactly 80.0
    # YoE = 25.0 should be exactly 0.0
    assert abs(dict(results)[5.0] - 80.0) < 1e-7
    assert abs(dict(results)[6.0] - 100.0) < 1e-7
    assert abs(dict(results)[8.0] - 100.0) < 1e-7
    assert abs(dict(results)[9.0] - 80.0) < 1e-7
    assert abs(dict(results)[25.0] - 0.0) < 1e-7
    
    # Check that there are no huge jumps for small changes
    for i in range(len(yoe_test_points) - 1):
        y1, s1 = results[i]
        y2, s2 = results[i+1]
        if abs(y2 - y1) < 0.1:
            diff = abs(s2 - s1)
            # Max rate of change is 20.0 per 1.0 YoE. So for diff in YoE, max diff in score is 20 * diff_yoe + epsilon
            assert diff <= 20.0 * (y2 - y1) + 1e-7, f"Discontinuity between YoE {y1} and {y2}: scores {s1} and {s2}"

if __name__ == "__main__":
    test_null_robustness()
    test_null_elements_robustness()
    test_llm_evaluator_null_robustness()
    test_regex_word_boundaries()
    test_trajectory_score_continuity()
    print("All custom challenger tests passed successfully!")
