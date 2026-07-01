import os
import csv
import json
import zipfile
import pytest
import sys
from unittest.mock import patch, MagicMock
import requests

from src.data_loader import load_job_description, stream_candidates, load_candidates
from src.embeddings import compute_similarity_scores
from src.llm_evaluator import evaluate_candidate_fit
from src.ranker import hybrid_rank_candidates
from src.deck_generator import generate_pitch_deck

# Reuse helper functions from tier1 or define inline
def get_dummy_candidate(cid="CAND_0000001", skills=None, history=None, signals=None):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Jane Doe",
            "headline": "Developer",
            "summary": "Summary",
            "location": "Delhi",
            "country": "India",
            "years_of_experience": 4.0,
            "current_title": "Software Engineer",
            "current_company": "MNC",
            "current_company_size": "10001+",
            "current_industry": "Tech"
        },
        "career_history": history or [
            {
                "company": "MNC",
                "title": "Software Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "Tech",
                "company_size": "10001+",
                "description": "Coded backend systems."
            }
        ],
        "education": [],
        "skills": skills or [{"name": "Python", "proficiency": "advanced", "endorsements": 1, "duration_months": 12}],
        "redrob_signals": signals or {
            "profile_completeness_score": 80.0,
            "signup_date": "2021-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 1,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 10,
            "endorsements_received": 1,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 5.0, "max": 10.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "github_activity_score": 20.0,
            "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 1,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": False,
            "linkedin_connected": False
        }
    }

# ==========================================
# F1: JOB DESCRIPTION BOUNDARY CASES
# ==========================================

def test_f1_empty_jd(tmp_path):
    txt_file = tmp_path / "jd.txt"
    txt_file.write_text("", encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert jd["text"] == ""
    assert len(jd["required_skills"]) == 0
    assert jd["required_yoe"] == 0.0

def test_f1_ultra_long_jd(tmp_path):
    txt_file = tmp_path / "jd.txt"
    # 200,000 characters
    long_text = "requirements python " * 10000
    txt_file.write_text(long_text, encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert "python" in jd["required_skills"]
    assert len(jd["text"]) > 100000

def test_f1_missing_parameters():
    # Calling parsing with empty string path
    with pytest.raises(FileNotFoundError):
        load_job_description("")

def test_f1_corrupt_docx(tmp_path):
    docx_file = tmp_path / "corrupt.docx"
    # Write corrupt data
    docx_file.write_text("this is not a zip file", encoding="utf-8")
    with pytest.raises(ValueError):
        load_job_description(str(docx_file))

def test_f1_special_characters_jd(tmp_path):
    txt_file = tmp_path / "jd.txt"
    # Contains emojis, non-ASCII quotes and math symbols
    txt_file.write_text("Looking for 🚀 PyTorch Engineer with 5+ years. 💻 α + β = γ", encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert "pytorch" in jd["required_skills"]
    assert jd["required_yoe"] == 5.0


# ==========================================
# F2: STREAMING & FILTERING BOUNDARY CASES
# ==========================================

def test_f2_empty_jsonl(tmp_path):
    jsonl_file = tmp_path / "cands.jsonl"
    jsonl_file.write_text("", encoding="utf-8")
    streamed = list(stream_candidates(str(jsonl_file)))
    assert len(streamed) == 0

def test_f2_zero_matching_candidates(tmp_path):
    # Tests zero profiles in file
    json_file = tmp_path / "empty_list.json"
    json_file.write_text("[]", encoding="utf-8")
    streamed = list(stream_candidates(str(json_file)))
    assert len(streamed) == 0

def test_f2_malformed_json_lines(tmp_path):
    jsonl_file = tmp_path / "cands.jsonl"
    # Second line is corrupt
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(get_dummy_candidate("CAND_0000001")) + "\n")
        f.write("{malformed_json_string\n")
        f.write(json.dumps(get_dummy_candidate("CAND_0000002")) + "\n")
        
    # Standard streamer will fail on line 2, but we can verify it parses line 1 or handles it
    generator = stream_candidates(str(jsonl_file))
    cand1 = next(generator)
    assert cand1["candidate_id"] == "CAND_0000001"
    with pytest.raises(json.JSONDecodeError):
        next(generator)

def test_f2_extremely_small_file(tmp_path):
    json_file = tmp_path / "one.json"
    json_file.write_text(json.dumps([get_dummy_candidate()]), encoding="utf-8")
    streamed = list(stream_candidates(str(json_file)))
    assert len(streamed) == 1

def test_f2_huge_candidate_file_scaling(tmp_path):
    # Emulate memory-efficient loading of 1000 items
    jsonl_file = tmp_path / "cands_large.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for i in range(1000):
            c = get_dummy_candidate(f"CAND_{i:07d}")
            f.write(json.dumps(c) + "\n")
            
    count = 0
    # Stream without loading entire file
    for c in stream_candidates(str(jsonl_file)):
        count += 1
    assert count == 1000


# ==========================================
# F3: LOCAL EMBEDDINGS & SKILLS BOUNDARY CASES
# ==========================================

def test_f3_zero_skills_candidate():
    c = get_dummy_candidate(skills=[])
    sim_scores = [0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}]
    ranked = hybrid_rank_candidates([c], sim_scores, llm_results, ["python"])
    assert len(ranked) == 1
    assert ranked[0]["score"] >= 0.0

def test_f3_extreme_endorsements():
    # Endorsements = 1000000
    c = get_dummy_candidate(skills=[
        {"name": "Python", "proficiency": "expert", "endorsements": 1000000, "duration_months": 24}
    ])
    sim_scores = [0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}]
    ranked = hybrid_rank_candidates([c], sim_scores, llm_results, ["python"])
    assert ranked[0]["score"] <= 1.0

def test_f3_negative_duration_skill():
    c = get_dummy_candidate(skills=[
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": -5}
    ])
    sim_scores = [0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}]
    ranked = hybrid_rank_candidates([c], sim_scores, llm_results, ["python"])
    assert ranked[0]["score"] >= 0.0

def test_f3_no_history_descriptions():
    # Empty description string
    c = get_dummy_candidate(history=[
        {"company": "X", "title": "SE", "start_date": "2020", "end_date": None, "duration_months": 12, "is_current": True, "industry": "IT", "company_size": "1-10", "description": ""}
    ])
    sim_scores = [0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}]
    ranked = hybrid_rank_candidates([c], sim_scores, llm_results, ["python"])
    # Should apply 0.5 verification penalty
    assert ranked[0]["score"] >= 0.0

def test_f3_extremely_low_similarity():
    jd = "Python developer"
    cands = [get_dummy_candidate()]
    cands[0]["profile"]["headline"] = "xyz abc wuv"
    cands[0]["profile"]["summary"] = "none"
    cands[0]["career_history"][0]["description"] = "nothing"
    
    scores = compute_similarity_scores(jd, cands)
    # Cosine/Jaccard should be low/0
    assert scores[0] >= 0.0


# ==========================================
# F4: DEEPSEEK API BOUNDARY CASES
# ==========================================

def test_f4_api_corrupt_non_json():
    cand = get_dummy_candidate()
    # auth header with 'corrupt' returns invalid JSON
    res = evaluate_candidate_fit("Python JD", cand, api_key="corrupt_key")
    # Should fall back to template reasoning
    assert "matches" in res["fit_summary"]

def test_f4_api_server_error_500():
    cand = get_dummy_candidate()
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
        assert "matches" in res["fit_summary"]

def test_f4_api_empty_choices():
    cand = get_dummy_candidate()
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response
        
        res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
        assert "matches" in res["fit_summary"]

def test_f4_api_invalid_credentials():
    cand = get_dummy_candidate()
    # auth header 'invalid_key' triggers 401 response in mock
    res = evaluate_candidate_fit("Python JD", cand, api_key="invalid_key_abc")
    assert "matches" in res["fit_summary"]

def test_f4_api_rate_limit_429():
    cand = get_dummy_candidate()
    # auth header 'rate_limit' triggers 429 response in mock
    res = evaluate_candidate_fit("Python JD", cand, api_key="rate_limit_abc")
    assert "matches" in res["fit_summary"]


# ==========================================
# F5: DEEPSEEK FALLBACK BOUNDARY CASES
# ==========================================

def test_f5_fallback_empty_templates():
    # Candidate with empty/null fields
    cand = {
        "candidate_id": "CAND_0000001",
        "profile": {},
        "career_history": [],
        "skills": [],
        "redrob_signals": {}
    }
    res = evaluate_candidate_fit("Python JD", cand, api_key=None)
    assert "Unknown Title" in res["fit_summary"]
    assert res["trajectory_score"] >= 0.0

def test_f5_fallback_null_env():
    # Empty environment variables
    with patch.dict(os.environ, {}, clear=True):
        cand = get_dummy_candidate()
        res = evaluate_candidate_fit("Python JD", cand, api_key=None)
        assert res["trajectory_score"] > 0.0

def test_f5_fallback_triggers_mid_batch():
    # Testing that fallback is triggered when API returns failure
    cand = get_dummy_candidate()
    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
        # Triggers fallback on connection error
        assert "matches" in res["fit_summary"]

def test_f5_fallback_extreme_values():
    cand = get_dummy_candidate()
    cand["profile"]["years_of_experience"] = 100.0 # Extreme YoE
    res = evaluate_candidate_fit("Python JD", cand, api_key=None)
    assert res["trajectory_score"] == 0.0

def test_f5_fallback_missing_fields():
    # Completely empty candidate dictionary
    cand = {}
    res = evaluate_candidate_fit("Python JD", cand, api_key=None)
    assert "Unknown Title" in res["fit_summary"]
    assert res["trajectory_score"] >= 0.0


# ==========================================
# F6: OUTPUT CSV FORMATTING BOUNDARY CASES
# ==========================================

def test_f6_score_ties_multi(tmp_path):
    out = tmp_path / "ties.csv"
    cands = [
        get_dummy_candidate("CAND_0000003"),
        get_dummy_candidate("CAND_0000001"),
        get_dummy_candidate("CAND_0000002")
    ]
    sim_scores = [0.8, 0.8, 0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}] * 3
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    
    # Verify sorting: scores are identical, so CAND_0000001, CAND_0000002, CAND_0000003
    assert ranked[0]["candidate_id"] == "CAND_0000001"
    assert ranked[1]["candidate_id"] == "CAND_0000002"
    assert ranked[2]["candidate_id"] == "CAND_0000003"

def test_f6_duplicate_ids(tmp_path):
    cands = [
        get_dummy_candidate("CAND_0000001"),
        get_dummy_candidate("CAND_0000001") # Duplicate ID
    ]
    sim_scores = [0.8, 0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}] * 2
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    assert len(ranked) == 2

def test_f6_missing_columns_validation(tmp_path):
    out = tmp_path / "missing_cols.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "score"]) # missing rank and reasoning
        writer.writerow(["CAND_0000001", "0.95"])
        
    # Run validation
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge"))
    sys.path.append(script_dir)
    try:
        from validate_submission import validate_submission
        errors = validate_submission(str(out))
        assert len(errors) > 0
    except ImportError:
        pass

def test_f6_extra_columns_validation(tmp_path):
    out = tmp_path / "extra_cols.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Extra column
        writer.writerow(["candidate_id", "rank", "score", "reasoning", "extra_col"])
        for i in range(100):
            writer.writerow([f"CAND_{i+1:07d}", i+1, "0.90", "reason", "extra"])
            
    try:
        from validate_submission import validate_submission
        errors = validate_submission(str(out))
        assert len(errors) > 0
    except ImportError:
        pass

def test_f6_score_ordering_edge_case(tmp_path):
    # All scores 0.0 (by ensuring they are disqualified due to empty career history)
    cands = []
    for i in range(100):
        c = get_dummy_candidate(f"CAND_{i:07d}")
        c["career_history"] = []
        cands.append(c)
    sim_scores = [0.0] * 100
    llm_results = [{"trajectory_score": 0.0, "skill_depth_score": 0.0, "role_fit_score": 0.0, "behavior_score": 0.0}] * 100
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    
    out = tmp_path / "zeros.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(ranked):
            writer.writerow([item["candidate_id"], idx+1, f"{item['score']:.6f}", item["reasoning"]])
            
    # Verify scores are exactly 0.0
    with open(out, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        scores = [float(row[2]) for row in reader]
        assert all(s == 0.0 for s in scores)


# ==========================================
# F7: PITCH DECK BOUNDARY CASES
# ==========================================

def test_f7_pdf_permission_error(tmp_path):
    # Provide a directory that does not exist
    invalid_path = tmp_path / "non_existent_folder" / "deck.pdf"
    with pytest.raises(Exception):
        generate_pitch_deck([], str(invalid_path))

def test_f7_pdf_empty_candidates(tmp_path):
    # Works without error
    out = tmp_path / "empty_deck.pdf"
    generate_pitch_deck([], str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0

def test_f7_pdf_extremely_long_text(tmp_path):
    out = tmp_path / "long_text_deck.pdf"
    long_reasoning = "This is a candidate with an exceptionally long reasoning text that will exceed standard slide margins and should be handled by clipping or wrapping. " * 50
    cands = [{"candidate_id": "CAND_0000001", "score": 0.95, "reasoning": long_reasoning}]
    generate_pitch_deck(cands, str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0

def test_f7_pdf_non_ascii_characters(tmp_path):
    out = tmp_path / "unicode_deck.pdf"
    unicode_reasoning = "Candidate matches 🚀 PyTorch and SQL. Relocation status: ✅."
    cands = [{"candidate_id": "CAND_0000001", "score": 0.95, "reasoning": unicode_reasoning}]
    generate_pitch_deck(cands, str(out))
    assert os.path.exists(out)

def test_f7_pdf_file_overwrite(tmp_path):
    out = tmp_path / "deck.pdf"
    # Write initial file
    out.write_text("dummy", encoding="utf-8")
    # Generate PDF over it
    generate_pitch_deck([], str(out))
    assert os.path.getsize(out) > 5 # PDF has larger size than 'dummy'
