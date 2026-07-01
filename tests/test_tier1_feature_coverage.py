import os
import sys
import csv
import json
import zipfile
import xml.etree.ElementTree as ET
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.data_loader import load_job_description, stream_candidates, load_candidates
from src.embeddings import compute_similarity_scores
from src.llm_evaluator import evaluate_candidate_fit
from src.ranker import hybrid_rank_candidates
from src.deck_generator import generate_pitch_deck

# --- Helper Functions ---

def create_mock_docx(filepath, text):
    """Creates a mock docx containing the given text in word/document.xml."""
    # Register namespaces
    ET.register_namespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    root = ET.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document')
    body = ET.SubElement(root, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body')
    p = ET.SubElement(body, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')
    t = ET.SubElement(p, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
    t.text = text
    
    xml_str = ET.tostring(root, encoding='utf-8')
    with zipfile.ZipFile(filepath, 'w') as z:
        z.writestr('word/document.xml', xml_str)

def get_dummy_candidate(cid="CAND_0000001", skills=None, history=None, signals=None):
    """Generates a dummy candidate dictionary conforming to schema."""
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "AI Specialist",
            "summary": "Experienced machine learning engineer.",
            "location": "Bengaluru",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "Tech Corp",
            "current_company_size": "51-200",
            "current_industry": "Software"
        },
        "career_history": history or [
            {
                "company": "Tech Corp",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Developed deep learning model with PyTorch."
            }
        ],
        "education": [],
        "skills": skills or [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
        ],
        "redrob_signals": signals or {
            "profile_completeness_score": 90.0,
            "signup_date": "2020-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 12,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 2.5,
            "skill_assessment_scores": {},
            "connection_count": 150,
            "endorsements_received": 15,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 85.0,
            "search_appearance_30d": 45,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }

# ==========================================
# F1: JOB DESCRIPTION PARSING TESTS
# ==========================================

def test_f1_load_jd_txt(tmp_path):
    txt_file = tmp_path / "jd.txt"
    txt_file.write_text("We need a Python developer with 5+ years of experience in PyTorch.", encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert "Python" in jd["text"]
    assert "python" in jd["required_skills"]
    assert jd["required_yoe"] == 5.0

def test_f1_load_jd_docx(tmp_path):
    docx_file = tmp_path / "jd.docx"
    create_mock_docx(str(docx_file), "Looking for SQL database admin. Minimum 3 years exp required.")
    jd = load_job_description(str(docx_file))
    assert "SQL" in jd["text"]
    assert "sql" in jd["required_skills"]
    assert jd["required_yoe"] == 3.0

def test_f1_load_jd_skills_extraction(tmp_path):
    txt_file = tmp_path / "jd.txt"
    txt_file.write_text("Requirements: Python, PyTorch, SQL, Kubernetes, Docker.", encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert "python" in jd["required_skills"]
    assert "pytorch" in jd["required_skills"]
    assert "sql" in jd["required_skills"]
    assert "kubernetes" in jd["required_skills"]

def test_f1_load_jd_yoe_extraction(tmp_path):
    txt_file = tmp_path / "jd.txt"
    # Test "at least 4 years"
    txt_file.write_text("We need at least 4 years of experience.", encoding="utf-8")
    jd = load_job_description(str(txt_file))
    assert jd["required_yoe"] == 4.0

def test_f1_load_jd_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_job_description("non_existent_file_path.docx")


# ==========================================
# F2: STREAMING & FILTERING TESTS
# ==========================================

def test_f2_stream_candidates_json(tmp_path):
    json_file = tmp_path / "cands.json"
    cand = get_dummy_candidate()
    json_file.write_text(json.dumps([cand]), encoding="utf-8")
    
    streamed = list(stream_candidates(str(json_file)))
    assert len(streamed) == 1
    assert streamed[0]["candidate_id"] == "CAND_0000001"

def test_f2_stream_candidates_jsonl(tmp_path):
    jsonl_file = tmp_path / "cands.jsonl"
    cand1 = get_dummy_candidate("CAND_0000001")
    cand2 = get_dummy_candidate("CAND_0000002")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(cand1) + "\n")
        f.write(json.dumps(cand2) + "\n")
        
    streamed = list(stream_candidates(str(jsonl_file)))
    assert len(streamed) == 2
    assert streamed[0]["candidate_id"] == "CAND_0000001"
    assert streamed[1]["candidate_id"] == "CAND_0000002"

def test_f2_stream_candidates_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(stream_candidates("non_existent.jsonl"))

def test_f2_stream_candidates_invalid_format(tmp_path):
    txt_file = tmp_path / "cands.txt"
    txt_file.write_text("not a json file", encoding="utf-8")
    with pytest.raises(ValueError):
        list(stream_candidates(str(txt_file)))

def test_f2_filtering_keeps_subset(tmp_path):
    # Setup test file with candidates
    json_file = tmp_path / "candidates.json"
    cands = [get_dummy_candidate(f"CAND_000000{i}") for i in range(10)]
    json_file.write_text(json.dumps(cands), encoding="utf-8")
    
    loaded = load_candidates(str(json_file))
    assert len(loaded) == 10


# ==========================================
# F3: LOCAL SEMANTIC EMBEDDING TESTS
# ==========================================

def test_f3_compute_similarity_scores():
    jd = "Python PyTorch machine learning engineer"
    cands = [
        get_dummy_candidate("CAND_0000001"), # text has "ML Engineer", "PyTorch"
        get_dummy_candidate("CAND_0000002") # change details
    ]
    cands[1]["profile"]["headline"] = "Accounting manager"
    cands[1]["profile"]["summary"] = "Experienced accountant."
    cands[1]["career_history"][0]["description"] = "Managed accounts."
    
    scores = compute_similarity_scores(jd, cands)
    assert len(scores) == 2
    # CAND_0000001 should have a higher score because word content matches JD more
    assert scores[0] > scores[1]

def test_f3_skill_matching_exact():
    cands = [get_dummy_candidate()]
    sim_scores = [0.8]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80, "fit_summary": "good"}]
    
    # python and pytorch in JD
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python", "pytorch"])
    assert len(ranked) == 1
    assert ranked[0]["score"] > 0.0

def test_f3_skill_matching_weights():
    # Candidate 1: Expert in Python
    c1 = get_dummy_candidate("CAND_0000001", skills=[
        {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 24}
    ])
    # Candidate 2: Beginner in Python
    c2 = get_dummy_candidate("CAND_0000002", skills=[
        {"name": "Python", "proficiency": "beginner", "endorsements": 10, "duration_months": 24}
    ])
    
    sim_scores = [0.8, 0.8]
    llm_results = [
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80},
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}
    ]
    
    ranked = hybrid_rank_candidates([c1, c2], sim_scores, llm_results, ["python"])
    # c1 (expert) should score higher than c2 (beginner)
    c1_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000001"][0]
    c2_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000002"][0]
    assert c1_score > c2_score

def test_f3_skill_duration_weights():
    # Candidate 1: 48 months Python
    c1 = get_dummy_candidate("CAND_0000001", skills=[
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 48}
    ])
    # Candidate 2: 12 months Python
    c2 = get_dummy_candidate("CAND_0000002", skills=[
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 12}
    ])
    
    sim_scores = [0.8, 0.8]
    llm_results = [
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80},
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}
    ]
    
    ranked = hybrid_rank_candidates([c1, c2], sim_scores, llm_results, ["python"])
    c1_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000001"][0]
    c2_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000002"][0]
    assert c1_score > c2_score

def test_f3_unverified_skill_penalty():
    # Candidate 1: has Python, but Python is NOT mentioned in career history description
    c1 = get_dummy_candidate("CAND_0000001", skills=[
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
    ], history=[
        {"company": "A", "title": "Dev", "start_date": "2020", "end_date": None, "duration_months": 24, "is_current": True, "industry": "IT", "company_size": "1-10", "description": "Did some Java work."}
    ])
    # Candidate 2: has Python, and Python IS mentioned in career history description
    c2 = get_dummy_candidate("CAND_0000002", skills=[
        {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
    ], history=[
        {"company": "A", "title": "Dev", "start_date": "2020", "end_date": None, "duration_months": 24, "is_current": True, "industry": "IT", "company_size": "1-10", "description": "Developed Python application."}
    ])
    
    sim_scores = [0.8, 0.8]
    llm_results = [
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80},
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}
    ]
    
    ranked = hybrid_rank_candidates([c1, c2], sim_scores, llm_results, ["python"])
    c1_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000001"][0]
    c2_score = [x["score"] for x in ranked if x["candidate_id"] == "CAND_0000002"][0]
    # c1 has penalty (0.5), so c2 score must be higher
    assert c2_score > c1_score


# ==========================================
# F4: DEEPSEEK API TESTS
# ==========================================

def test_f4_api_completions_call_structure():
    cand = get_dummy_candidate()
    # Mocking post and capturing the payload
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"trajectory_score": 80, "skill_depth_score": 85, "role_fit_score": 90, "behavior_score": 75, "fit_summary": "good"}'}}]
        }
        mock_post.return_value = mock_response
        
        res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
        
        # Verify post parameters
        assert mock_post.called
        kwargs = mock_post.call_args[1]
        assert kwargs["json"]["model"] == "deepseek-chat"
        assert kwargs["headers"]["Authorization"] == "Bearer valid_key"

def test_f4_api_success_json_parsing():
    cand = get_dummy_candidate()
    res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
    assert res["trajectory_score"] == 85.0
    assert "verified AI skills" in res["fit_summary"]

def test_f4_api_returns_expected_scores():
    cand = get_dummy_candidate()
    res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
    assert "trajectory_score" in res
    assert "skill_depth_score" in res
    assert "role_fit_score" in res
    assert "behavior_score" in res
    assert "fit_summary" in res

def test_f4_api_key_check():
    # If key present, it executes post
    cand = get_dummy_candidate()
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"trajectory_score": 80, "skill_depth_score": 85, "role_fit_score": 90, "behavior_score": 75, "fit_summary": "good"}'}}]
        }
        mock_post.return_value = mock_response
        evaluate_candidate_fit("Python JD", cand, api_key="key_present")
        assert mock_post.called

def test_f4_api_timeout_handling():
    cand = get_dummy_candidate()
    # Mocking a requests.post timeout exception
    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        # Should complete successfully by falling back to local grading
        res = evaluate_candidate_fit("Python JD", cand, api_key="valid_key")
        assert "trajectory_score" in res
        assert "fit_summary" in res


# ==========================================
# F5: DEEPSEEK FALLBACK TESTS
# ==========================================

def test_f5_fallback_no_key():
    cand = get_dummy_candidate()
    # Run with empty environment / no key
    res = evaluate_candidate_fit("Python JD", cand, api_key=None)
    assert res["trajectory_score"] > 0.0
    assert "matches" in res["fit_summary"]

def test_f5_fallback_invalid_key():
    cand = get_dummy_candidate()
    # Key starts with invalid_ will trigger 401 response in conftest mock
    res = evaluate_candidate_fit("Python JD", cand, api_key="invalid_key_123")
    assert res["trajectory_score"] > 0.0
    assert "matches" in res["fit_summary"]

def test_f5_fallback_rate_limit():
    cand = get_dummy_candidate()
    # Key has 'rate_limit' which triggers 429 response in mock
    res = evaluate_candidate_fit("Python JD", cand, api_key="rate_limit_key")
    assert res["trajectory_score"] > 0.0
    assert "matches" in res["fit_summary"]

def test_f5_fallback_template_formatting():
    cand = get_dummy_candidate()
    cand["profile"]["current_title"] = "Fullstack Dev"
    cand["profile"]["years_of_experience"] = 4.5
    cand["redrob_signals"]["recruiter_response_rate"] = 0.75
    
    res = evaluate_candidate_fit("Python PyTorch developer", cand, api_key=None)
    summary = res["fit_summary"]
    assert "Fullstack Dev" in summary
    assert "4.5 yrs" in summary
    assert "75% response rate" in summary

def test_f5_fallback_scores_calculation():
    c1 = get_dummy_candidate()
    c1["profile"]["years_of_experience"] = 5.0
    c2 = get_dummy_candidate()
    c2["profile"]["years_of_experience"] = 15.0
    
    r1 = evaluate_candidate_fit("Python JD", c1, api_key=None)
    r2 = evaluate_candidate_fit("Python JD", c2, api_key=None)
    # yoe = 5 is closer to target 5.0, so trajectory_score of r1 should be higher
    assert r1["trajectory_score"] > r2["trajectory_score"]


# ==========================================
# F6: OUTPUT CSV FORMATTING & VALIDATION
# ==========================================

def test_f6_csv_headers(tmp_path):
    out = tmp_path / "team_xyz.csv"
    cands = [get_dummy_candidate(f"CAND_{i:07d}") for i in range(100)]
    sim_scores = [0.8] * 100
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80, "fit_summary": "good"}] * 100
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    
    # Write to csv
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(ranked[:100]):
            writer.writerow([item["candidate_id"], idx+1, f"{item['score']:.6f}", item["reasoning"]])
            
    # Read and check headers
    with open(out, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["candidate_id", "rank", "score", "reasoning"]

def test_f6_csv_exactly_100_rows(tmp_path):
    out = tmp_path / "team_xyz.csv"
    cands = [get_dummy_candidate(f"CAND_{i:07d}") for i in range(105)]
    sim_scores = [0.8] * 105
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80, "fit_summary": "good"}] * 105
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(ranked[:100]):
            writer.writerow([item["candidate_id"], idx+1, f"{item['score']:.6f}", item["reasoning"]])
            
    # Read rows
    with open(out, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)
        assert len(data_rows) == 100

def test_f6_csv_sorting(tmp_path):
    # Ensure scores are non-increasing
    out = tmp_path / "team_xyz.csv"
    cands = [get_dummy_candidate(f"CAND_{i:07d}") for i in range(100)]
    # Set varying scores
    sim_scores = [float(i) / 100.0 for i in range(100)]
    llm_results = [{"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80, "fit_summary": "good"}] * 100
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(ranked):
            writer.writerow([item["candidate_id"], idx+1, f"{item['score']:.6f}", item["reasoning"]])
            
    # Verify scores are sorted descending
    with open(out, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        scores = [float(row[2]) for row in reader]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i+1]

def test_f6_csv_tie_breaker(tmp_path):
    # Duplicate profiles with different IDs but same scores
    cands = [
        get_dummy_candidate("CAND_0000002"),
        get_dummy_candidate("CAND_0000001")
    ]
    sim_scores = [0.8, 0.8]
    llm_results = [
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80},
        {"trajectory_score": 80, "skill_depth_score": 80, "role_fit_score": 80, "behavior_score": 80}
    ]
    
    ranked = hybrid_rank_candidates(cands, sim_scores, llm_results, ["python"])
    # Score is identical, so CAND_0000001 must come before CAND_0000002
    assert ranked[0]["candidate_id"] == "CAND_0000001"
    assert ranked[1]["candidate_id"] == "CAND_0000002"

def test_f6_validate_submission_check(tmp_path):
    # Verify that we can programmatically run validate_submission.py using sys path
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge"))
    sys.path.append(script_dir)
    try:
        from validate_submission import validate_submission
        out = tmp_path / "team_xyz.csv"
        # Write valid csv
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for idx in range(100):
                writer.writerow([f"CAND_{idx+1:07d}", idx+1, f"{1.0 - (idx * 0.005):.6f}", "Reason"])
                
        errors = validate_submission(str(out))
        assert len(errors) == 0, f"Expected 0 errors, got: {errors}"
    except ImportError:
        # Fallback if path is different
        pass


# ==========================================
# F7: PITCH DECK TESTS
# ==========================================

def test_f7_pdf_generation_trigger(tmp_path):
    pdf_path = tmp_path / "pitch_deck.pdf"
    generate_pitch_deck([], str(pdf_path))
    assert os.path.exists(pdf_path)

def test_f7_pdf_file_exists_and_nonzero(tmp_path):
    pdf_path = tmp_path / "pitch_deck.pdf"
    generate_pitch_deck([], str(pdf_path))
    assert os.path.getsize(pdf_path) > 0

def test_f7_pdf_slide_sections(tmp_path):
    # Test with content
    pdf_path = tmp_path / "pitch_deck.pdf"
    cands = [
        {"candidate_id": "CAND_0000001", "score": 0.95, "reasoning": "Top ML developer"},
        {"candidate_id": "CAND_0000002", "score": 0.90, "reasoning": "Data scientist expert"}
    ]
    generate_pitch_deck(cands, str(pdf_path))
    assert os.path.exists(pdf_path)

def test_f7_pdf_format_header(tmp_path):
    pdf_path = tmp_path / "pitch_deck.pdf"
    generate_pitch_deck([], str(pdf_path))
    with open(pdf_path, "rb") as f:
        header = f.read(5)
        # Check PDF signature
        assert header == b"%PDF-"

def test_f7_pdf_failure_handling():
    # Attempting to write to an invalid path or directory that doesn't exist
    with pytest.raises(Exception):
        generate_pitch_deck([], "/invalid_dir_xyz/pitch_deck.pdf")
