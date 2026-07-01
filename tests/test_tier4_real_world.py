import os
import csv
import json
import pytest
from unittest.mock import patch, MagicMock

# Helper to write candidate JSONL file
def write_temp_candidates(filepath, count=105):
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(count):
            cand = {
                "candidate_id": f"CAND_{i+1:07d}",
                "profile": {
                    "anonymized_name": f"Candidate {i+1}",
                    "headline": "AI Engineer",
                    "summary": "Specialist in Deep Learning.",
                    "location": "Bengaluru",
                    "country": "India",
                    "years_of_experience": 5.0,
                    "current_title": "ML Engineer",
                    "current_company": "AILab",
                    "current_company_size": "51-200",
                    "current_industry": "Software"
                },
                "career_history": [
                    {
                        "company": "AILab",
                        "title": "ML Engineer",
                        "start_date": "2021-01-01",
                        "end_date": None,
                        "duration_months": 36,
                        "is_current": True,
                        "industry": "Software",
                        "company_size": "51-200",
                        "description": "Implemented AI models in Python and PyTorch."
                    }
                ],
                "education": [],
                "skills": [
                    {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
                    {"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
                ],
                "redrob_signals": {
                    "profile_completeness_score": 90.0,
                    "signup_date": "2020-01-01",
                    "last_active_date": "2026-06-01",
                    "open_to_work_flag": True,
                    "profile_views_received_30d": 10,
                    "applications_submitted_30d": 3,
                    "recruiter_response_rate": 0.8,
                    "avg_response_time_hours": 1.0,
                    "skill_assessment_scores": {},
                    "connection_count": 100,
                    "endorsements_received": 10,
                    "notice_period_days": 30,
                    "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
                    "preferred_work_mode": "remote",
                    "willing_to_relocate": True,
                    "github_activity_score": 80.0,
                    "search_appearance_30d": 20,
                    "saved_by_recruiters_30d": 5,
                    "interview_completion_rate": 0.9,
                    "offer_acceptance_rate": 0.8,
                    "verified_email": True,
                    "verified_phone": True,
                    "linkedin_connected": True
                }
            }
            f.write(json.dumps(cand) + "\n")

# ==========================================
# TIER 4: REAL WORLD SCENARIO E2E TESTS
# ==========================================

def test_scenario1_greenfield_pipeline(tmp_path, cli_runner):
    """Scenario 1: Standard Greenfield Pipeline (full flow, valid inputs, DeepSeek API active)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("We are hiring a PyTorch and Python developer.", encoding="utf-8")
    
    cand_file = tmp_path / "candidates.jsonl"
    write_temp_candidates(cand_file, count=110)
    
    out_csv = tmp_path / "team_greenfield.csv"
    
    # Run CLI with valid API key
    res = cli_runner([
        "--candidates", str(cand_file),
        "--job_description", str(jd_file),
        "--out", str(out_csv)
    ], env={"DEEPSEEK_API_KEY": "valid_key"})
    
    # Assert execution succeeds
    assert res.returncode == 0, f"CLI execution failed: {res.stderr}"
    
    # Assert CSV exists and is valid
    assert os.path.exists(out_csv)
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["candidate_id", "rank", "score", "reasoning"]
        data_rows = list(reader)
        assert len(data_rows) == 100
        
    # Assert Pitch Deck exists
    deck_path = os.path.join(os.path.dirname(out_csv), "pitch_deck.pdf")
    assert os.path.exists(deck_path)
    assert os.path.getsize(deck_path) > 0

def test_scenario2_offline_fallback_run(tmp_path, cli_runner):
    """Scenario 2: Offline/Fallback Grading Run (no API key, fallback template reasoning)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("Looking for Python development skills.", encoding="utf-8")
    
    cand_file = tmp_path / "candidates.jsonl"
    write_temp_candidates(cand_file, count=102)
    
    out_csv = tmp_path / "team_offline.csv"
    
    # Run CLI with no API key
    res = cli_runner([
        "--candidates", str(cand_file),
        "--job_description", str(jd_file),
        "--out", str(out_csv)
    ], env={"DEEPSEEK_API_KEY": ""}) # empty key
    
    # Assert execution succeeds
    assert res.returncode == 0
    assert os.path.exists(out_csv)
    
    # Assert fallback template reasoning is visible in CSV
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        first_row = next(reader)
        reasoning = first_row[3]
        # Should match fallback style: "[title] with [yoe] yrs... matches [x] key skills..."
        assert "yrs of experience" in reasoning
        assert "matches" in reasoning

def test_scenario3_keyword_stuffing_detection(tmp_path, cli_runner):
    """Scenario 3: Keyword Stuffing Detection (verify practitioner beats gamed profiles)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("We need a PyTorch Python engineer.", encoding="utf-8")
    
    # Write two specific candidates
    # CAND_0000001: Genuine ML developer with PyTorch in description
    c1 = {
        "candidate_id": "CAND_0000001",
        "profile": {"anonymized_name": "P1", "headline": "ML", "summary": "ML summary", "location": "BLR", "country": "IN", "years_of_experience": 5.0, "current_title": "ML Engineer", "current_company": "X", "current_company_size": "11-50", "current_industry": "IT"},
        "career_history": [{"company": "X", "title": "ML Eng", "start_date": "2020", "end_date": None, "duration_months": 24, "is_current": True, "industry": "IT", "company_size": "11-50", "description": "Developed PyTorch models."}],
        "education": [],
        "skills": [{"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}],
        "redrob_signals": {"profile_completeness_score": 80.0, "signup_date": "2020", "last_active_date": "2026", "open_to_work_flag": True, "profile_views_received_30d": 1, "applications_submitted_30d": 1, "recruiter_response_rate": 0.5, "avg_response_time_hours": 1, "skill_assessment_scores": {}, "connection_count": 10, "endorsements_received": 1, "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 5, "max": 10}, "preferred_work_mode": "remote", "willing_to_relocate": True, "github_activity_score": 50, "search_appearance_30d": 1, "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.9, "offer_acceptance_rate": 0.8, "verified_email": True, "verified_phone": True, "linkedin_connected": True}
    }
    # CAND_0000002: Keyword stuffer (has PyTorch skill, but no PyTorch in description)
    c2 = {
        "candidate_id": "CAND_0000002",
        "profile": {"anonymized_name": "P2", "headline": "Dev", "summary": "Dev summary", "location": "BLR", "country": "IN", "years_of_experience": 5.0, "current_title": "Software Dev", "current_company": "X", "current_company_size": "11-50", "current_industry": "IT"},
        "career_history": [{"company": "X", "title": "Dev", "start_date": "2020", "end_date": None, "duration_months": 24, "is_current": True, "industry": "IT", "company_size": "11-50", "description": "Did HTML coding."}], # No PyTorch
        "education": [],
        "skills": [{"name": "PyTorch", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}],
        "redrob_signals": {"profile_completeness_score": 80.0, "signup_date": "2020", "last_active_date": "2026", "open_to_work_flag": True, "profile_views_received_30d": 1, "applications_submitted_30d": 1, "recruiter_response_rate": 0.5, "avg_response_time_hours": 1, "skill_assessment_scores": {}, "connection_count": 10, "endorsements_received": 1, "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 5, "max": 10}, "preferred_work_mode": "remote", "willing_to_relocate": True, "github_activity_score": 50, "search_appearance_30d": 1, "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.9, "offer_acceptance_rate": 0.8, "verified_email": True, "verified_phone": True, "linkedin_connected": True}
    }
    
    cand_file = tmp_path / "cands.jsonl"
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(c1) + "\n")
        f.write(json.dumps(c2) + "\n")
        
    out_csv = tmp_path / "team_stuffed.csv"
    
    res = cli_runner([
        "--candidates", str(cand_file),
        "--job_description", str(jd_file),
        "--out", str(out_csv)
    ])
    
    assert res.returncode == 0
    
    # CAND_0000001 (genuine) must be ranked above CAND_0000002 (stuffed/penalized)
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        row1 = next(reader)
        row2 = next(reader)
        
        assert row1[0] == "CAND_0000001"
        assert row2[0] == "CAND_0000002"

def test_scenario4_mismatched_incomplete_profile_handling(tmp_path, cli_runner):
    """Scenario 4: Mismatched / Incomplete Profile Handling (robust parsing, default scoring)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("We need an engineer.", encoding="utf-8")
    
    # Candidate with missing/null properties
    c1 = {
        "candidate_id": "CAND_0000001",
        "profile": {
            # missing current_title, summary, headline
            "anonymized_name": "P1",
            "years_of_experience": 2.0,
            "current_company": "X",
            "current_company_size": "1-10",
            "current_industry": "IT"
        },
        "career_history": [], # empty career history
        "skills": [], # empty skills
        "redrob_signals": {} # empty signals
    }
    
    cand_file = tmp_path / "cands.jsonl"
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(c1) + "\n")
        
    out_csv = tmp_path / "team_incomplete.csv"
    res = cli_runner([
        "--candidates", str(cand_file),
        "--job_description", str(jd_file),
        "--out", str(out_csv)
    ])
    
    assert res.returncode == 0
    assert os.path.exists(out_csv)
    
    # CSV should still have 100 rows due to padding
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
        assert len(rows) == 100

def test_scenario5_equal_score_tie_breaker_resolution(tmp_path, cli_runner):
    """Scenario 5: Equal Score Tie-Breaker Resolution (sorted alphabetically by candidate_id)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("Looking for PyTorch developer.", encoding="utf-8")
    
    # Generate 5 candidates with identical credentials
    cand_file = tmp_path / "cands.jsonl"
    with open(cand_file, "w", encoding="utf-8") as f:
        # Write them in reverse order
        for i in [5, 3, 1, 2, 4]:
            c = {
                "candidate_id": f"CAND_000000{i}",
                "profile": {"anonymized_name": f"P{i}", "headline": "Dev", "summary": "Dev", "location": "BLR", "country": "IN", "years_of_experience": 5.0, "current_title": "Engineer", "current_company": "X", "current_company_size": "1-10", "current_industry": "IT"},
                "career_history": [],
                "skills": [],
                "redrob_signals": {}
            }
            f.write(json.dumps(c) + "\n")
            
    out_csv = tmp_path / "team_ties.csv"
    res = cli_runner([
        "--candidates", str(cand_file),
        "--job_description", str(jd_file),
        "--out", str(out_csv)
    ])
    
    assert res.returncode == 0
    
    # Check that candidates are sorted alphabetically CAND_0000001, CAND_0000002, CAND_0000003, ...
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        row1 = next(reader)
        row2 = next(reader)
        row3 = next(reader)
        row4 = next(reader)
        row5 = next(reader)
        
        assert row1[0] == "CAND_0000001"
        assert row2[0] == "CAND_0000002"
        assert row3[0] == "CAND_0000003"
        assert row4[0] == "CAND_0000004"
        assert row5[0] == "CAND_0000005"
