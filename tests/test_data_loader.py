import os
import json
import tempfile
import zipfile
import unittest
import xml.etree.ElementTree as ET
from typing import Dict, Any

from src.data_loader import (
    parse_docx_text,
    load_job_description,
    stream_candidates,
    filter_top_candidates,
    score_candidate_fast,
    calculate_title_fit
)

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.test_dir.cleanup()

    def create_mock_docx(self, filename: str, text: str) -> str:
        filepath = os.path.join(self.test_dir.name, filename)
        from docx import Document
        doc = Document()
        for line in text.split('\n'):
            doc.add_paragraph(line)
        doc.save(filepath)
        return filepath

    def get_dummy_candidate(self, cid: str, title: str, yoe: float, skills: list) -> Dict[str, Any]:
        return {
            "candidate_id": cid,
            "profile": {
                "anonymized_name": "Test Candidate",
                "headline": "Specialist",
                "summary": "Summary text",
                "years_of_experience": yoe,
                "current_title": title
            },
            "career_history": [
                {
                    "company": "PrevCorp",
                    "title": title,
                    "description": "Used various tech"
                }
            ],
            "skills": [{"name": s} for s in skills]
        }

    def test_parse_docx_text_and_job_description_extraction(self):
        # 1. Test using a mock docx file
        jd_text = (
            "Job Title: Machine Learning Engineer\n"
            "Requirements:\n"
            "- Python and PyTorch\n"
            "- At least 5 years of experience in deep learning\n"
        )
        mock_path = self.create_mock_docx("job_description.docx", jd_text)
        
        # Verify parse_docx_text
        extracted_text = parse_docx_text(mock_path)
        self.assertIn("Machine Learning Engineer", extracted_text)
        self.assertIn("At least 5 years", extracted_text)
        
        # Verify load_job_description
        jd = load_job_description(mock_path)
        self.assertEqual(jd["target_title"], "Machine Learning Engineer")
        self.assertIn("python", jd["required_skills"])
        self.assertIn("pytorch", jd["required_skills"])
        self.assertEqual(jd["required_yoe"], 5.0)

        # 2. Try running on the real job_description.docx if available
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "job_description.docx"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "PUB_India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "job_description.docx"),
            "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx",
            "data/PUB_India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx"
        ]
        real_path = None
        for p in possible_paths:
            if os.path.exists(p):
                real_path = p
                break
                
        if real_path:
            real_jd = load_job_description(real_path)
            self.assertTrue(len(real_jd["text"]) > 0)
            self.assertIsNotNone(real_jd["target_title"])
            self.assertIsInstance(real_jd["required_skills"], list)
            self.assertIsInstance(real_jd["required_yoe"], float)
            print(f"Verified real job description: Title='{real_jd['target_title']}', YoE={real_jd['required_yoe']}, Skills={real_jd['required_skills']}")

    def test_stream_candidates(self):
        # 1. Test stream_candidates on JSON
        json_path = os.path.join(self.test_dir.name, "candidates.json")
        cands = [
            self.get_dummy_candidate("CAND_0000001", "Developer", 3.0, ["Python"]),
            self.get_dummy_candidate("CAND_0000002", "Manager", 8.0, ["Java"])
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cands, f)
            
        streamed = list(stream_candidates(json_path))
        self.assertEqual(len(streamed), 2)
        self.assertEqual(streamed[0]["candidate_id"], "CAND_0000001")
        self.assertEqual(streamed[1]["candidate_id"], "CAND_0000002")

        # 2. Test stream_candidates on JSONL
        jsonl_path = os.path.join(self.test_dir.name, "candidates.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
                
        streamed_l = list(stream_candidates(jsonl_path))
        self.assertEqual(len(streamed_l), 2)
        self.assertEqual(streamed_l[0]["candidate_id"], "CAND_0000001")
        self.assertEqual(streamed_l[1]["candidate_id"], "CAND_0000002")

    def test_filter_top_candidates(self):
        # Setup JD requirements
        jd_requirements = {
            "target_title": "ML Engineer",
            "required_yoe": 5.0,
            "required_skills": ["Python", "PyTorch"]
        }
        
        # Construct candidates with varying suitability
        # C1: Perfect fit (Score 1.0)
        c1 = self.get_dummy_candidate("CAND_0000001", "ML Engineer", 5.0, ["Python", "PyTorch"])
        # C2: Perfect fit (Score 1.0) - tie breaker
        c2 = self.get_dummy_candidate("CAND_0000002", "ML Engineer", 5.0, ["Python", "PyTorch"])
        # C3: Perfect fit (Score 1.0) - tie breaker loser
        c3 = self.get_dummy_candidate("CAND_0000003", "ML Engineer", 5.0, ["Python", "PyTorch"])
        # C4: Partial fit (Title & YoE match, no skills: 0.4*1.0 + 0.4*1.0 + 0.2*0.0 = 0.8)
        c4 = self.get_dummy_candidate("CAND_0000004", "ML Engineer", 5.0, [])
        # C5: Poor fit (No title, low YoE: 0.4*0.0 + 0.4*0.4 + 0.2*0.0 = 0.16)
        c5 = self.get_dummy_candidate("CAND_0000005", "Backend Engineer", 2.0, [])
        
        # Test 1: filter to top 2 (should be C1 and C2 because of tie-breaking on candidate_id)
        candidates_stream = (c for c in [c3, c2, c1, c4, c5])
        top_2 = filter_top_candidates(candidates_stream, jd_requirements, top_n=2)
        
        self.assertEqual(len(top_2), 2)
        self.assertEqual(top_2[0]["candidate_id"], "CAND_0000001")
        self.assertEqual(top_2[1]["candidate_id"], "CAND_0000002")
        
        # Test 2: filter to top 4 (should sort by score descending, then candidate_id ascending)
        candidates_stream = (c for c in [c5, c4, c3, c2, c1])
        top_4 = filter_top_candidates(candidates_stream, jd_requirements, top_n=4)
        
        self.assertEqual(len(top_4), 4)
        self.assertEqual(top_4[0]["candidate_id"], "CAND_0000001")
        self.assertEqual(top_4[1]["candidate_id"], "CAND_0000002")
        self.assertEqual(top_4[2]["candidate_id"], "CAND_0000003")
        self.assertEqual(top_4[3]["candidate_id"], "CAND_0000004")

    def test_heap_element_comparisons_and_coercion(self):
        from src.data_loader import HeapElement
        
        # Test candidate_id coercion to string
        h1 = HeapElement(0.85, 12345, {"id": 12345})
        self.assertEqual(h1.candidate_id, "12345")
        
        h2 = HeapElement(0.85, "12345", {"id": 12345})
        h3 = HeapElement(0.85, "12346", {"id": 12346})
        h4 = HeapElement(0.90, "12347", {"id": 12347})
        
        # Test eq
        self.assertEqual(h1, h2)
        self.assertNotEqual(h2, h3)
        
        # Test lt (lower score is lt; for same score, larger candidate_id is lt)
        self.assertTrue(h3 < h4)  # 0.85 < 0.90
        self.assertTrue(h3 < h2)  # both 0.85, "12346" > "12345"
        self.assertFalse(h2 < h3)

    def test_score_candidate_fast_robustness(self):
        jd_requirements = {
            "target_title": "ML Engineer",
            "required_yoe": 5.0,
            "required_skills": ["Python", "PyTorch"]
        }
        
        # Test candidate with None profile, None skills, None career history
        c_bad = {
            "candidate_id": "CAND_BAD",
            "profile": None,
            "career_history": None,
            "skills": None
        }
        
        # This should not raise any exceptions
        score = score_candidate_fast(c_bad, jd_requirements)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0) # fit=0.0, yoe=0.0, skills=0.0 -> score=0.0
        
        # Test candidate with partially invalid structures
        c_partial = {
            "candidate_id": "CAND_PART",
            "profile": {
                "years_of_experience": "not a number",
                "current_title": None
            },
            "career_history": [
                "invalid_history_item",
                {"title": "ML Engineer"}
            ],
            "skills": [
                "not a dict skill",
                {"name": "Python"}
            ]
        }
        score_partial = score_candidate_fast(c_partial, jd_requirements)
        self.assertIsInstance(score_partial, float)

if __name__ == '__main__':
    unittest.main()
