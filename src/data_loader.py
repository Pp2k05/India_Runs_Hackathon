import os
import json
import zipfile
import re
import heapq
import xml.etree.ElementTree as ET
from typing import Generator, List, Dict, Any
from functools import total_ordering
from docx import Document

@total_ordering
class HeapElement:
    def __init__(self, score: float, candidate_id: str, candidate: Dict[str, Any]):
        self.score = score
        self.candidate_id = str(candidate_id)
        self.candidate = candidate

    def __eq__(self, other):
        if not isinstance(other, HeapElement):
            return NotImplemented
        return self.score == other.score and self.candidate_id == other.candidate_id

    def __lt__(self, other):
        if not isinstance(other, HeapElement):
            return NotImplemented
        if self.score != other.score:
            return self.score < other.score
        return self.candidate_id > other.candidate_id

def parse_docx_text(docx_path: str) -> str:
    try:
        doc = Document(docx_path)
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        try:
            with zipfile.ZipFile(docx_path) as z:
                xml_content = z.read('word/document.xml')
                root = ET.fromstring(xml_content)
                texts = []
                for elem in root.iter():
                    if elem.tag.endswith('}t') and elem.text:
                        texts.append(elem.text)
                return '\n'.join(texts)
        except Exception:
            raise ValueError(f"Failed to parse docx {docx_path}: {e}")

def parse_docx(filepath: str) -> str:
    """Alias for compatibility with existing code."""
    return parse_docx_text(filepath)

def load_job_description(filepath: str) -> Dict[str, Any]:
    """
    Extracts text and key requirements from .docx/text files.
    Returns a dict with 'text', 'target_title', 'required_skills', and 'required_yoe'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Job description file not found at {filepath}")
    
    if filepath.endswith('.docx'):
        text = parse_docx_text(filepath)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
    text_lower = text.lower()
    
    # 1. Title Extraction
    target_title = ""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:5]:
        match = re.search(r'(?:job\s+)?title\s*:\s*(.+)', line, re.IGNORECASE)
        if not match:
            match = re.search(r'(?:role|position)\s*:\s*(.+)', line, re.IGNORECASE)
        if match:
            target_title = match.group(1).strip()
            break
    if not target_title and lines:
        first_line = lines[0]
        if not any(word in first_line.lower() for word in ["description", "job details", "overview"]):
            target_title = first_line
        else:
            target_title = "software engineer"
    if not target_title:
        target_title = "software engineer"
            
    # 2. Skill Extraction
    possible_skills = [
        "python", "sql", "pytorch", "tensorflow", "machine learning", 
        "deep learning", "nlp", "llm", "computer vision", "fine-tuning",
        "java", "c++", "aws", "docker", "kubernetes", "git", "spark", "hadoop",
        "scikit-learn", "pandas", "numpy", "bert", "transformers", 
        "prompt engineering", "langchain", "fastapi", "flask", "django"
    ]
    required_skills = []
    for skill in possible_skills:
        if skill == "c++":
            pattern = r'\bc\+\+(?!\w)'
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            required_skills.append(skill)
            
    # 3. Years of Experience (YoE) Extraction
    yoe_patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'(\d+(?:\.\d+)?)\s*-\s*\d+\s*years?',
        r'at least\s*(\d+(?:\.\d+)?)\s*years?',
        r'minimum\s*(\d+(?:\.\d+)?)\s*years?'
    ]
    required_yoe = 0.0
    for pattern in yoe_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                required_yoe = float(match.group(1))
                break
            except ValueError:
                pass
                
    return {
        "text": text,
        "target_title": target_title,
        "required_skills": required_skills,
        "required_yoe": required_yoe
    }

def calculate_title_fit(candidate_title: str, target_title: str) -> float:
    """Computes title fit score [0.0, 1.0] between candidate title and target job title."""
    if not candidate_title or not target_title:
        return 0.0
        
    c_title = candidate_title.lower().strip()
    t_title = target_title.lower().strip()
    
    if t_title in c_title or c_title in t_title:
        return 1.0
        
    synonyms = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "llm": "large language model",
    }
    
    # Clean up and replace synonyms (matching whole words)
    def replace_synonyms(text: str) -> str:
        words = re.findall(r'\b\w+\b', text)
        replaced_words = [synonyms.get(w, w) for w in words]
        return " ".join(replaced_words)
        
    c_title_replaced = replace_synonyms(c_title)
    t_title_replaced = replace_synonyms(t_title)
    
    if t_title_replaced in c_title_replaced or c_title_replaced in t_title_replaced:
        return 1.0
        
    stop_words = {"and", "or", "of", "in", "for", "the", "a", "an", "at", "to", "with", "specialist"}
    c_tokens = set(w for w in re.findall(r'\b\w+\b', c_title_replaced) if w not in stop_words)
    t_tokens = set(w for w in re.findall(r'\b\w+\b', t_title_replaced) if w not in stop_words)
    
    if not t_tokens:
        return 0.0
        
    overlap = c_tokens.intersection(t_tokens)
    return len(overlap) / len(t_tokens)

def score_candidate_title(candidate: Dict[str, Any], target_title: str) -> float:
    """Scores candidate title fit using current title (weight 1.0) and past titles (weight 0.5)."""
    profile = candidate.get('profile') or {}
    current_title = profile.get('current_title') or ''
    current_fit = calculate_title_fit(str(current_title), target_title)
    
    past_fits = []
    career_history = candidate.get('career_history') or []
    if not isinstance(career_history, list):
        career_history = []
    for job in career_history:
        if isinstance(job, dict):
            past_title = job.get('title') or ''
            if past_title:
                past_fits.append(calculate_title_fit(str(past_title), target_title))
        
    max_past_fit = max(past_fits) if past_fits else 0.0
    return min(1.0, current_fit + 0.5 * max_past_fit)

def score_candidate_fast(candidate: Dict[str, Any], jd_requirements: Dict[str, Any]) -> float:
    """
    Computes fast filter score for a candidate profile.
    Formula: 0.4 * title_fit + 0.4 * yoe_fit + 0.2 * basic_skill_overlap
    """
    target_title = jd_requirements.get('target_title', '')
    required_yoe = jd_requirements.get('required_yoe', 0.0)
    required_skills = jd_requirements.get('required_skills', [])
    
    title_fit = score_candidate_title(candidate, target_title)
    
    profile = candidate.get('profile') or {}
    yoe_val = profile.get('years_of_experience')
    if yoe_val is None:
        candidate_yoe = 0.0
    else:
        try:
            candidate_yoe = float(yoe_val)
        except (ValueError, TypeError):
            candidate_yoe = 0.0
            
    if required_yoe <= 0.0:
        yoe_fit = 1.0
    else:
        yoe_fit = min(1.0, candidate_yoe / required_yoe)
        
    skills = candidate.get('skills') or []
    if not isinstance(skills, list):
        skills = []
    candidate_skills = set(
        str(s.get('name', '')).lower() 
        for s in skills 
        if isinstance(s, dict) and s.get('name')
    )
    req_skills_set = set(str(s).lower() for s in required_skills if s)
    
    if not req_skills_set:
        skill_overlap = 1.0
    else:
        overlap_count = len(candidate_skills.intersection(req_skills_set))
        skill_overlap = overlap_count / len(req_skills_set)
        
    final_score = 0.4 * title_fit + 0.4 * yoe_fit + 0.2 * skill_overlap
    return round(final_score, 4)

def stream_candidates(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Streams candidates line-by-line from a JSONL file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Candidates file not found at {filepath}")
        
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                yield data
    else:
        raise ValueError("Unsupported candidate file format. Must be .json or .jsonl")

def filter_top_candidates(
    candidate_stream: Generator[Dict[str, Any], None, None],
    jd_requirements: Dict[str, Any],
    top_n: int = 500
) -> List[Dict[str, Any]]:
    """
    Fast streaming filter to find the top_n candidates from the stream.
    Uses a min-heap to keep track of the top candidates while maintaining constant memory.
    """
    heap = []
    
    for idx, candidate in enumerate(candidate_stream):
        candidate_id = candidate.get('candidate_id', '')
        if not candidate_id:
            candidate_id = f"CAND_{idx+1:07d}"
            candidate['candidate_id'] = candidate_id
            
        score = score_candidate_fast(candidate, jd_requirements)
        
        heap_item = HeapElement(score, candidate_id, candidate)
        
        if len(heap) < top_n:
            heapq.heappush(heap, heap_item)
        else:
            if heap_item > heap[0]:
                heapq.heappushpop(heap, heap_item)
                
    sorted_elements = sorted(heap, key=lambda x: (-x.score, x.candidate_id))
    return [el.candidate for el in sorted_elements]

def load_candidates(data_dir_or_file: str) -> List[Dict[str, Any]]:
    """
    Parses all candidate profiles and returns them as a list.
    If data_dir_or_file is a directory, it looks for candidates.jsonl or sample_candidates.json.
    """
    if os.path.isdir(data_dir_or_file):
        for filename in ['candidates.jsonl', 'sample_candidates.json']:
            path = os.path.join(data_dir_or_file, filename)
            if os.path.exists(path):
                return list(stream_candidates(path))
        raise FileNotFoundError(f"No candidates.jsonl or sample_candidates.json found in {data_dir_or_file}")
    else:
        return list(stream_candidates(data_dir_or_file))
