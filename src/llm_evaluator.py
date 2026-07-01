import os
import json
import time
import requests
from typing import Dict, Any

def evaluate_candidate_fit(job_desc: str, candidate: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    Queries the DeepSeek API to analyze and score a candidate's fit.
    Retries on 429 Rate Limits using exponential backoff.
    Falls back to a dynamic template with unique reasoning per candidate on failures.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    cid = candidate.get("candidate_id", "UNKNOWN") if candidate else "UNKNOWN"
    profile = (candidate.get("profile") or {}) if candidate else {}
    title = profile.get("current_title") or "Unknown Title"
    
    yoe_val = profile.get("years_of_experience")
    yoe = float(yoe_val) if yoe_val is not None else 0.0
    
    skills = (candidate.get("skills") or []) if candidate else []
    skill_names = [s.get("name", "").lower() for s in skills if s and s.get("name")]
    
    signals = (candidate.get("redrob_signals") or {}) if candidate else {}
    
    response_rate_val = signals.get("recruiter_response_rate")
    response_rate = float(response_rate_val) if response_rate_val is not None else 0.5
    
    git_score_val = signals.get("github_activity_score")
    git_score = float(git_score_val) if git_score_val is not None else -1.0
    
    # Precompute metrics for fallback
    jd_lower = job_desc.lower() if job_desc else ""
    matched_skills = [s for s in skill_names if s in jd_lower]
    overlap_count = len(matched_skills)
    
    if api_key and not api_key.startswith("invalid_"):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        prompt = (
            f"Analyze candidate fit for the following job description:\n\n"
            f"JOB DESCRIPTION:\n{job_desc}\n\n"
            f"CANDIDATE PROFILE:\n"
            f"Current Title: {title}\n"
            f"Years of Experience: {yoe}\n"
            f"Skills: {', '.join(skill_names)}\n"
            f"Behavioral signals: Recruiter response rate: {response_rate}, GitHub activity score: {git_score}\n\n"
            f"Provide your analysis in JSON format exactly as follows:\n"
            f"{{\n"
            f"  \"trajectory_score\": <0-100 float>,\n"
            f"  \"skill_depth_score\": <0-100 float>,\n"
            f"  \"role_fit_score\": <0-100 float>,\n"
            f"  \"behavior_score\": <0-100 float>,\n"
            f"  \"fit_summary\": \"<narrative reasoning sentence>\"\n"
            f"}}\n"
            f"Ensure no extra text or markdown formatting is returned outside the JSON."
        )
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert technical recruiter assessing candidate profiles. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        # Exponential backoff parameters
        max_retries = 3
        backoff_factor = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    result_json = response.json()
                    content = result_json["choices"][0]["message"]["content"].strip()
                    parsed = json.loads(content)
                    
                    required_keys = ["trajectory_score", "skill_depth_score", "role_fit_score", "behavior_score", "fit_summary"]
                    if all(k in parsed for k in required_keys):
                        return {
                            "trajectory_score": float(parsed["trajectory_score"]),
                            "skill_depth_score": float(parsed["skill_depth_score"]),
                            "role_fit_score": float(parsed["role_fit_score"]),
                            "behavior_score": float(parsed["behavior_score"]),
                            "fit_summary": str(parsed["fit_summary"])
                        }
                elif response.status_code == 429:
                    time.sleep(backoff_factor ** attempt)
                    continue
                else:
                    break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor ** attempt)
                    continue
                break
                
    # Fallback Dynamic grading logic
    # Calculate scores based on the new logic fields
    # 1. Trajectory score (based on YoE target 5-9, peak 6-8)
    if 6.0 <= yoe <= 8.0:
        traj_score = 100.0
    elif 5.0 <= yoe < 6.0:
        traj_score = 80.0 + 20.0 * (yoe - 5.0)
    elif 8.0 < yoe <= 9.0:
        traj_score = 100.0 - 20.0 * (yoe - 8.0)
    elif yoe < 5.0:
        traj_score = (yoe / 5.0) * 80.0
    else:
        traj_score = max(0.0, 80.0 - 5.0 * (yoe - 9.0))
        
    # 2. Skill depth score
    skill_depth_score = float(min(100.0, overlap_count * 20.0))
    
    # 3. Role fit score (based on current title relevance)
    title_lower = title.lower()
    if any(k in title_lower for k in ["ai", "ml", "machine learning", "nlp", "deep learning"]):
        role_fit_score = 100.0
    elif any(k in title_lower for k in ["engineer", "developer", "programmer"]):
        role_fit_score = 80.0
    else:
        role_fit_score = 40.0
        
    # 4. Behavior score (based on platform response rate)
    behavior_score = float((response_rate * 60.0) + (max(0.0, git_score) * 0.4))
    
    # 5. Programmatic reasoning based on highest/lowest features
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        career_history = candidate.get("career_history") or []
        if not isinstance(career_history, list):
            career_history = []
        history_desc = " ".join([h.get("description", "") for h in career_history if isinstance(h, dict) and h.get("description")])
        candidate_text = f"{headline} {summary} {history_desc}".strip()
        tfidf = vectorizer.fit_transform([job_desc or "", candidate_text])
        similarity = float((tfidf[0] * tfidf[1].T).toarray()[0][0])
    except Exception:
        similarity = 0.0

    channels = {
        "Technical": 0.5 * (skill_depth_score + similarity * 100),
        "Career": 0.5 * (traj_score + role_fit_score),
        "Behavioral": behavior_score
    }
    
    highest_channel = max(channels, key=channels.get)
    remaining_channels = {k: v for k, v in channels.items() if k != highest_channel}
    if remaining_channels:
        lowest_channel = min(remaining_channels, key=remaining_channels.get)
    else:
        lowest_channel = highest_channel

    import hashlib
    h_val = int(hashlib.md5(cid.encode()).hexdigest(), 16)
    
    intros = [
        f"This {title} brings {yoe:.1f} years of relevant experience.",
        f"Offering {yoe:.1f} years of tenure, this {title} is a solid prospect.",
        f"A dedicated {title} with a {yoe:.1f}-year track record.",
        f"Evaluating this {title} profile reveals {yoe:.1f} years of industry presence.",
        f"With {yoe:.1f} years under their belt, this {title} shows clear capability."
    ]
    
    strengths = [
        f"Their primary advantage is a high {highest_channel.lower()} alignment.",
        f"They excel particularly in the {highest_channel.lower()} dimension.",
        f"We highlight their exceptional {highest_channel.lower()} fit for this role.",
        f"The candidate's {highest_channel.lower()} score stands out as their strongest asset.",
        f"A major positive signal is their robust {highest_channel.lower()} metrics."
    ]
    
    weaknesses = [
        f"Conversely, the {lowest_channel.lower()} factors require some development.",
        f"However, we must acknowledge their {lowest_channel.lower()} profile as a relative weak point.",
        f"A notable concern is their lower performance in {lowest_channel.lower()} assessments.",
        f"The primary gap in this application is the {lowest_channel.lower()} evaluation.",
        f"While strong overall, their {lowest_channel.lower()} dimensions are less convincing."
    ]
    
    conclusions = [
        f"They successfully hit {overlap_count} key JD skills and maintain a {int(response_rate * 100)}% response rate.",
        f"Matching {overlap_count} required skills, they also show a {int(response_rate * 100)}% response rate.",
        f"The profile aligns with {overlap_count} core skills, backed by a {int(response_rate * 100)}% response metric.",
        f"JD alignment is confirmed via {overlap_count} overlapping skills and a {int(response_rate * 100)}% recruiter response.",
        f"They demonstrate {overlap_count} relevant skills and a healthy {int(response_rate * 100)}% platform response rate."
    ]
    
    idx_i = h_val % 5
    idx_s = (h_val // 5) % 5
    idx_w = (h_val // 25) % 5
    idx_c = (h_val // 125) % 5
    
    reasoning = f"{intros[idx_i]} {strengths[idx_s]} {weaknesses[idx_w]} {conclusions[idx_c]}"
    
    return {
        "trajectory_score": traj_score,
        "skill_depth_score": skill_depth_score,
        "role_fit_score": role_fit_score,
        "behavior_score": behavior_score,
        "fit_summary": reasoning
    }
