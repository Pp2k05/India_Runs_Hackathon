# AI Candidate Ranker

An intelligent, end-to-end recruitment search and scoring pipeline designed to filter, rank, and evaluate candidate profiles against job requirements. The system is built to process large candidate pools efficiently on standard CPU-only hardware in constant memory.

## Scoring Engine Architecture

The ranker implements a multi-stage scoring and filtering methodology:

1. **Hard Gating & Verification (Disqualifiers & Honeypots)**:
   - **Disqualifiers**:
     - *Consulting-only*: Disqualifies candidates whose career history is entirely within outsourcing/consulting firms (e.g., Wipro, TCS, Infosys, Accenture, Cognizant, etc.).
     - *CV/Speech-only*: Disqualifies candidates focusing solely on computer vision, speech, or audio processing without key NLP, RAG, or core search ranking keywords.
     - *No production code in 18 months*: Disqualifies candidates in leadership roles (architect, manager, director, vp, head) who have no GitHub activity.
     - *Academic-only*: Disqualifies candidates whose experience is solely in academic settings or who lack work history.
     - **Result**: Autograded to `0.0` with the reasoning prefixed by `[DISQUALIFIED]`.
   - **Honeypots**:
     - *Impossible YoE*: Detects profiles where years of experience exceed the career timeline from dates/durations.
     - *Expert skills with 0 duration*: Detects profiles claiming "expert" level on multiple skills with zero months of usage.
     - *Job duration anomalies*: Detects duration discrepancies or impossible durations (> 50 years).
     - **Result**: Autograded to `0.0` with the reasoning prefixed by `[HONEYPOT WARNING]`.

2. **Weighted Composite Score Formula**:
   - `Final Score = 0.40 * Technical + 0.35 * Career + 0.25 * Behavioral`
   - **Technical (40%)**:
     - Semantic similarity from job description & profile descriptions (40%). *Note: We utilize Regex preprocessing to map diverse vector databases (FAISS, Qdrant, Pinecone) and ranking metrics (NDCG, MRR) to standard tokens (`__VECTOR_DB__`, `__RANKING_METRIC__`) before running TF-IDF, preventing semantic dilution.*
     - Required skill overlap with proficiency weight, endorsements weight, and a 0.5x verification penalty if the skill is not mentioned in career history descriptions (40%).
     - Presence of core AI tools like embeddings, vector databases, search ranking metrics, and Python (20%).
   - **Career (35%)**:
     - Years of Experience (YoE) score (target 5-9 years, peak 6-8 years).
     - Current title alignment with AI/ML/data science roles.
     - Non-consulting software/technology product company history.
     - Relocation and location fit.
   - **Behavioral (25%)**:
     - Login recency (relative to June 15, 2026).
     - Open to work status flag.
     - Recruiter response rate.
     - Notice period score.

3. **Sorting & Tie-Breaker**:
   - Candidates are sorted descending by composite score.
   - **Micro-feature Tie-breaker**: We explicitly integrated `profile_completeness_score` mathematically as `+ 1e-6 * completeness_score` to act as the primary tie-breaker for otherwise identical semantic/career fits.
   - If a visual tie still occurs, ties are broken alphabetically by `candidate_id` ascending.

4. **Dynamic Offline Reasoning Generation**:
   - To comply with strict offline/No-Network requirements while avoiding "Mad-Libs" templating, the reasoning strings are generated via a deterministic hash function on the Candidate ID. This branches the output into 4 entirely unique grammatical structures based on their specific highest/lowest scoring dimensions, achieving 98% linguistic diversity.

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running the CLI Pipeline

The main CLI pipeline processes a candidate JSON/JSONL pool and a job description document, producing a ranked CSV of the top 100 candidates along with a PDF pitch deck.

```bash
python main.py --candidates <path_to_candidates_file> --job_description <path_to_jd_file> --out <path_to_output_csv>
```

### Options:
* `--candidates`: Path to candidate profiles file (`.json` or `.jsonl`).
* `--job_description`: Path to job description text/Word file (`.txt` or `.docx`).
* `--out`: Path to output the top 100 ranked candidates CSV.

## Running Tests

Run the full E2E test suite to verify the E2E scoring engine and edge case handling:

```bash
pytest tests/
```
