# E2E Test Infra: India Runs Candidate Ranking System

## Test Philosophy
- Opaque-box, requirement-driven testing. Runs against the CLI entry point (`main.py`) to verify system behavior from end-to-end.
- Methodology: Category-Partition, Boundary Value Analysis, Pairwise Combinatorial Testing, and Real-World Workloads.
- Execution environment: Runs on CPU, offline (mocked network APIs).

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| F1| Job Description Parsing | ORIGINAL_REQUEST §1, F1 | 5 | 5 | ✓ |
| F2| Streaming & Filtering | ORIGINAL_REQUEST §1, F2 | 5 | 5 | ✓ |
| F3| Local Semantic Embedding | ORIGINAL_REQUEST §1, F3 | 5 | 5 | ✓ |
| F4| DeepSeek Scoring | ORIGINAL_REQUEST §1, F4 | 5 | 5 | ✓ |
| F5| DeepSeek Fallback | ORIGINAL_REQUEST §1, F5 | 5 | 5 | ✓ |
| F6| Output CSV Formatting | ORIGINAL_REQUEST §1, F6 | 5 | 5 | ✓ |
| F7| Pitch Deck Generation | ORIGINAL_REQUEST §1, F7 | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**: Pytest framework, run via `pytest tests/`.
- **Invocation**: CLI calls to `python main.py` with arguments like `--candidates`, `--job_description`, `--out`, and `--metadata`.
- **Mocks**:
  - Sentence Transformers mocked to bypass large downloads and return consistent unit vector similarity.
  - DeepSeek Chat API mocked to return deterministic, structured JSON responses.
- **Directory Layout**:
  - `tests/` - Contains the E2E test files.
    - `tests/conftest.py` - Setup for temp dirs, CLI execution helpers, and mocks.
    - `tests/test_tier1_feature_coverage.py` - Feature coverage tests.
    - `tests/test_tier2_boundary_cases.py` - Edge and boundary cases.
    - `tests/test_tier3_cross_features.py` - Pairwise combination tests.
    - `tests/test_tier4_real_world.py` - End-to-end user scenarios.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Standard Greenfield Pipeline | F1, F2, F3, F4, F6, F7 | High |
| 2 | Offline/Fallback Grading Run | F1, F2, F3, F5, F6, F7 | High |
| 3 | Keyword Stuffing Detection | F1, F2, F3, F6 | Medium |
| 4 | Mismatched / Incomplete Profile Handling | F2, F3, F4, F6 | High |
| 5 | Equal Score Tie-Breaker Resolution | F2, F3, F6 | Medium |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥5 test cases per feature (at least 35 tests total).
- **Tier 2 (Boundary & Corner Cases)**: ≥5 test cases per feature (at least 35 tests total).
- **Tier 3 (Cross-Feature combinations)**: Pairwise coverage of major feature interactions (at least 6 tests).
- **Tier 4 (Real-World Application)**: Realistic scenarios (at least 5 tests).
