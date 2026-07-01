import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# 1. Mock sentence_transformers before any test imports it
class MockSentenceTransformer:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, sentences, convert_to_tensor=True):
        # Generate stable mock embedding vector of dimension 384
        if isinstance(sentences, str):
            # Deterministic vector based on string length and first characters
            val = float(len(sentences)) / 100.0
            emb = np.zeros(384)
            emb[0] = val
            emb[1] = 1.0 - val
            return emb
        else:
            embs = []
            for s in sentences:
                val = float(len(s)) / 100.0
                emb = np.zeros(384)
                emb[0] = val
                emb[1] = 1.0 - val
                embs.append(emb)
            return np.array(embs)

# Mock util.cos_sim for Sentence Transformers
def mock_cos_sim(a, b):
    # a and b are numpy arrays or lists
    # Compute simple cosine similarity
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    if len(a_arr.shape) == 1:
        a_arr = np.expand_dims(a_arr, axis=0)
    if len(b_arr.shape) == 1:
        b_arr = np.expand_dims(b_arr, axis=0)
        
    dot_products = np.dot(a_arr, b_arr.T)
    a_norms = np.linalg.norm(a_arr, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b_arr, axis=1, keepdims=True)
    
    sims = dot_products / (a_norms * b_norms.T + 1e-8)
    return sims

mock_st = MagicMock()
mock_st.SentenceTransformer = MockSentenceTransformer
mock_st.util.cos_sim = mock_cos_sim
sys.modules["sentence_transformers"] = mock_st


# 2. Setup mock requests for DeepSeek chat API
@pytest.fixture(autouse=True)
def mock_deepseek_api():
    import requests
    original_post = requests.post

    def mock_post(url, *args, **kwargs):
        if "api.deepseek.com" in url:
            # Check for API Key in headers
            headers = kwargs.get("headers", {})
            auth = headers.get("Authorization", "")
            
            response = MagicMock()
            
            # Simulated invalid key
            if "invalid_key" in auth:
                response.status_code = 401
                response.text = "Unauthorized: Invalid API key"
                return response
            
            # Simulated rate limit
            if "rate_limit" in auth:
                response.status_code = 429
                response.text = "Too Many Requests: Rate limit exceeded"
                return response
            
            # Simulated corrupt response
            if "corrupt" in auth:
                response.status_code = 200
                response.json.return_value = {
                    "choices": [{"message": {"content": "not-valid-json-string-here"}}]
                }
                return response
            
            # Standard successful response
            response.status_code = 200
            response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"trajectory_score": 85.0, "skill_depth_score": 90.0, "role_fit_score": 80.0, "behavior_score": 85.0, "fit_summary": "DeepSeek evaluated: Strong profile with verified AI skills."}'
                        }
                    }
                ]
            }
            return response
            
        return original_post(url, *args, **kwargs)

    with patch("requests.post", side_effect=mock_post) as mock:
        yield mock


# 3. CLI Helper Fixture
@pytest.fixture
def cli_runner():
    """Fixture to execute main.py in-process with mocks active."""
    def _run(args_list, env=None):
        import sys
        import io
        from unittest.mock import patch
        import main
        
        # Setup environment override
        env_patch = patch.dict(os.environ, env or {})
        
        # Setup sys.argv
        argv_patch = patch.object(sys, 'argv', ["main.py"] + args_list)
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        class MockResult:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
                
        with env_patch, argv_patch, patch('sys.stdout', stdout_capture), patch('sys.stderr', stderr_capture):
            try:
                main.main()
                returncode = 0
            except SystemExit as e:
                returncode = e.code if isinstance(e.code, int) else 0
            except Exception as e:
                stderr_capture.write(str(e))
                returncode = 1
                
        return MockResult(returncode, stdout_capture.getvalue(), stderr_capture.getvalue())
    return _run
