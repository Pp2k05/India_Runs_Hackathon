import os
import sys
import time
import subprocess
import ctypes
from ctypes import wintypes

def get_peak_memory_win(pid):
    # Open process with query and read access
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return 0
        
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
        
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        peak_mem = counters.PeakWorkingSetSize
    else:
        peak_mem = 0
        
    ctypes.windll.kernel32.CloseHandle(handle)
    return peak_mem

def main():
    candidates_path = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    out_path = "ranked_candidates.csv"
    jd_path = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx"
    
    print("Starting candidate ranker on full candidate pool...")
    print(f"Candidates file size: {os.path.getsize(candidates_path) / (1024*1024):.2f} MB")
    
    start_time = time.time()
    
    # We clear DEEPSEEK_API_KEY in the environment of the subprocess to force offline/fallback mode.
    # Otherwise, it will try to make real API requests to DeepSeek and hang.
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = ""
    
    # Spawn subprocess
    proc = subprocess.Popen(
        [sys.executable, "rank.py", "--candidates", candidates_path, "--out", out_path, "--job_description", jd_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor peak memory
    peak_bytes = 0
    while proc.poll() is None:
        mem = get_peak_memory_win(proc.pid)
        if mem > peak_bytes:
            peak_bytes = mem
        time.sleep(0.1)
        
    # Get final output
    stdout, stderr = proc.communicate()
    end_time = time.time()
    
    elapsed = end_time - start_time
    peak_mb = peak_bytes / (1024 * 1024)
    
    print("\n--- Execution Results ---")
    print(f"Status: {'Success' if proc.returncode == 0 else 'Failed (code ' + str(proc.returncode) + ')'}")
    print(f"Total Runtime: {elapsed:.2f} seconds")
    print(f"Peak Memory Usage: {peak_mb:.2f} MB")
    
    if proc.returncode != 0:
        print("\nStderr output:")
        print(stderr)
        sys.exit(1)
        
    print("\nStdout output:")
    print(stdout)
    
    # Run the validation script
    val_script = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py"
    if os.path.exists(val_script):
        print("\nRunning submission validator script...")
        val_res = subprocess.run(
            [sys.executable, val_script, out_path],
            capture_output=True,
            text=True
        )
        print(f"Validator Status Code: {val_res.returncode}")
        print("Validator stdout:")
        print(val_res.stdout)
        if val_res.returncode != 0:
            print("Validator stderr:")
            print(val_res.stderr)
            sys.exit(1)
    else:
        print(f"\nWarning: Validator script {val_script} not found.")

if __name__ == "__main__":
    main()
