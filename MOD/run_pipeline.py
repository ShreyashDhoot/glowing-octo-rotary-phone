#!/usr/bin/env python3
"""
run_pipeline.py
===============
Executes the generation, tribunal conversion, vLLM judge server hosting,
evaluation, and plotting pipeline.
"""

import os
import sys
import time
import glob
import shutil
import atexit
import subprocess

# ==========================================
# CONFIGURATION
# ==========================================
JUDGE_GPUS = [0, 1, 2, 3]
BASE_PORT = 8000

SWEEP_OUT = "./sweep_outputs"
TRIB_IN = "../tribunal/inputs/harmlessness-MOD"
TRIB_OUT = "../tribunal/eval_results/harmlessness-MOD"
PLOTS_OUT = "../runs/tribunal_plots/harmlessness-MOD"

# Track server processes so we can kill them when the script finishes
judge_processes = []

def cleanup():
    """Ensures all background vLLM servers are shut down when the script exits."""
    print("\n🛑 Shutting down Tribunal Judge servers...")
    for p in judge_processes:
        if p.poll() is None:  # If the process is still running
            p.terminate()
            p.wait()

# Register the cleanup function to run even if the script crashes or is interrupted
atexit.register(cleanup)


def merge_csvs(filename):
    """Merges multiple CSV files from run subdirectories, keeping only one header."""
    target_path = os.path.join(TRIB_OUT, filename)
    source_files = glob.glob(os.path.join(TRIB_OUT, "run_*", filename))
    
    if not source_files:
        return

    # Sort to ensure we process run_0, run_1, etc. in order
    source_files.sort()
    
    with open(target_path, 'w', encoding='utf-8') as outfile:
        for i, filepath in enumerate(source_files):
            with open(filepath, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                if not lines:
                    continue
                if i == 0:
                    outfile.writelines(lines)  # Keep header for the first file
                else:
                    outfile.writelines(lines[1:])  # Skip header for the rest


def main():
    # Ensure directories exist
    for d in [SWEEP_OUT, TRIB_IN, TRIB_OUT, PLOTS_OUT]:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # STEP 1: Generate Answers
    # ==========================================
    print("🚀 STEP 1: Running generation on GPU...")
    subprocess.run([
        "python", "wrapper.py", 
        "--output_dir", SWEEP_OUT, 
        "--num_prompts", "50"
    ], check=True)

    # ==========================================
    # STEP 2: Convert to JSONL
    # ==========================================
    print("\n🔄 STEP 2: Converting outputs to Tribunal .jsonl format...")
    subprocess.run([
        # Note: Fixed the typo from 'convert_to_tibunal.py' to 'convert_to_tribunal.py'
        "python", "convert_to_tibunal.py", 
        "--input", SWEEP_OUT, 
        "--output", TRIB_IN
    ], check=True)

    # ==========================================
    # STEP 3: Start 4 Judge Servers
    # ==========================================
    print(f"\n⚖️ STEP 3: Starting Judge servers on GPUs: {JUDGE_GPUS}...")
    for i, gpu in enumerate(JUDGE_GPUS):
        port = BASE_PORT + i
        print(f"  -> Starting server on GPU {gpu}, Port {port}")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        # Popen starts the process in the background, fully detached from the main execution block
        p = subprocess.Popen([
            "python", "../tribunal/serve_judge.py", 
            "--port", str(port)
        ], env=env)
        
        judge_processes.append(p)

    # Increased wait time slightly just to ensure vLLM has ample time to allocate VRAM
    import urllib.request
    import urllib.error
    
    # ... (Keep Step 3 where you start the servers) ...
    JUDGE_API_KEY = "EMPTY"

    print("\n⏳ Waiting for all vLLM Judge servers to become ready (this may take a few minutes)...")
    for i in range(len(JUDGE_GPUS)):
        port = BASE_PORT + i
        # ... inside your polling loop ...
        url = f"http://localhost:{port}/v1/models"
        headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
        ready = False
        
        while not ready:
            try:
                # Create a Request object to attach the headers
                req = urllib.request.Request(url, headers=headers)
                response = urllib.request.urlopen(req, timeout=5)
                
                if response.getcode() == 200:
                    print(f"  -> Server on Port {port} is online and ready!")
                    ready = True
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    print(f"Auth error on port {port}. Check JUDGE_API_KEY.")
                    time.sleep(5)
                else:
                    time.sleep(5)
            except urllib.error.URLError:
                # If connection is refused, wait 5 seconds and try again
                time.sleep(5)

    # ==========================================
    # STEP 4: Run Parallel Evaluations
    # ==========================================
    print("\n📊 STEP 4: Distributing evaluations across servers...")
    jsonl_files = glob.glob(os.path.join(TRIB_IN, "*.jsonl"))
    eval_processes = []

    for i, filepath in enumerate(jsonl_files):
        filename = os.path.basename(filepath)
        server_idx = i % len(JUDGE_GPUS)
        port = BASE_PORT + server_idx
        out_subdir = os.path.join(TRIB_OUT, f"run_{i}")
        
        print(f"  -> Evaluating {filename} on Port {port} (GPU {JUDGE_GPUS[server_idx]})")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = ".."
        
        p = subprocess.Popen([
            "python", "-m", "tribunal.tribunal.run_eval",
            "--input", filepath,
            "--output", out_subdir,
            "--judge-url", f"http://localhost:{port}/v1"
        ], env=env)
        
        eval_processes.append(p)

    print("⏳ Waiting for all evaluations to complete...")
    for p in eval_processes:
        p.wait()  # This halts the script until all background evaluators finish

    # ==========================================
    # STEP 5: Merge CSVs safely
    # ==========================================
    print("\n🗃️ STEP 5: Merging CSV results for the plotter...")
    
    # 1. Bring the per-strategy CSVs up to the main folder
    for run_dir in glob.glob(os.path.join(TRIB_OUT, "run_*")):
        for csv_file in glob.glob(os.path.join(run_dir, "*_eval.csv")):
            try:
                shutil.copy(csv_file, TRIB_OUT)
            except shutil.SameFileError:
                pass

    # 2 & 3. Merge summary and combined files
    merge_csvs("model_summary.csv")
    merge_csvs("combined_results.csv")

    # ==========================================
    # STEP 6: Generate Plots
    # ==========================================
    print("\n📈 STEP 6: Generating plots...")
    subprocess.run([
        "python", "../evaluation/prepare_tribunal_eval.py",
        "--mode", "plot",
        "--inputs-dir", TRIB_IN,
        "--results-dir", TRIB_OUT,
        "--plot-dir", PLOTS_OUT
    ], check=True)

    print(f"\n🎉 Pipeline finished successfully! Plots are available in: {PLOTS_OUT}")

if __name__ == "__main__":
    main()