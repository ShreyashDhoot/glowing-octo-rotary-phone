"""Launch the vLLM server that hosts the judge model.
Run this in its own terminal (tmux is convenient over SSH) and leave it up
while evaluations run:
    python serve_judge.py
The default judge (Qwen2.5-32B-Instruct, 4-bit) needs roughly 24GB of VRAM.
For a smaller GPU, change `judge_model` in tribunal/config.py to a smaller
instruct model and lower --gpu-memory-utilization below if needed.
"""
import argparse
import subprocess
import sys
from tribunal.config import CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000,
                     help="Port to serve the judge model on (default 8000).")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
args = parser.parse_args()

JUDGE_MODEL = CONFIG["judge_model"]
command = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", JUDGE_MODEL,
    "--quantization", "bitsandbytes",
    "--load-format", "bitsandbytes",
    "--dtype", "half",
    "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    "--port", str(args.port),
    "--api-key", "EMPTY",
]
print(f"Starting vLLM judge server ({JUDGE_MODEL}) on port {args.port}.")
print("Leave this running while evaluations run.")
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
try:
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
except KeyboardInterrupt:
    print("\nStopping server.")
    process.terminate()