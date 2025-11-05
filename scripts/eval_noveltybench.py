import subprocess
from pathlib import Path

EVAL_DIR = Path("noveltybench/results/curated/custom-model")

commands = [
    [
        "python", "noveltybench/src/inference.py",
        "--mode", "transformers",
        "--model", "path/to/your/custom/model-or-hf-id",
        "--data", "curated",
        "--eval-dir", str(EVAL_DIR),
        "--num-generations", "5",
        "--temperature", "0.8",
        "--top_p", "0.9",
    ],
    ["python", "noveltybench/src/partition.py", "--eval-dir", str(EVAL_DIR), "--alg", "classifier"],
    ["python", "noveltybench/src/score.py", "--eval-dir", str(EVAL_DIR)],
    ["python", "noveltybench/src/summarize.py", "--eval-dir", str(EVAL_DIR)],
]

for cmd in commands:
    subprocess.run(cmd, check=True)
