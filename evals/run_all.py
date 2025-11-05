import subprocess, sys, os

path = sys.argv[1]  # path to generations.jsonl

print("\n--- Running Evaluation Metrics ---\n")
metrics = [
    ("Unique 4-grams", ["python", "evals/unique_ngrams.py", path]),
    ("Self-BLEU", ["python", "evals/self_bleu.py", path]),
    ("Embedding Diversity", ["python", "evals/embedding_diversity.py", path]),
    ("Prompt-Aware Diversity", ["python", "evals/prompt_aware_diversity.py", path]),
]

for name, cmd in metrics:
    print(f"→ {name}")
    subprocess.run(cmd)
    print()
