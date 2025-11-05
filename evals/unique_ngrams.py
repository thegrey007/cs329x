import json, sys
from collections import Counter
from nltk.util import ngrams

# Usage: python evals/unique_ngrams.py path/to/generations.jsonl
path = sys.argv[1]

all_ngrams = Counter()
total_tokens = 0

with open(path) as f:
    for line in f:
        item = json.loads(line)
        for g in item["responses"]:
            toks = g.split()
            total_tokens += len(toks)
            all_ngrams.update(ngrams(toks, 4))

unique_4grams = len(all_ngrams)
score = unique_4grams / max(total_tokens, 1)
print(f"Unique 4-gram diversity: {score:.4f}")
