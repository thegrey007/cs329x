import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

path = sys.argv[1]
model = SentenceTransformer("all-MiniLM-L6-v2")

scores = []

def is_factual(prompt):
    factual_keywords = ["who", "what", "when", "where", "why", "how", "explain", "define"]
    return any(k in prompt.lower() for k in factual_keywords)

with open(path) as f:
    for line in f:
        item = json.loads(line)
        prompt = item["prompt"]
        gens = item["responses"]
        if len(gens) < 2:
            continue
        emb = model.encode(gens)
        sim = cosine_similarity(emb)
        diversity = 1 - np.mean(sim[np.triu_indices_from(sim, k=1)])
        if is_factual(prompt):
            # For factual: reward consistency → lower diversity
            score = 1 - diversity
        else:
            # For creative: reward variety → higher diversity
            score = diversity
        scores.append(score)

print(f"Prompt-aware diversity: {np.mean(scores):.4f}")
