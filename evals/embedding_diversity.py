import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

path = sys.argv[1]
model = SentenceTransformer("all-MiniLM-L6-v2")

all_scores = []

with open(path) as f:
    for line in f:
        data = json.loads(line)
        gens = data["responses"]
        if len(gens) < 2:
            continue
        emb = model.encode(gens)
        sim = cosine_similarity(emb)
        avg_sim = np.mean(sim[np.triu_indices_from(sim, k=1)])
        all_scores.append(1 - avg_sim)  # diversity = 1 - similarity

final_score = np.mean(all_scores)
print(f"Embedding diversity: {final_score:.4f}")
