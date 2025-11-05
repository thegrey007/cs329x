import json, sys
from sacrebleu import sentence_bleu

path = sys.argv[1]
scores = []

with open(path) as f:
    for line in f:
        data = json.loads(line)
        gens = data["responses"]
        if len(gens) < 2: 
            continue
        for i, hyp in enumerate(gens):
            refs = gens[:i] + gens[i+1:]
            scores.append(sentence_bleu(hyp, refs).score)

avg_self_bleu = sum(scores)/len(scores)
print(f"Self-BLEU: {avg_self_bleu:.2f}")
