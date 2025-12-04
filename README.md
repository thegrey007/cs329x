# cs329x

Data: 

A) Ultra-feedback sampled preference pairs (Chosen =  highest overall_score, rejected = lowest overall_score)
B) Ultra-feedback sampled preference pairs (Chosen = highest avg. embedding distance above threshold, lowest avg. embedding distance below threshold)
C) Ultra-feedback sampled preference pairs (Chosen = highest LLM-as-judge diversity above threshold, lowest LLM-as-judge diversity above threshold)          —— IF TIME
D) Allen-AI Tulu 2.5 sampled preference pairs (Pre-constructed)
	- Allen-AI dataset roughly similar to Ultra-feedback, but can also push this as a Out-of-domain evaluation, good as well

Models:

1) Standard DPO
2) DPO with Diversity Aware Loss Term
3) DPO with Two-Adapter approach 

Training (Start with 7/8B instruct tuned models, can try another size if time):

- Will use ultra feedback (Train-test-split)
- Baselines:1) trained on A) : standard DPO baseline1) trained on B) : evaluate effectiveness of custom pair creation
- Approaches:2), 3) each trained on A), B) : motivates need for custom pairs, effectiveness of approach on custom pairs
- Evaluations: For each model:Self-BLEU (Zhu et al 2018), Distinct-N (Li et al 2016), Semantic Diversity (Li et Al 2024) on Allen-AI Tulu 2.5Novelty Bench on their prompts, or Allen-AI prompts
