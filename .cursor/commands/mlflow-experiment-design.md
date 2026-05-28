# MLflow Experiment Design

## Objective
Design MLflow experiments and ablation plan.

## Context
Required ablations: collaborative only; +tags; +BERTopic; +TMDB metadata.

## Instructions
1. Define experiment name and tags
2. List params to log (seeds, K, embedding dim, lr, split dates)
3. List metrics: Recall@K, Precision@K, NDCG@K, RMSE
4. Artifacts: model, encoders, config snapshot
5. Registry promotion criteria

## Output
Experiment matrix (≥4 runs) + logging checklist

## Related
- `.cursor/commands/recommendation-metrics.md`
