# Model Card Generator

## Objective
Draft a Model Card for the registered recommender.

## Context
Use latest MLflow Production run metrics and `.cursor/rules/business-rules.md`.

## Instructions
Include sections:
1. Model description (PyTorch architecture summary)
2. Training data (MovieLens 20M, split policy)
3. Metrics (Recall@K, NDCG@K, RMSE vs baselines)
4. Limitations (cold start, bias, popularity bias)
5. Ethical considerations (non-personal data, pseudonymous IDs)
6. Reproducibility (`dvc repro`, seeds, Docker image tag)

## Output
`docs/MODEL_CARD.md` ready for team review

## Related
- `.cursor/commands/pre-deploy-validation.md`
