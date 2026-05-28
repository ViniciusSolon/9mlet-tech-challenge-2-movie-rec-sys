# Pre-Deploy Validation

## Objective
Checklist before promoting model or releasing Docker image.

## Context
See `.cursor/context/deployment.md`.

## Instructions
Validate:
1. Business rules (metrics thresholds, top-K behavior)
2. Security (no secrets, safe pickle sources)
3. `dvc repro` success + `dvc.lock` committed
4. MLflow run complete (Recall@K, NDCG@K, RMSE)
5. Model Registry promotion justified
6. Model Card updated
7. `ruff` + `pytest` pass

## Constraints
- Do not modify code unless asked
- Flag blockers vs warnings

## Output
Checklist with pass/fail per item

## Related
- `.cursor/commands/model-card-generator.md`
