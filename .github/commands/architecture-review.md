# Architecture Review

## Objective
Validate ML system architecture before or during implementation.

## Context
See `.cursor/context/architecture.md` and `.cursor/context/ml-pipeline.md`.

## Instructions
1. Evaluate alignment with MovieLens 20M + MLOps requirements
2. Check coupling between data, features, models, training
3. Verify Factory/Strategy placement
4. Assess reproducibility (DVC, seeds, MLflow)
5. Suggest improvements with trade-offs

## Constraints
- No implementation
- Focus on batch pipeline, not generic REST CRUD

## Output
Strengths, weaknesses, recommendations, trade-offs

## Related
- `.cursor/commands/dvc-pipeline-review.md`
