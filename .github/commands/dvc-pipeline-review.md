# DVC Pipeline Review

## Objective
Review `dvc.yaml` and stage scripts for correctness and reproducibility.

## Context
See `.cursor/context/ml-pipeline.md`.

## Instructions
1. Verify stage order: preprocess → enrich_metadata → feature_eng → train → evaluate
2. Check deps/outs are complete and paths consistent
3. Ensure no API calls without cached outs
4. Confirm `params.yaml` drives hyperparameters
5. Validate `dvc repro` idempotency and lock file hygiene

## Output
Stage diagram, issues, fixes (as diffs if requested)

## Related
- `.cursor/commands/feature-engineering-review.md`
