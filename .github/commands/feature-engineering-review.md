# Feature Engineering Review

## Objective
Review feature_eng stage: matrices, encoders, optional BERTopic/TMDB features.

## Context
MovieLens: ratings, tags, genome; optional TMDB via `links.csv`.

## Instructions
1. Validate no temporal leakage in content features
2. Check aggregation of tags per movie (not per rating row for BERTopic)
3. Dimensionality and sparsity of user-item matrix
4. Hybrid feature concatenation plan for PyTorch
5. DVC outputs and cache strategy

## Output
Findings, leakage risks, optimization suggestions

## Related
- `.cursor/context/ml-pipeline.md`
- `.cursor/commands/dvc-pipeline-review.md`
