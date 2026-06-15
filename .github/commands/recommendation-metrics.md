# Recommendation Metrics

## Objective
Define or review metric implementations for ranking and rating prediction.

## Context
See `.cursor/rules/business-rules.md`.

## Instructions
1. Specify Recall@K, Precision@K, NDCG@K, RMSE definitions for this project
2. Hold-out protocol (temporal split)
3. How to handle already-seen items in evaluation
4. Unit test cases with hand-calculated examples
5. MLflow logging field names

## Output
Metric spec + pytest test case outline

## Related
- `src/evaluation/`
