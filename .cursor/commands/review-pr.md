# Review PR

## Objective
Senior ML engineer PR review.

## Context
Python + PyTorch + DVC + MLflow. See `.cursor/rules/git-rules.md`.

## Instructions
Evaluate:
1. Clarity and SOLID in `src/`
2. Business rules (top-K, leakage, cold start)
3. pytest coverage and ruff compliance
4. DVC/MLflow impact if pipeline changed
5. Reproducibility (seeds, lock files)

## Output
Praise, issues, actionable suggestions (educational tone)

## Related
- `.cursor/rules/business-rules.md`
