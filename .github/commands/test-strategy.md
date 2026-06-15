# Test Strategy

## Objective
Define pytest strategy for a module or DVC stage.

## Context
See `.cursor/rules/testing-rules.md` and `.cursor/rules/testing.mdc`.

## Instructions
1. Critical scenarios (metrics, splits, Factory, Strategy)
2. Unit tests with synthetic MovieLens fixtures
3. Integration tests for stage scripts
4. Pipeline smoke on tiny data
5. Edge cases: empty users, cold start, leakage

## Constraints
- pytest only (no Jest/Vitest)
- Plan before code unless asked to implement

## Output
Test plan with file paths under `tests/`

## Related
- `.cursor/commands/dvc-pipeline-review.md`
