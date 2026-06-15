# Refactor Controlled

## Objective
Refactor Python ML code without changing behavior.

## Context
PyTorch / sklearn / pipeline code in `src/`. See `.cursor/rules/code-style.md`.

## Instructions
1. Improve readability and SOLID
2. Reduce complexity (extract functions ≤ 20 lines)
3. Keep public APIs and metrics identical
4. Keep pytest green

## Output
Refactored code + change list + test status

## Related
- `.cursor/rules/testing-rules.md`
