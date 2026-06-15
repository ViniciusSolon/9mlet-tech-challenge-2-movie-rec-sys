# Generate Boilerplate

## Objective
Generate Python boilerplate following project patterns.

## Context
Stack: PyTorch, DVC, MLflow. Patterns: Factory, Strategy. See `.cursor/rules/code-style.md`.

## Instructions
Generate for requested module:
- Type hints + Google docstrings
- `snake_case` files under correct `src/` layer
- Ruff-compliant code
- Stub tests in `tests/`

## Constraints
- Only `allowed-libs.md` dependencies
- No Node.js / TypeScript
- Functions ≤ 20 lines

## Output
Code + brief structure explanation

## Related
- `.cursor/context/architecture.md`
