# Code Style Rules

## Project

**FIAP Tech Challenge 02** — Python 3.11+ recommendation system (MovieLens 20M + MLOps).

## Mandatory Tooling (before any feature code)

1. **Ruff** — lint + format (`ruff.toml` or `pyproject.toml` section)  
2. **Pytest** — `tests/` mirroring `src/`  
3. **Poetry or uv** — `pyproject.toml` with locked dependencies  
4. **pre-commit** — ruff + trailing whitespace + end-of-file  

**Development must not start until `ruff check .` passes on the scaffold.**

## Language Rules

- **Code (English):** names, docstrings, logs, errors, commits, file paths  
- **Comments (pt-BR):** inline comments explaining non-obvious ML/domain logic  

## Python Standards

- **Type hints** on all public functions, methods, and module APIs  
- **Docstrings:** Google style on public APIs  
- **Functions:** ≤ 20 lines (refactor if longer)  
- **SOLID:** single responsibility per module; depend on abstractions (Factory, Strategy)  
- **Naming:** `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants  
- **Files:** `snake_case.py` under `src/`  

## Reproducibility

- Set seeds in one place (`src/training/seeds.py`): `random`, `numpy`, `torch`  
- Read hyperparameters from `configs/*.yaml` or `params.yaml`, not hardcoded magic numbers  
- Log seed and config hash to MLflow on every train run  

## ML-Specific

- No training logic in notebooks committed as production path — use `src/` + DVC stages  
- Separate I/O (data loaders) from model logic  
- Use `pathlib.Path` for file paths  

## Prohibited

- ESLint, Prettier, TypeScript, or `package.json` as primary toolchain  
- Hardcoded secrets or API keys  
- `eval()`, unpickling untrusted files  
- Committing raw MovieLens or large artifacts to Git  
- Code that fails `ruff check`  

## Related

- `.cursor/rules/testing-rules.md`
- `.cursor/rules/python-ml.mdc`
- `.cursor/context/tech-stack.md`
