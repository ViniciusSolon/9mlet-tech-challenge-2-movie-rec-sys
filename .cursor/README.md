# .cursor — FIAP Tech Challenge 02

Configuration and context for AI-assisted development of the **MovieLens 20M recommendation system** (Python + PyTorch + DVC + MLflow + Docker).

## Structure

### `context/`
- `project-goals.md` — Business objectives and constraints
- `architecture.md` — ML system architecture and patterns
- `ml-pipeline.md` — DVC stages, artifacts, MLflow ablations
- `tech-stack.md` — Official Python/MLOps stack
- `deployment.md` — Docker, DVC, MLflow, reproducibility

### `rules/`
- `*.md` — Human-readable rules (code, tests, security, AI, business)
- `python-ml.mdc` — Cursor rule: Python ML (always apply + `**/*.py`)
- `mlops.mdc` — Cursor rule: DVC/MLflow/Docker
- `testing.mdc` — Cursor rule: pytest standards

### `libs/`
- `allowed-libs.md` / `forbidden-libs.md` — Python dependency policy
- `ai-models.md` — LLM model selection by phase

### `commands/`
Reusable prompts for kickoff, architecture, DVC, MLflow, metrics, PR review, etc.

See also: **`AGENTS.md`** (root) for Data Engineer / ML Engineer / MLOps / Reviewer roles.

## Workflow

### Developers
1. `poetry install` or `uv sync`
2. Configure `ruff` + `pytest` + `pre-commit` (see `rules/code-style.md`)
3. Read `context/` before coding
4. Run `dvc repro` for pipeline changes

### AI
1. Load `.mdc` rules (always apply)
2. Read `rules/` and `context/`
3. Check `libs/` before adding dependencies
4. Use `commands/` for structured tasks
5. Output must pass `ruff check` and `pytest`

## Principles

- Explicit context over long prompts
- Reproducibility over speed
- Batch ML pipeline over generic web CRUD
- AI proposes; humans approve

## Maintenance

Keep `context/ml-pipeline.md` in sync with `dvc.yaml` when stages change.
