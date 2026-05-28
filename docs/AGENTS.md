# Agents — FIAP Tech Challenge 02

Multi-role guidance for AI-assisted work on the MovieLens 20M recommendation system.

## Data Engineer

**Focus:** DVC, raw/processed data, MovieLens ETL, TMDB enrich cache.

**Responsibilities:**
- `dvc.yaml` stages: preprocess, enrich_metadata
- Data quality checks and schema validation
- Temporal splits without leakage
- DVC remote and `.gitignore` for `data/raw/`

**Read:** `.cursor/context/ml-pipeline.md`, `.cursor/commands/dvc-pipeline-review.md`

**Do not:** Train PyTorch models or promote MLflow Registry alone.

---

## ML Engineer

**Focus:** PyTorch models, sklearn baselines, features, metrics.

**Responsibilities:**
- `src/models/` (Factory), `src/features/`, `src/training/`
- Recall@K, Precision@K, NDCG@K, RMSE
- BERTopic/sentence-transformers in feature_eng only
- Ablation experiments per MLflow plan

**Read:** `.cursor/rules/business-rules.md`, `.cursor/commands/recommendation-metrics.md`

**Do not:** Change DVC remote credentials or skip evaluate before registry.

---

## MLOps Engineer

**Focus:** Docker, MLflow, CI, reproducibility.

**Responsibilities:**
- Multi-stage Dockerfile, docker-compose (app + mlflow)
- MLflow tracking and Model Registry workflow
- `poetry install` / `uv sync` + `dvc repro` documentation
- GitHub Actions: ruff, pytest

**Read:** `.cursor/context/deployment.md`, `.cursor/rules/mlops.mdc`

**Do not:** Commit secrets or raw MovieLens to Git.

---

## Reviewer

**Focus:** PR quality, challenge compliance, anti-hallucination on metrics.

**Responsibilities:**
- Verify ruff + pytest pass
- Confirm business rules (top-K, seen filter, cold start documented)
- Challenge overengineering (`.cursor/commands/challenge-solution.md`)
- Ensure MLflow metrics exist for claimed performance

**Read:** `.cursor/commands/review-pr.md`, `TODO.md`

**Do not:** Approve PRs that add forbidden libs or Node.js tooling.

---

## Shared Rules

All agents follow:
- `.cursor/rules/python-ml.mdc`
- `.cursor/rules/mlops.mdc`
- `.cursor/rules/testing.mdc`
- `.cursor/libs/allowed-libs.md`

**Golden rule:** AI proposes; humans approve. Metrics must come from real MLflow runs.
