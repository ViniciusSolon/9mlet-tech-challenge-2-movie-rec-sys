# Technology Stack

## Objective

Document the official technology stack for the FIAP Tech Challenge 02 recommendation system (MovieLens 20M + MLOps).

## Language & Runtime

| Component | Choice | Notes |
|-----------|--------|-------|
| **Language** | Python 3.11+ | Type hints mandatory on public APIs |
| **Package manager** | Poetry or uv | Lock file committed (`poetry.lock` / `uv.lock`) |
| **Config** | Pydantic Settings | `.env` + `configs/` YAML |

## Machine Learning

| Component | Choice | Purpose |
|-----------|--------|---------|
| **Deep learning** | PyTorch | Neural recommender (MLP / embedding-based) |
| **Baselines** | Scikit-Learn | Classical baselines + metrics helpers |
| **Experiment tracking** | MLflow | Params, metrics, artifacts, Model Registry |
| **Data versioning** | DVC | Pipeline stages, remote storage, reproducibility |
| **Numerics** | NumPy, Pandas | ETL, feature matrices |
| **NLP (optional)** | sentence-transformers, BERTopic | Content features from tags/metadata |

## Data & Storage

| Component | Choice | Purpose |
|-----------|--------|---------|
| **Dataset** | MovieLens 20M | Ratings, movies, tags, links, genome-scores |
| **Artifacts** | Parquet / pickle (versioned by DVC) | Processed features, encoders |
| **Database** | PostgreSQL | Optional: metadata cache, serving catalog, experiment metadata |
| **DB access** | SQLAlchemy, psycopg2 (or psycopg) | Parameterized queries only |

## API & Serving (optional)

| Component | Choice | Purpose |
|-----------|--------|---------|
| **Serving** | FastAPI | Inference API only when required for demo/deploy |
| **Validation** | Pydantic | Request/response schemas |

## Code Quality & Testing

| Tool | Purpose |
|------|---------|
| **Ruff** | Lint + format (replaces black/isort/flake8) |
| **Pytest** | Unit, integration, pipeline smoke tests |
| **pytest-cov** | Coverage gates in CI |
| **pre-commit** | ruff + pytest hooks |

## Infrastructure

| Tool | Purpose |
|------|---------|
| **Docker** | Multi-stage image (builder + runtime) |
| **Docker Compose** | App + MLflow server + PostgreSQL |
| **GitHub Actions** | CI: ruff, pytest, optional `dvc repro` smoke |

## Dependencies Policy

- Allowed: `.cursor/libs/allowed-libs.md`
- Forbidden: `.cursor/libs/forbidden-libs.md`

## Related Documentation

- `.cursor/context/architecture.md`
- `.cursor/context/ml-pipeline.md`
- `.cursor/context/deployment.md`
- `.cursor/rules/code-style.md`
