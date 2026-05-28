# Deployment & Reproducibility

## Objective

Document how to build, run, and reproduce the ML pipeline in clean environments.

## Prerequisites

- Python 3.11+
- Poetry or uv
- Docker & Docker Compose
- DVC CLI
- Git + DVC remote configured

## Environment Variables (`.env`)

```env
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=movielens-recommender

# DVC
DVC_REMOTE_URL=./dvc-storage
# or s3://bucket/path

# Data
DATA_DIR=./data
RAW_DATA_PATH=./data/raw

# Optional TMDB enrich
TMDB_API_KEY=your_key_here

# Optional PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/recommender

# Reproducibility
PYTHONHASHSEED=42
TORCH_SEED=42
NUMPY_SEED=42
```

Copy from `.env.example` — never commit `.env`.

## Local Setup

```bash
# 1. Dependencies
poetry install
# or: uv sync

# 2. Validate environment
poetry run python scripts/validate_env.py

# 3. Pull versioned data (after DVC init)
dvc pull

# 4. Run full pipeline
dvc repro

# 5. View experiments
mlflow ui
```

## Docker Multi-Stage

**Builder stage:** install Poetry/uv, compile dependencies, copy `src/`  
**Runtime stage:** slim Python image, non-root user, only runtime deps + artifacts

```bash
docker build -t fiap-recommender:latest .
```

## Docker Compose

Services (typical):

| Service | Role |
|---------|------|
| `app` | Training/inference container |
| `mlflow` | Tracking server (port 5000) |
| `postgres` | Optional metadata DB (port 5432) |

```bash
docker compose up --build
```

Volumes: `./data`, `./mlruns`, `./dvc-storage`

## DVC Remote

- **Local dev:** directory remote in `dvc-storage/`  
- **Team/prod:** S3-compatible or shared drive  
- Raw MovieLens stays out of Git; tracked by DVC hash  

## CI/CD (GitHub Actions)

1. Checkout + install Poetry/uv  
2. `ruff check .` and `ruff format --check .`  
3. `pytest --cov=src`  
4. Optional: `dvc repro -f` on smoke subset (scheduled or manual)  
5. Build Docker image on `main` tag  

## Pre-Deploy Checklist

- [ ] `dvc repro` succeeds on clean machine  
- [ ] `ruff check` passes  
- [ ] `pytest` passes with coverage thresholds  
- [ ] MLflow run logged with required metrics (Recall@K, NDCG@K, RMSE)  
- [ ] Model registered in MLflow Registry (Staging → Production)  
- [ ] Model Card updated  
- [ ] No secrets in image or Git  

## Rollback

1. Promote previous MLflow Registry version to Production  
2. `git checkout` previous `dvc.lock` + code tag  
3. `dvc checkout` artifacts for that commit  
4. Rebuild Docker image from tag  

## Related Documentation

- `.cursor/context/ml-pipeline.md`
- `.cursor/context/tech-stack.md`
- `.cursor/commands/pre-deploy-validation.md`
