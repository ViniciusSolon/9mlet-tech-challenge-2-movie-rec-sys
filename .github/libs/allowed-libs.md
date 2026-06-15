# Allowed Libraries

## Core ML

| Library | Use |
|---------|-----|
| **torch** | Neural recommender (embeddings, MLP) |
| **scikit-learn** | Baselines, preprocessing helpers, metrics |
| **mlflow** | Tracking, registry, artifacts |
| **dvc** | Data versioning, pipeline stages |
| **pandas** | ETL, feature tables |
| **numpy** | Numerical operations |

## Config & API

| Library | Use |
|---------|-----|
| **pydantic** | Settings, FastAPI schemas |
| **pydantic-settings** | `.env` loading |
| **fastapi** | Optional inference API |
| **uvicorn** | ASGI server for FastAPI |

## Database

| Library | Use |
|---------|-----|
| **sqlalchemy** | ORM / connection management |
| **psycopg2** or **psycopg** | PostgreSQL driver |

## NLP (optional)

| Library | Use |
|---------|-----|
| **sentence-transformers** | Text embeddings for tags/overview |
| **bertopic** | Topic features per movie (feature_eng only) |

## DevOps & Quality

| Library | Use |
|---------|-----|
| **poetry** or **uv** | Dependency management |
| **ruff** | Lint + format |
| **pytest**, **pytest-cov** | Tests and coverage |
| **pre-commit** | Git hooks |
| **httpx** or **requests** | TMDB API (cached, not per-epoch) |

## Standard Library

Prefer `pathlib`, `dataclasses`, `typing`, `logging` before adding deps.

## Approval

New dependencies require Tech Lead approval and update to this file.

## Related

- `.cursor/libs/forbidden-libs.md`
- `.cursor/context/tech-stack.md`
