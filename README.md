# MovieLens 20M — Recommendation System (FIAP Tech Challenge 02)

Sistema de recomendação personalizada com **PyTorch**, **DVC**, **MLflow** e **Docker**, usando o dataset [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

## Status do scaffold

- **Blocos 0–1:** estrutura `src/`, Factory, Strategy, Ruff, pytest, Docker mínimo
- **Etapa de scraping TMDB:** concluída — [docs/GUIA_SCRAPING_E_PIPELINE.md](docs/GUIA_SCRAPING_E_PIPELINE.md) (leigos) · [docs/PRE_ETAPA_METADADOS.md](docs/PRE_ETAPA_METADADOS.md) (comandos)

## Estrutura

```
src/
  domain/       # UserId, MovieId, Rating, RecommendationList
  data/         # preprocessors (Strategy), external/ (TMDB)
  features/     # feature_eng (Bloco 4)
  models/       # Factory + stubs PyTorch/sklearn
  training/     # seeds, loops (Bloco 5)
  evaluation/   # métricas @K (Bloco 5)
  serving/      # inferência (opcional)
configs/        # YAML
data/           # raw / processed (não versionar CSV brutos no Git)
scripts/        # hello_train, fetch TMDB, relatórios de metadados
tests/          # espelha src/
```

## Desenvolvimento local

```bash
# Python 3.11+
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"

ruff check .
ruff format .
pytest

pre-commit install
pre-commit run --all-files
```

## Etapa de scraping TMDB (já concluída no repo)

Metadados consolidados em `data/processed/movie_metadata.parquet` (versionado no Git).

Para **refazer** a coleta:

```bash
python scripts/fetch_external_metadata.py --limit 3   # teste
python scripts/fetch_external_metadata.py --resume  # ~27k filmes
python scripts/metadata_coverage_report.py          # relatório P.8
```

## Docker

```bash
# Build e smoke test
docker build -t movie-rec-sys .
docker run --rm movie-rec-sys

# App + MLflow (UI em http://localhost:5000)
docker compose up --build
```

Copie `.env.example` para `.env` só se for **refazer** a coleta TMDB.

## Dados para o time

| Artefato | Como obter |
|----------|------------|
| Metadados TMDB | `git pull` → `data/processed/movie_metadata.parquet` (~7,5 MB) |
| MovieLens 20M | [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) → `data/raw/` (`movie.csv`, `link.csv`, `rating.csv`, …) |
| Cache JSON do scrap | Local apenas — **não** vai no Git |

**Setup rápido após clone:**

```bash
pip install -e ".[dev]"
# Baixar MovieLens em data/raw/ (CSV não estão no repositório)
```

## Plano de execução

Ver [TODO.md](TODO.md), [docs/](docs/) e contexto em [.cursor/context/](.cursor/context/).

## Licença

MIT — ver [LICENSE](LICENSE).
