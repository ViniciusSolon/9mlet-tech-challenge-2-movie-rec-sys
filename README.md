# MovieLens 20M — Recommendation System (FIAP Tech Challenge 02)

Sistema de recomendação personalizada com **PyTorch**, **DVC**, **MLflow** e **Docker**, usando o dataset [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

## Status

| Etapa do PDF | Situação |
|--------------|----------|
| 1 — Clean Code | ✅ Factory, Strategy, ruff, testes |
| 2 — Ambiente | ✅ `uv sync` + `uv.lock` + Settings |
| 3 — Docker + DVC + MLflow | ✅ `dvc repro` + compose |
| 4 — Modelo + Registry + Model Card | ✅ MLP/Embedding + baselines + Registry; vídeo STAR pendente |

Detalhes: [TODO.md](TODO.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/AUDITORIA_DESAFIO.md](docs/AUDITORIA_DESAFIO.md)

## Estrutura

```
src/
  domain/       # IDs, Rating, RecommendationList
  data/         # preprocessors (Strategy), splits, external/ (TMDB)
  features/     # feature_eng (próx.: BERTopic)
  models/       # Factory: MLP, Embedding, sklearn, MostPopular
  training/     # seeds
  evaluation/   # métricas @K + Registry
  serving/      # inferência (opcional / bônus)
configs/        # settings + YAML
scripts/        # pipeline DVC, validate_env, fetch TMDB
tests/          # espelha src/
docs/           # Model Card, scraping, auditoria
```

## Pré-requisitos

1. **Python 3.11+** e [uv](https://github.com/astral-sh/uv) *ou* Docker Desktop  
2. **Git**  
3. **MovieLens 20M** em `data/raw/` (não versionado no Git):
   - Aceitos: `rating.csv` **ou** `ratings.csv`, `movie.csv`/`movies.csv`, `link.csv`/`links.csv`
   - Fonte: [Kaggle — MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

## Setup local (recomendado para dev)

```bash
git clone <url-do-repositorio>
cd 9mlet-tech-challenge-2-movie-rec-sys
uv sync
cp .env.example .env   # preencha TMDB_API_KEY só se for re-coletar metadados
python scripts/validate_env.py
```

O parquet TMDB (`data/processed/movie_metadata.parquet`) já pode vir no repo — **não** é necessário rerodar o scrap.

## Pipeline DVC

```bash
# opcional: dados sintéticos para smoke test
python scripts/create_dummy_data.py

dvc repro
```

Stages: `preprocess → enrich_metadata → feature_eng → train → evaluate`

## Docker + MLflow

```bash
# garantir CSVs em data/raw/ (ou dummy)
docker compose run train          # executa dvc repro
docker compose up                 # sobe MLflow em http://localhost:5000
```

## Treino / avaliação manual

```bash
python scripts/train.py --model-type torch_mlp --epochs 10
python scripts/evaluate.py        # baselines + torch + Registry Staging→Production
```

## Documentação

| Doc | Conteúdo |
|-----|----------|
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Performance, limitações, vieses |
| [docs/AUDITORIA_DESAFIO.md](docs/AUDITORIA_DESAFIO.md) | Checklist vs PDF do desafio |
| [docs/IMPLEMENTACAO_MLP_PYTORCH.md](docs/IMPLEMENTACAO_MLP_PYTORCH.md) | Detalhe do treino neural |
| [docs/DOCUMENTACAO_ETAPA2.md](docs/DOCUMENTACAO_ETAPA2.md) | Ambiente e deps |
| [docs/GUIA_SCRAPING_E_PIPELINE.md](docs/GUIA_SCRAPING_E_PIPELINE.md) | Etapa TMDB |

## Licença

MIT — ver [LICENSE](LICENSE).
