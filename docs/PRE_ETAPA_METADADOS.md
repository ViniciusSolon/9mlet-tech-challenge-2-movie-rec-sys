# Etapa de scraping — Coleta de metadados TMDB (concluída)

**Guia para leigos (scraping + pipeline futuro):** [GUIA_SCRAPING_E_PIPELINE.md](GUIA_SCRAPING_E_PIPELINE.md).

Detalhes técnicos da coleta: [FLUXO_COLETA_METADADOS.md](FLUXO_COLETA_METADADOS.md).

## Status

| Item | Status |
|------|--------|
| P.1–P.7 Coleta e parquet | Concluído |
| P.8 Relatório de cobertura | [METADATA_COVERAGE_REPORT.md](METADATA_COVERAGE_REPORT.md) |
| P.9 Validação manual (10 filmes) | [METADATA_VALIDATION_SAMPLE.md](METADATA_VALIDATION_SAMPLE.md) |
| P.10 Documentação | Este arquivo |

**Última coleta:** `movie_metadata.parquet` com **26.717** filmes `ok` de **27.278** (~97,9%).

---

## Pré-requisitos

1. `.env` com `TMDB_API_KEY`
2. MovieLens em `data/raw/`: `movie.csv` e `link.csv` (ou `movies.csv` / `links.csv`)
3. `pip install -e ".[dev]"`

---

## Passo a passo executado

### 1. Configurar ambiente

```bash
copy .env.example .env
# Editar TMDB_API_KEY no .env
pip install -e ".[dev]"
```

### 2. Coletar metadados (P.4–P.7)

```bash
# Teste
python scripts/fetch_external_metadata.py --limit 3

# Produção (~27k filmes, barra de progresso no terminal)
python scripts/fetch_external_metadata.py --resume
```

**Saídas:**

| Artefato | Caminho |
|----------|---------|
| Cache JSON | `data/raw/external_metadata/{movieId}.json` |
| Parquet consolidado | `data/processed/movie_metadata.parquet` |
| Log | `data/logs/fetch_metadata.log` |

### 3. Relatório de cobertura (P.8)

```bash
python scripts/metadata_coverage_report.py
```

Gera `docs/METADATA_COVERAGE_REPORT.md`.

### 4. Amostra para validação manual (P.9)

```bash
python scripts/metadata_validation_sample.py
```

Abrir `docs/METADATA_VALIDATION_SAMPLE.md`, conferir 10 filmes no TMDB e marcar `[x]`.

---

## Resultado da coleta (referência)

| Métrica | Valor |
|---------|-------|
| Total | 27.278 |
| ok | 26.717 |
| not_found | 309 |
| missing_tmdb_id | 252 |
| api_error | 0 |

Entre os `ok`: ~100% com overview/genres; ~86% com keywords (ver relatório P.8).

---

## Flags do fetch

| Flag | Descrição |
|------|-----------|
| `--limit N` | Só N primeiros filmes |
| `--resume` | Reutiliza JSON com `fetch_status=ok` |
| `--raw-dir PATH` | Pasta raw alternativa |

## Variáveis `.env`

| Variável | Padrão |
|----------|--------|
| `TMDB_API_KEY` | obrigatório |
| `TMDB_LANGUAGE` | `en-US` |
| `TMDB_MIN_INTERVAL` | `0.26` |
| `TMDB_MAX_RETRIES` | `4` |

## Colunas do parquet

`movie_id`, `tmdb_id`, `imdb_id`, `title`, `overview`, `genres`, `keywords`, `release_year`, `original_language`, `vote_average`, `popularity`, `fetch_status`

Status: `ok` | `missing_tmdb_id` | `not_found` | `api_error`

---

## Próximo passo do projeto

**Bloco 3** (EDA MovieLens) e **Bloco 4** (`feature_eng` faz merge do parquet com ratings).

O pipeline **não** chama TMDB de novo — só lê `movie_metadata.parquet`.

---

## Scripts relacionados

| Script | Função |
|--------|--------|
| `scripts/fetch_external_metadata.py` | Coleta TMDB |
| `scripts/metadata_coverage_report.py` | Relatório P.8 |
| `scripts/metadata_validation_sample.py` | Checklist P.9 |
