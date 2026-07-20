# Relatório de cobertura — metadados TMDB (P.8)

> Gerado a partir de `data/processed/movie_metadata.parquet`.

## Resumo da coleta (`fetch_status`)

| Status | Quantidade | % do total |
|--------|------------|------------|
| **ok** | 26717 | 97.94% |
| not_found | 309 | 1.13% |
| missing_tmdb_id | 252 | 0.92% |
| api_error | 0 | 0.0% |
| **Total** | 27278 | 100% |

## Campos preenchidos (somente filmes com `fetch_status=ok`)

| Campo | % com dado |
|-------|------------|
| title | 100.0% |
| overview (sinopse) | 99.95% |
| genres | 99.74% |
| keywords | 85.74% |
| release_year | 99.99% |

## Interpretação

- **overview** alimenta BERTopic / embeddings de texto no `feature_eng`.
- **not_found** / **missing_tmdb_id**: filme segue só com MovieLens (`movie.csv`, tags).
- Cobertura **≥ 95%** em `ok` é adequada para o Tech Challenge.
