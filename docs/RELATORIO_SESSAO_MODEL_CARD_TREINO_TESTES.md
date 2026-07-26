# Relatório do trabalho local (pré-commit) — Model Card, treino GPU, testes e demo texto

**Branch:** `feat/bloco5-mlp-pytorch-training`  
**Base HEAD atual:** `9999e8a` (*feat: merge training pipeline and MLflow registry evaluation*)  
**Data:** 2026-07-24  
**Responsável principal deste pacote:** Vítor  
**Status Git:** alterações **ainda não commitadas / não enviadas** ao remoto

> Este documento resume **tudo que foi feito e testado** nesta sessão de trabalho, para apoiar o commit/PR e o vídeo STAR.  
> **Não versionar:** `mlflow.db`, pasta `artifacts/` (pesada), `.env`, CSVs brutos, cache JSON TMDB.

---

## 1. Objetivo desta rodada

1. Fechar **Model Card** e **auditoria vs PDF** do Tech Challenge.  
2. Atualizar `TODO.md` / `README` ao estado real do código.  
3. Rodar pipeline **local (caminho B)** com **GPU (RTX 3060)**.  
4. Regenerar `movie_metadata.parquet` a partir do cache TMDB.  
5. Avaliar modelo vs baselines, promover no **MLflow Registry**.  
6. Criar testes “reais” (golden JSON) e demo de recomendação por histórico (`llm/`).

---

## 2. Documentação entregue

| Arquivo | Conteúdo |
|---------|----------|
| `docs/MODEL_CARD.md` | Model Card com métricas reais, limitações, vieses, leitura leiga |
| `docs/AUDITORIA_DESAFIO.md` | Checklist de aderência ao enunciado (PDF) |
| `docs/README.md` | Índice atualizado apontando Model Card e auditoria |
| `llm/examples/prompt_avaliacao_imparcial.md` | Prompt para LLM externo julgar coerência das indicações da Ana |

---

## 3. Código e refatorações principais

### Dados e pipeline
- `src/data/external/movielens_io.py` — `load_ratings()` aceita `rating.csv` **ou** `ratings.csv`.
- `scripts/preprocess.py` — usa `load_ratings` (compatível Kaggle).
- `dvc.yaml` — deps alinhadas aos nomes Kaggle (`rating.csv`, `movie.csv`, `link.csv`).
- `scripts/enrich_metadata.py` — join por `movieId` (snake_case do parquet ↔ MovieLens).
- `data/processed/movie_metadata.parquet` — **regenerado** do cache (27.278 linhas, ~7,8 MB; não é mais o dummy de 3 linhas).

### Split temporal
- Novo `src/data/splits.py` — split temporal preferencial; timestamps string/unix.
- `scripts/train.py` e `scripts/evaluate.py` passam a usar esse split.
- Testes: `tests/unit/test_splits.py`.

### Modelos e avaliação
- Novo `src/models/most_popular.py` + Factory.
- `scripts/evaluate.py` — baselines (MostPopular, KNN, RF), tabela `comparison`, métricas @K, Registry Staging→Production.
- Amostragem em baselines/ranking para caber em 20M ratings.
- Nomes de métrica MLflow sem `@` (`precision_at_10`).
- Log PyTorch em `pickle` (evita erro `pt2` / `input_example`).
- `torch_mlp` / `torch_embedding` — `predict()` move tensores para o device do modelo (CUDA).

### Treino
- PyTorch reinstalado com **CUDA 12.4** (`torch 2.6.0+cu124`).
- Treino GPU: batch 8192, early stopping, melhor `val_mse ≈ 0,749` (época 4).
- Checkpoint: `models/model.pth` (~21 MB).

---

## 4. O que rodamos e testamos (evidências)

### 4.1 Ambiente
- `.env` a partir de `.env.example`.
- Deps completas no `.venv` (torch CUDA, mlflow, dvc, bertopic, etc.).
- MLflow local: `http://127.0.0.1:5000` (Docker Desktop estava off).

### 4.2 Dados
| Item | Resultado |
|------|-----------|
| `rating.csv` | ~20.000.263 linhas |
| preprocess | 19.761.138 ratings salvos |
| enrich | 27.278 filmes |
| feature_eng | `features_ratings.parquet` OK |
| parquet TMDB | ok=26.717 · not_found=309 · missing_tmdb_id=252 |

### 4.3 Treino GPU (`torch_mlp`)
| Item | Valor |
|------|--------|
| Device | **cuda** (RTX 3060) |
| Users / items | 138.480 / 27.278 |
| Early stop | época 7 |
| Melhor val_mse | **0,7489** (época 4) |
| Artefato | `models/model.pth` |

### 4.4 Evaluate + Registry
| Modelo | RMSE | MAE | P@10 | R@10 | NDCG@10 | HR@10 |
|--------|------|-----|------|------|---------|-------|
| **torch_mlp (champion)** | **0,780** | **0,600** | **0,166** | **0,042** | **0,179** | **0,626** |
| most_popular | 0,887 | 0,691 | — | — | — | — |
| sklearn_random_forest | 0,906 | 0,720 | — | — | — | — |
| sklearn_knn | 0,948 | 0,760 | — | — | — | — |

- Split temporal: **respeitado** (`True`).
- Champion: **torch_mlp** → Registry **`movie-rec-sys` v1 / Production**.
- Artefato: `metrics.json` (seção `comparison` + `champion`).

### 4.5 Testes automatizados
| Comando / suíte | Resultado |
|-----------------|-----------|
| `ruff check src tests` | OK (após ajustes) |
| `pytest tests/unit` | OK (torch skip se ausente; no ambiente GPU/local passou) |
| `pytest tests/unit/test_splits.py` | OK |
| `pytest tests/unit/test_most_popular.py` | OK |
| `pytest tests/integration/test_real_inference.py` | **2 passed** |
| `python scripts/run_inference_cases.py` | **PASS** (7/7 casos) |

Golden de inferência: `tests/fixtures/inference_cases.json`  
(scores absolutos ±0,05 + ordens de preferência).

### 4.6 Demo texto (`llm/`)
| Item | Detalhe |
|------|---------|
| Script | `llm/recommend_from_history.py` |
| Entrada | `llm/examples/historico_exemplo.json` (Ana + 8 filmes/notas) |
| Saída | top 10 com título/score/sinopse + JSON `*_recomendacoes.json` |
| Prompt auditoria | `llm/examples/prompt_avaliacao_imparcial.md` |

**Histórico efetivo da Ana (após match):** Toy Story, Jumanji, Heat **(1972 — possível mismatch)**, GoldenEye, Usual Suspects, Pulp Fiction, Shawshank, Forrest Gump.

**Top 10 obtido (amostra):** Band of Brothers, One Shot, Silence of the Lambs, Pearl Jam Live, People on Sunday, Battlestar Galactica, We Stand Alone Together, Star Wars IV, Dark Knight, Dawn Patrol.

---

## 5. Arquivos novos relevantes (para o commit)

Incluir (sugerido):

```
docs/MODEL_CARD.md
docs/AUDITORIA_DESAFIO.md
docs/RELATORIO_SESSAO_MODEL_CARD_TREINO_TESTES.md   # este arquivo
llm/recommend_from_history.py
llm/examples/historico_exemplo.json
llm/examples/prompt_avaliacao_imparcial.md
scripts/run_inference_cases.py
src/data/splits.py
src/models/most_popular.py
tests/fixtures/inference_cases.json
tests/integration/...
tests/unit/test_most_popular.py
tests/unit/test_splits.py
+ alterações em train/evaluate/preprocess/enrich/README/TODO/dvc.yaml/...
+ data/processed/movie_metadata.parquet (parquet real ~7,8 MB)
+ metrics.json (resultados do evaluate)
```

**Evitar no commit:**

```
mlflow.db
artifacts/
models/model.pth          # opcional: grande; se não versionar, documentar como gerar
.env
data/raw/*.csv
data/raw/external_metadata/*.json
```

> `models/model.pth` (~21 MB) e `training_history.json` podem ficar só locais ou no DVC/MLflow; o relatório e o Model Card já registram como reproduzir.

---

## 6. Conclusões técnicas

1. O **MLP PyTorch na GPU** treinou no MovieLens 20M e **venceu** baselines em RMSE/MAE.  
2. **Hit Rate@10 ≈ 63%** no ranking amostrado — sinal útil de personalização.  
3. Model Card + Registry Production atendem o fechamento da Etapa 4 do PDF (falta vídeo STAR).  
4. A demo `llm/` torna o caso de uso tangível para o STAR, com ressalva do match **Heat 1972 vs 1995**.  
5. Testes golden protegem o checkpoint atual contra regressão silenciosa.

---

## 7. Próximos passos sugeridos antes/depois do push

1. Revisar `.gitignore` (garantir `artifacts/`, `mlflow.db`, `*.pth` se aplicável).  
2. Commit semântico, ex.:  
   `feat: model card, treino GPU, evaluate/registry, testes golden e demo llm`  
3. (Opcional) Corrigir resolução de título com desempate por ano/gênero.  
4. Gravar **vídeo STAR**.  
5. Preencher tabela do Model Card já está feita — manter `metrics.json` alinhado no PR.

---

## 8. Comandos úteis de reprodução

```bash
# ambiente
python -m pip install -e ".[dev]"
# torch CUDA (se necessário)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# MLflow
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./artifacts --serve-artifacts

# pipeline
python scripts/preprocess.py --strategy explicit
python scripts/enrich_metadata.py
python scripts/feature_engineering.py
python scripts/train.py --model-type torch_mlp --epochs 10 --batch-size 8192
python scripts/evaluate.py --model-type torch_mlp

# testes
pytest tests/unit tests/integration -q
python scripts/run_inference_cases.py
python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json
```
