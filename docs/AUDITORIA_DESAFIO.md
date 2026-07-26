# Auditoria — Aderência ao Tech Challenge Fase 02

Cruzamento do enunciado (`docs/pdf_extract.txt`) com o estado do repositório em **2026-07-25**.

## Entregáveis do PDF

| Exigência | Status | Evidência |
|-----------|--------|-----------|
| Repo GitHub + clean code | ✅ | `src/`, Factory, Strategy, ruff, testes |
| `pyproject.toml` + lock | ✅ | `pyproject.toml`, `uv.lock` (uv ≡ Poetry/uv do PDF) |
| `.dockerignore` / `.gitignore` / `.env.example` | ✅ | raiz |
| Commits semânticos | ✅ | `feat:` / `fix:` / `docs:` / `test:` |
| PyTorch MLP/embedding | ✅ | `torch_mlp.py`, `torch_embedding.py`, early stopping |
| Scikit-Learn baselines | ✅ | KNN, RandomForest + MostPopular |
| ≥ 4 métricas | ✅ | RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate |
| MLflow tracking | ✅ | `train.py` / `evaluate.py` |
| Model Registry Staging→Production | ✅ | champion = `torch_mlp` em Production |
| **Model Card** | ✅ | `docs/MODEL_CARD.md` (métricas do run completo) |
| Dockerfile multi-stage | ✅ | builder + runtime; copia `dvc.yaml`/`params.yaml` |
| docker-compose train + MLflow | ✅ | `docker-compose.yml` |
| DVC ≥ 3 stages + aliases Kaggle/GroupLens | ✅ | 5 stages + `prepare_raw_aliases` |
| Seeds + `.env` + validate_env | ✅ | 27/27 checks |
| README com instruções | ✅ | `uv sync --extra dev` documentado |
| Vídeo STAR ≤ 5 min | ❌ | pendente (fora do código) |
| Deploy nuvem (opcional) | ❌ | bônus |

## Critérios de nota (pesos)

| Critério | Peso | Avaliação interna |
|----------|------|-------------------|
| Clean code | 15% | Forte (helpers em `src/training`, `src/evaluation`) |
| Reprodutibilidade | 15% | Forte (`uv` + lock + validate_env) |
| Docker | 15% | Bom |
| DVC + pipeline | 15% | Bom (remote `./dvc-storage`) |
| Rede neural PyTorch | 15% | Forte (champion MLP, early stopping) |
| MLflow + Registry | 10% | Forte (≥ 3 runs + Production) |
| Vídeo STAR | 10% | Pendente |
| Bônus cloud | 5% | Opcional |

## Pendência obrigatória restante

1. **Vídeo STAR (6.5)** — único gap obrigatório de entrega fora do código.
