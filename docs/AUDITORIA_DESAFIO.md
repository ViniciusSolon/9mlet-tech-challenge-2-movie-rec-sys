# Auditoria — Aderência ao Tech Challenge Fase 02

Cruzamento do enunciado (`docs/pdf_extract.txt`) com o estado do repositório em **2026-07-24** (branch de modelagem + refatoração Vítor).

## Entregáveis do PDF

| Exigência | Status | Evidência |
|-----------|--------|-----------|
| Repo GitHub + clean code | ✅ | `src/`, SOLID, Factory, Strategy, ruff, testes |
| `pyproject.toml` + lock | ✅ | `pyproject.toml`, `uv.lock` |
| `.dockerignore` / `.gitignore` / `.env.example` | ✅ | raiz do repo |
| Commits semânticos | ✅ | histórico `feat:` / `docs:` / `test:` |
| PyTorch MLP/embedding | ✅ | `torch_mlp.py`, `torch_embedding.py`, `train.py` |
| Scikit-Learn baselines | ✅ | KNN, RandomForest + **MostPopular** |
| ≥ 4 métricas | ✅ | RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate |
| MLflow tracking | ✅ | `train.py` / `evaluate.py` |
| Model Registry Staging→Production | ✅ | `evaluation/registry.py` |
| **Model Card** | ✅ | `docs/MODEL_CARD.md` |
| Dockerfile multi-stage | ✅ | `Dockerfile` |
| docker-compose train + MLflow | ✅ | `docker-compose.yml` |
| DVC ≥ 3 stages + `dvc repro` | ✅ | `dvc.yaml` (5 stages) + `dvc.lock` |
| Seeds + `.env` + validate_env | ✅ | `seeds.py`, `settings.py`, `validate_env.py` |
| README com instruções | 🟡 | atualizado nesta refatoração; expandir se necessário |
| Vídeo STAR ≤ 5 min | ❌ | fora do código — pendente Fernando/equipe |
| Deploy nuvem (opcional) | ❌ | bônus — não iniciado |

## Critérios de nota (pesos)

| Critério | Peso | Avaliação interna |
|----------|------|-------------------|
| Clean code | 15% | Forte (refatorar scripts longos continua desejável) |
| Reprodutibilidade | 15% | Forte (`uv sync`) |
| Docker | 15% | Bom (compose funcional; imagem ainda depende de volume) |
| DVC + pipeline | 15% | Bom (paths Kaggle singular alinhados) |
| Rede neural PyTorch | 15% | Forte (early stopping + baselines) |
| MLflow + Registry | 10% | Bom (≥ 3 runs no evaluate: popular + 2 sklearn + torch) |
| Vídeo STAR | 10% | Pendente |
| Bônus cloud | 5% | Opcional |

## Lacunas conscientes (não bloqueiam o PDF, mas melhoram a nota/STAR)

1. **BERTopic** no `feature_eng` — diferencial do plano interno; **não** é requisito literal do PDF.  
2. **Ablation** colaborativo vs +TMDB vs +BERTopic — recomendado para o vídeo.  
3. **EDA formal** e script Kaggle download — qualidade de dados.  
4. Preencher tabela numérica do Model Card após o evaluate de entrega.  
5. CI GitHub Actions (`pytest` + `ruff`).

## Refatorações feitas nesta rodada (Vítor)

- Model Card criado  
- Nomes MovieLens flexíveis (`rating.csv` / `ratings.csv`, etc.) + `dvc.yaml` alinhado ao Kaggle  
- Split temporal preferencial (`src/data/splits.py`) em train/evaluate  
- Baseline **MostPopular** + tabela de comparação no `evaluate.py`  
- RMSE via `root_mean_squared_error` (sklearn ≥ 1.5)  
- README e TODO atualizados
