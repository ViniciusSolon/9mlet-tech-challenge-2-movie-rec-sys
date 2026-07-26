# Implementação MLP/Embedding PyTorch para Recomendação

**Branch:** `feat/bloco5-mlp-pytorch-training`  
**Responsável:** Vinicius (Vini)  
**Etapa do projeto:** Etapa 4 — Bloco 5 (Modelagem)  
**Task:** *Treinar MLP/embedding PyTorch para recomendação*

---

## O que foi feito

Substituição dos **stubs** (placeholders) dos modelos PyTorch por implementações reais, junto com a infraestrutura de treino (early stopping, seeds, MLflow) e avaliação com ≥ 4 métricas.

---

## Arquivos modificados / criados

| Arquivo | Tipo | O que faz |
|---|---|---|
| `src/models/torch_mlp.py` | Modificado | MLP com embeddings user+item concatenados → 2 camadas ocultas com ReLU e Dropout, inicialização Xavier, predição de rating |
| `src/models/torch_embedding.py` | Modificado | Dot-product collaborative filter: `<e_u, e_i> + b_u + b_i + bias_global`, estilo matrix factorisation |
| `src/models/factory.py` | Modificado | Lazy imports para modelos torch (evita `ImportError` quando torch não está instalado) |
| `src/evaluation/metrics.py` | Criado | Métricas de ranking e rating: Precision@K, Recall@K, NDCG@K, Hit Rate@K, RMSE, MAE |
| `src/evaluation/__init__.py` | Modificado | Exporta as funções de `metrics.py` como API pública do módulo |
| `scripts/train.py` | Modificado | Treino completo: val split, early stopping por `patience`, `ReduceLROnPlateau`, seeds globais, log MLflow (params, métricas por época, artefatos) |
| `scripts/evaluate.py` | Modificado | Avaliação com 6 métricas: RMSE, MAE, MSE + Precision@10, Recall@10, NDCG@10, Hit Rate@10 |
| `params.yaml` | Modificado | Adicionados parâmetros de treino: `hidden_dim`, `dropout`, `val_split`, `patience`, `seed` |
| `dvc.yaml` | Modificado | Stage `train` atualizado para passar os novos parâmetros ao script |
| `tests/unit/test_metrics.py` | Criado | 23 testes unitários cobrindo todas as métricas (casos limite, valores conhecidos, bounds) |
| `tests/unit/test_model_factory.py` | Modificado | Testes para os modelos torch (forward shape, predict antes de fit); skip automático quando torch não instalado |

---

## Arquitetura dos modelos

### `TorchMLPRecommender` (`torch_mlp.py`)

```
Embedding(user_idx, dim) ──┐
                            ├── concat → Linear(2*dim, hidden) → ReLU → Dropout
Embedding(item_idx, dim) ──┘          → Linear(hidden, hidden/2) → ReLU → Dropout
                                       → Linear(hidden/2, 1)
```

### `TorchEmbeddingRecommender` (`torch_embedding.py`)

```
score = <e_user, e_item>  +  bias_user  +  bias_item  +  global_bias
```

---

## Pipeline de treino (`scripts/train.py`)

1. Carrega `data/processed/features_ratings.parquet`
2. Aplica seeds globais (`training/seeds.py`)
3. Separa `val_split` % dos dados para validação
4. Treina por até `epochs` épocas com mini-batches
5. Ao fim de cada época: calcula `val_mse` e ajusta LR com `ReduceLROnPlateau`
6. **Early stopping**: para se `val_mse` não melhora por `patience` épocas consecutivas
7. Salva o checkpoint da melhor época em `models/model.pth`
8. Loga tudo no MLflow: params, `train_mse`/`val_mse` por época, `best_val_mse`, artefatos

---

## Métricas de avaliação (`scripts/evaluate.py`)

| Métrica | Tipo | Descrição |
|---|---|---|
| RMSE | Rating | Raiz do erro quadrático médio |
| MAE | Rating | Erro absoluto médio |
| MSE | Rating | Erro quadrático médio |
| Precision@10 | Ranking | Fração dos top-10 que são relevantes (rating ≥ 4.0) |
| Recall@10 | Ranking | Fração dos itens relevantes recuperados no top-10 |
| NDCG@10 | Ranking | Ganho cumulativo descontado normalizado no corte 10 |
| Hit Rate@10 | Ranking | 1 se pelo menos 1 item relevante está no top-10 |

---


---

## Como rodar os testes

```bash
# Com Docker (sem torch instalado — testes torch são pulados automaticamente)
docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim \
  bash -c "pip install pytest numpy pandas --no-cache-dir && \
           PYTHONPATH=src python -m pytest tests/unit/test_metrics.py tests/unit/test_model_factory.py -v"

# Com o venv do projeto (torch necessário para os testes de modelo)
PYTHONPATH=src pytest tests/unit/test_metrics.py tests/unit/test_model_factory.py -v
```

---

## Parâmetros de treino (`params.yaml`)

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `epochs` | 10 | Máximo de épocas |
| `lr` | 0.001 | Learning rate inicial |
| `batch_size` | 1024 | Tamanho do mini-batch |
| `embedding_dim` | 32 | Dimensão dos embeddings |
| `hidden_dim` | 128 | Largura da primeira camada oculta (MLP) |
| `dropout` | 0.2 | Probabilidade de dropout |
| `val_split` | 0.1 | Fração dos dados para validação |
| `patience` | 3 | Épocas sem melhora para early stopping |
| `seed` | 42 | Seed global para reprodutibilidade |
