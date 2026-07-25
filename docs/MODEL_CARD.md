# Model Card — Movie Rec Sys (MovieLens 20M)

**Projeto:** FIAP Tech Challenge Fase 02 — sistema de recomendação  
**Modelo em produção (Registry):** `torch_mlp`  
**Arquiteturas candidatas:** `torch_mlp` (padrão), `torch_embedding`  
**Baselines:** `most_popular`, `sklearn_knn`, `sklearn_random_forest`  
**Última atualização:** 2026-07-25  
**Responsável (Model Card):** Vítor

---

## 1. Descrição do modelo

Sistema de recomendação **colaborativo** treinado com **PyTorch**, analogia e-commerce do enunciado:

| MovieLens | E-commerce |
|-----------|------------|
| `userId` | Cliente |
| `movieId` | Produto (SKU) |
| `rating` + `timestamp` | Interação / engajamento |

### `TorchMLPRecommender` (modelo central / champion)

Embeddings de usuário e item concatenados → MLP (2 camadas ocultas, ReLU, Dropout) → score de rating.

```
e_u, e_i → concat → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)
```

### `TorchEmbeddingRecommender` (alternativa)

Filtro colaborativo por produto interno + biases:

`score = ⟨e_u, e_i⟩ + b_u + b_i + b_global`

### Treino

- Loss: MSE  
- Otimizador: Adam + `ReduceLROnPlateau`  
- **Early stopping** por `val_mse` (`patience` em `params.yaml`)  
- Checkpoint: `models/model.pth`  
- Seeds globais: `src/training/seeds.py`  
- Tracking: MLflow (params, métricas por época, artefato do modelo)

Código: `src/models/torch_mlp.py`, `src/models/torch_embedding.py`, `scripts/train.py`.

---

## 2. Dados de treinamento

| Item | Detalhe |
|------|---------|
| **Dataset** | [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) (≥ 10k interações; atende o PDF) |
| **Arquivos** | `rating(s).csv`, `movie(s).csv`, `link(s).csv` — nomes GroupLens **ou** Kaggle |
| **Metadados** | `data/processed/movie_metadata.parquet` (TMDB, etapa de scraping) |
| **Feedback** | Explícito (estrelas); limiar implícito configurável (`rating ≥ 4`) |
| **Features no modelo neural atual** | `user_idx`, `movie_idx` (IDs contíguos); colunas de conteúdo podem ser anexadas no parquet |

### Política de split

Preferência por **split temporal** em `timestamp` (`src/data/splits.py`):

1. Ordena por tempo  
2. Reserva os últimos `test_ratio` (padrão 20% na avaliação; `val_split` no treino) para hold-out  
3. Se não houver `timestamp`, faz fallback aleatório com seed fixa  

Regra de negócio: sem leakage de futuro → passado (`assert_temporal_order`).

---

## 3. Métricas de performance

Avaliadas em hold-out temporal (`scripts/evaluate.py`, jul/2026), com **≥ 4 métricas** exigidas pelo desafio.

### Leitura leiga

| Métrica | Em português simples | Melhor quando… |
|---------|----------------------|----------------|
| **RMSE** | Quanto erramos a nota (penaliza erros grandes). | **Menor** |
| **MAE** | Erro médio absoluto em estrelas. | **Menor** |
| **Precision@10** | Dos 10 sugeridos, quantos o usuário gostaria (nota ≥ 4)? | **Maior** |
| **Recall@10** | Dos filmes que ele gostaria no teste, quantos entraram no top 10? | **Maior** |
| **NDCG@10** | A ordem da lista está boa? | **Maior** |
| **Hit Rate@10** | Em quantos % dos usuários o top 10 tem ≥ 1 acerto? | **Maior** |

### Resultados do run atual (MovieLens completo no workspace)

| Modelo | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ | HR@10 ↑ |
|--------|--------|-------|--------|--------|-----------|---------|
| **torch_mlp (champion)** | **0,780** | **0,600** | **0,150** | **0,037** | **0,160** | **0,610** |
| most_popular | 0,887 | 0,691 | — | — | — | — |
| sklearn_random_forest | 0,906 | 0,720 | — | — | — | — |
| sklearn_knn | 0,948 | 0,760 | — | — | — | — |

> Ranking @10 do neural: até 200 usuários no hold-out. Baselines: RMSE/MAE em amostra (≤200k treino / ≤50k teste).

**Champion no Registry:** `torch_mlp` → `movie-rec-sys` **v2 / Production**  
**Run MLflow:** `a6845d82d6a54901a1f5d1e5a9d8f5d2` · split temporal OK · `metrics.json`

### Conclusão (para o vídeo STAR)

1. O **MLP PyTorch** é o modelo central e o champion em Production.  
2. Baselines ficam atrás em RMSE neste run.  
3. HR@10 ≈ 61% na amostra de ranking.  
4. BERTopic/conteúdo no forward são opcionais (extra `topics`) e **não** exigidos pelo PDF.

---

## 4. Limitações

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| Cold start (usuário novo) | Embeddings fracos | Fallback MostPopular |
| Cold start (item novo) | Poucos ratings | Metadados TMDB no parquet |
| Popularidade / long-tail | Skew de itens famosos | MostPopular expõe o viés |
| Score colaborativo puro | Sem sinopse no forward | Enrich pronto; conteúdo anexável |
| Escala 20M | Custo de treino/eval | Sampling + Docker + DVC |

---

## 5. Vieses e considerações éticas

- IDs MovieLens são **pseudônimos**; não há PII explícita no dataset.  
- Recomendações podem reforçar **vieses de popularidade** e de catálogo ocidental.  
- Não usar o modelo para decisões sensíveis.  
- Analogia e-commerce é **didática**.

---

## 6. Reprodutibilidade

| Passo | Comando / artefato |
|-------|--------------------|
| Ambiente | `uv sync --extra dev` (+ `uv.lock`) |
| Config | `.env` a partir de `.env.example` |
| Validação | `python scripts/validate_env.py` |
| Seeds | `params.yaml` → `train.seed` (42) |
| Pipeline | `dvc repro` |
| Docker | `docker compose run train` + MLflow `:5000` |
| Registry | Staging → Production via `MLflowRegistryManager` |

**MLflow local:** prefira `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` no `.env`. Use `http://localhost:5000` apenas com `docker compose up mlflow`.

---

## 7. Intended use

**Uso pretendido:** demonstração acadêmica de pipeline MLOps de recomendação.  

**Fora de escopo:** UI completa, auth, streaming em tempo real, deploy obrigatório em nuvem (bônus opcional).

---

## Referências

- Enunciado: `docs/Tech Challenge Fase 02.pdf` / `docs/pdf_extract.txt`  
- Auditoria: `AUDITORIA_TECH_CHALLENGE.md`  
- Business rules: `.cursor/rules/business-rules.md`
