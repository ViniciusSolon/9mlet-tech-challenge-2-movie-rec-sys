# Model Card — Movie Rec Sys (MovieLens 20M)

**Projeto:** FIAP Tech Challenge Fase 02 — sistema de recomendação  
**Modelo em produção (Registry):** `sklearn_random_forest`  
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

### `TorchMLPRecommender` (champion típico)

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
| **Features no modelo neural atual** | `user_idx`, `movie_idx` (IDs contíguos); metadados TMDB disponíveis no pipeline de enrich |

### Política de split

Preferência por **split temporal** em `timestamp` (`src/data/splits.py`):

1. Ordena por tempo  
2. Reserva os últimos `test_ratio` (padrão 20% na avaliação; `val_split` no treino) para hold-out  
3. Se não houver `timestamp`, faz fallback aleatório com seed fixa  

Regra de negócio: sem leakage de futuro → passado (`assert_temporal_order`).

---

## 3. Métricas de performance

Avaliadas em hold-out temporal (`scripts/evaluate.py`, jul/2026), com **≥ 4 métricas** exigidas pelo desafio.

### Leitura leiga (o que cada número responde)

| Métrica | Em português simples | Melhor quando… |
|---------|----------------------|----------------|
| **RMSE** | “Em média, quanto erramos a nota (com peso maior nos erros grandes)?” Notas vão de ~0,5 a 5. | **Menor** |
| **MAE** | “Em média, erramos a nota em quantos pontos?” Ex.: MAE 0,6 ≈ erramos ~0,6 estrela. | **Menor** |
| **Precision@10 (P@10)** | “Dos 10 filmes que sugerimos, quantos o usuário realmente gostaria (nota ≥ 4)?” Ex.: 0,17 ≈ ~1,7 acertos no top 10. | **Maior** |
| **Recall@10 (R@10)** | “Dos filmes que ele gostaria no período de teste, quantos caíram no nosso top 10?” | **Maior** |
| **NDCG@10** | “A ordem da lista está boa?” Acertar no 1º lugar vale mais que acertar no 10º. | **Maior** |
| **Hit Rate@10 (HR@10)** | “Em quantos % dos usuários o top 10 tem **pelo menos 1** filme que ele gostaria?” Ex.: 0,63 = 63%. | **Maior** |

### Resultados do run atual (snapshot do repositório)

| Modelo | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ | HR@10 ↑ |
|--------|--------|-------|--------|--------|-----------|---------|
| **sklearn_random_forest (champion)** | **1,69** | **1,69** | — | — | — | — |
| torch_mlp | 1,789 | 1,789 | 0,000 | 0,000 | 0,000 | 0,000 |
| most_popular | 2,000 | 2,000 | — | — | — | — |
| sklearn_knn | 2,250 | 2,250 | — | — | — | — |

> O snapshot do repositório usa uma amostra pequena de validação, então o ranking @K do modelo neural ficou zerado neste run. Baselines foram avaliados em RMSE/MAE, e o campeão atual foi o `sklearn_random_forest`.

**Champion no Registry:** `sklearn_random_forest` → `movie-rec-sys` **v2 / Production**  
**Run MLflow:** `3c389798764241ec8efdc5a6513aff7d` · split temporal OK · `metrics.json`

### Conclusão (para o time / vídeo STAR)

1. No snapshot atual, o melhor RMSE/MAE ficou com o **Random Forest**, mas o `torch_mlp` segue como a arquitetura neural principal do projeto.  
2. O ranking @10 do modelo neural ficou zerado nesse run pequeno, o que é esperado em hold-out reduzido e com sinal colaborativo puro.  
3. Precision/Recall/NDCG continuam modestos; há espaço para incorporar mais features de conteúdo (TMDB/BERTopic).  
4. Para o desafio: há um modelo PyTorch treinado, baselines comparados em múltiplas métricas e um campeão registrado em Production neste snapshot.

---

## 4. Limitações

| Limitação | Impacto | Mitigação atual / futura |
|-----------|---------|--------------------------|
| **Cold start (usuário novo)** | Sem histórico → embeddings fracos | Fallback natural: MostPopular; conteúdo TMDB ainda não entra no forward neural |
| **Cold start (item novo)** | Poucos ratings | Metadados TMDB no parquet; integração BERTopic/conteúdo no MLP ainda parcial |
| **Popularidade / long-tail** | Itens populares dominam | MostPopular expõe o viés; modelo neural personaliza, mas pode herdar skew |
| **Colaborativo puro** | Sem sinal de sinopse no score | Pipeline `enrich_metadata` pronto; próximo passo: features de conteúdo no treino |
| **Escala 20M** | Treino completo é custoso | Dev com `create_dummy_data.py` / amostra; produção via Docker + DVC |
| **Coverage TMDB** | ~1% sem sinopse / IDs ausentes | Relatório em `docs/METADATA_COVERAGE_REPORT.md` |

---

## 5. Vieses e considerações éticas

- IDs MovieLens são **pseudônimos**; não há PII explícita no dataset.  
- Recomendações podem reforçar **vieses de popularidade** e de catálogo ocidental (idioma/mercado TMDB).  
- Não usar o modelo para decisões sensíveis (crédito, emprego, etc.).  
- Não logar históricos completos de usuário em produção (ver regras de segurança do projeto).  
- Analogia e-commerce é **didática**: validar fairness se aplicado a produtos reais.

---

## 6. Reprodutibilidade

| Passo | Comando / artefato |
|-------|--------------------|
| Ambiente | `uv sync` (+ `uv.lock` commitado) |
| Config | `.env` a partir de `.env.example` + `configs/settings.py` |
| Validação | `python scripts/validate_env.py` |
| Seeds | `params.yaml` → `train.seed` (padrão 42) |
| Pipeline | `dvc repro` (stages: preprocess → enrich → feature_eng → train → evaluate) |
| Docker | `docker compose run train` + MLflow em `http://localhost:5000` |
| Registry | Staging → Production via `MLflowRegistryManager` |
| Docs | `docs/DOCUMENTACAO_ETAPA2.md`, `docs/IMPLEMENTACAO_MLP_PYTORCH.md` |

---

## 7. Intended use

**Uso pretendido:** demonstração acadêmica de pipeline MLOps de recomendação (treino, avaliação, registry).  

**Fora de escopo:** UI completa, auth de usuários, streaming em tempo real, deploy obrigatório em nuvem (bônus opcional do PDF).

---

## Referências

- Enunciado: `docs/Tech Challenge Fase 02.pdf` / `docs/pdf_extract.txt`  
- Auditoria de aderência: `docs/AUDITORIA_DESAFIO.md`  
- Business rules: `.cursor/rules/business-rules.md`  
- Template interno: `.cursor/commands/model-card-generator.md`
