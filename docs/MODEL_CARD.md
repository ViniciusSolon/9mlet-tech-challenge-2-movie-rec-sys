# Model Card — Movie Rec Sys (MovieLens 100k / 20M)

**Projeto:** FIAP Tech Challenge Fase 02 — sistema de recomendação  
**Modelo em produção (Registry):** `torch_mlp` (versão 3 em Production)  
**Arquiteturas candidatas:** `torch_mlp` (padrão), `torch_embedding`  
**Baselines:** `most_popular`, `sklearn_knn`, `sklearn_random_forest`  
**Última atualização:** 2026-07-26  
**Responsável (Model Card):** Vítor

---

## 1. Descrição do modelo

Sistema de recomendação **colaborativo** treinado com **PyTorch**, analogia e-commerce do enunciado:

| MovieLens | E-commerce |
|-----------|------------|
| `userId` | Cliente |
| `movieId` | Produto (SKU) |
| `rating` + `timestamp` | Interação / engajamento |

### `TorchMLPRecommender` (champion oficial)

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
| **Dataset** | MovieLens 100k (100.836 avaliações; atende com folga o mínimo de 10k do PDF) |
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
| **MAE** | “Em média, erramos a nota em quantos pontos?” Ex.: MAE 0,69 ≈ erramos ~0,69 estrela. | **Menor** |
| **Precision@10 (P@10)** | “Dos 10 filmes que sugerimos, quantos o usuário realmente gostaria (nota ≥ 4)?” | **Maior** |
| **Recall@10 (R@10)** | “Dos filmes que ele gostaria no período de teste, quantos caíram no nosso top 10?” | **Maior** |
| **NDCG@10** | “A ordem da lista está boa?” Acertar no 1º lugar vale mais que acertar no 10º. | **Maior** |
| **Hit Rate@10 (HR@10)** | “Em quantos % dos usuários o top 10 tem **pelo menos 1** filme que ele gostaria?” | **Maior** |

### Resultados do run oficial (`dvc repro` completo)

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ | Status |
|--------|--------|-------|------|--------|
| **torch_mlp (champion)** | **0,898** | **0,699** | **0,191** | **Production** |
| most_popular | 0,963 | 0,762 | 0,071 | Baseline |
| sklearn_random_forest | 0,998 | 0,796 | 0,002 | Baseline |
| sklearn_knn | 1,016 | 0,812 | -0,034 | Baseline |

> O modelo neural **PyTorch MLP (`torch_mlp`) superou todos os baselines** obtendo o menor RMSE (0,898 vs 0,963 do MostPopular) e foi promovido automaticamente para o estágio **Production** no Model Registry do MLflow.

**Champion no Registry:** `torch_mlp` → `movie-rec-sys` **v3 / Production**  
**Run MLflow:** `b4c9b6b961d54f0eadee03b99feee29e` · split temporal OK · `metrics.json`

### Conclusão (para o time / vídeo STAR)

1. A arquitetura **PyTorch MLP** foi a campeã absoluta com **RMSE de 0,898** e **MAE de 0,699**, superando com margem clara os baselines Scikit-Learn e MostPopular.
2. O modelo foi devidamente registrado no **MLflow Model Registry** e promovido para o estágio de **Production**.
3. O pipeline é 100% reprodutível via `dvc repro`, gerando o [dvc.lock](file:///C:/Users/Fernando%20Azevedo/FIAP/9mlet-tech-challenge-2-movie-rec-sys/dvc.lock) e o [metrics.json](file:///C:/Users/Fernando%20Azevedo/FIAP/9mlet-tech-challenge-2-movie-rec-sys/metrics.json).

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
