# Model Card — Movie Rec Sys (MovieLens 100k / 20M)

**Projeto:** FIAP Tech Challenge Fase 02 — Sistema de Recomendação  
**Modelo em Produção (Registry):** `torch_mlp` (versão promovida para Production)  
**Arquiteturas Candidatas:** `torch_mlp` (padrão neural), `torch_embedding`  
**Baselines:** `most_popular`, `sklearn_knn`, `sklearn_random_forest`  
**Última Atualização:** 2026-07-26  
**Responsável:** Grupo 9MLET

---

## 1. Descrição do Modelo

Sistema de recomendação **colaborativo neural** treinado com **PyTorch**, mapeado para o contexto de e-commerce:

| MovieLens | E-Commerce |
|-----------|------------|
| `userId` | Cliente / Usuário |
| `movieId` | Produto (SKU) |
| `rating` + `timestamp` | Interação / Engajamento com timestamp |

### `TorchMLPRecommender` (Modelo Central / Champion)

Embeddings de usuário e item concatenados passados por uma MLP (2 camadas ocultas com ReLU, Dropout e ajuste dinâmico):

```text
e_u (32D), e_i (32D) ──► Concat (64D) ──► Linear(128) ──► ReLU ──► Dropout(0.2) ──► Linear(64) ──► ReLU ──► Dropout(0.2) ──► Linear(1)
```

### `TorchEmbeddingRecommender` (Arquitetura Alternativa)

Filtro colaborativo por produto interno + termos de viés:
$$\text{score} = \langle \mathbf{e}_u, \mathbf{e}_i \rangle + b_u + b_i + b_{\text{global}}$$

### Treinamento

- **Função de Perda:** Perda Quadrática Média (MSE)  
- **Otimizador:** Adam com scheduler `ReduceLROnPlateau`  
- **Early Stopping:** Monitoramento da perda no conjunto de validação (`val_mse`, `patience=3`)  
- **Checkpoint de Produção:** `models/model.pth`  
- **Sementes Globais:** Fixadas em 42 (`src/training/seeds.py`)  
- **MLflow Tracking:** Registro de hiperparâmetros, curva de perda por época e artefato do modelo PyTorch

---

## 2. Dados de Treinamento

| Item | Detalhe |
|------|---------|
| **Dataset** | MovieLens 100k (100.836 avaliações de 610 usuários em 9.742 filmes/produtos) |
| **Arquivos Brutos** | `ratings.csv`, `movies.csv`, `links.csv` em `data/raw/` |
| **Metadados** | `data/processed/enriched_metadata.parquet` (Integração com API do TMDB) |
| **Feedback** | Explícito (estrelas de 0.5 a 5.0); limiar de relevância configurável (`rating ≥ 4.0`) |
| **Features no Modelo Neural** | `user_idx`, `movie_idx` (Indexação contígua de 0 a $N-1$) |

### Política de Particionamento (Split)

Utiliza **split temporal estrito** via `timestamp` (`src/data/splits.py`):
1. Ordena o histórico temporal do usuário.
2. Reserva os últimos `test_ratio` (20%) para hold-out de teste sem contaminação de dados futuros (*zero data leakage*).
3. Validação garantida por `assert_temporal_order`.

---

## 3. Métricas de Performance

Avaliadas em hold-out temporal (`scripts/evaluate.py`), cobrindo as 7 métricas registradas no `metrics.json`:

### Leitura das Métricas

| Métrica | Em Português Simples | Objetivo |
|---------|----------------------|----------|
| **RMSE** | Erro médio quadrático das notas previstas (penaliza erros grandes). | **Menor é melhor ↓** |
| **MAE** | Erro médio absoluto das notas previstas em estrelas. | **Menor é melhor ↓** |
| **R²** | Proporção da variância do rating explicada pelo modelo. | **Maior é melhor ↑** |
| **Precision@10** | Proporção de itens relevantes entre os 10 recomendados. | **Maior é melhor ↑** |
| **Recall@10** | Proporção de itens relevantes recuperados no Top-10. | **Maior é melhor ↑** |
| **NDCG@10** | Qualidade da ordenação do ranking (acertar no topo vale mais). | **Maior é melhor ↑** |
| **Hit Rate@10** | Percentual de usuários que receberam pelo menos 1 item relevante no Top-10. | **Maior é melhor ↑** |

### Resultados da Avaliação Oficial (`metrics.json`)

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ | Status no Registry |
|--------|--------|-------|------|--------------------|
| 🏆 **`torch_mlp` (Neural MLP)** | **0.8984** | **0.6990** | **0.1906** | **Production** |
| 🥈 `most_popular` | 0.9626 | 0.7620 | 0.0710 | Baseline |
| 🥉 `sklearn_random_forest` | 0.9975 | 0.7959 | 0.0022 | Baseline |
| 4️⃣ `sklearn_knn` | 1.0155 | 0.8122 | -0.0340 | Baseline |

> **Conclusão:** O modelo neural **PyTorch MLP (`torch_mlp`) superou todos os baselines**, alcançando o menor erro (RMSE **0,8984** vs 0,9626 do MostPopular) e sendo promovido para a versão oficial em **Production** no MLflow Model Registry.

---

## 4. Limitações Conhecidas

| Limitação | Impacto de Negócio | Mitigação no Pipeline |
|-----------|--------------------|-----------------------|
| **Cold Start (Novo Usuário)** | Sem histórico de notas | Fallback para o modelo `most_popular` |
| **Cold Start (Novo Produto)** | Poucos ratings registrados | Enriquecimento via metadados TMDB (`enriched_metadata.parquet`) |
| **Viés de Popularidade** | Skew para itens muito conhecidos | Comparação contra MostPopular para monitorar o ganho de personalização |
| **Escala de Avaliação** | Custo computacional de ranking | Inspecção amostrada por lote com limite em `metrics.py` |

---

## 5. Vieses e Considerações Éticas

- **Privacidade (PII):** Os identificadores de usuário e produto no dataset são puramente numéricos e pseudonimizados.
- **Viés de Catálogo:** O modelo colaborativo herda a distribuição do histórico do usuário e pode favorecer itens com maior volume de votos.
- **Uso Recomendado:** Recomendação de catálogo e e-commerce. Não aplicável para scoring financeiro, crédito ou decisões de alto risco.

---

## 6. Reprodutibilidade

| Etapa | Comando / Arquivo |
|-------|-------------------|
| **Instalação** | `uv sync --extra dev` |
| **Sanidade do Ambiente** | `uv run python scripts/validate_env.py` (25/25 checks) |
| **Download de Dados** | `uv run python scripts/download_dataset.py --variant small` |
| **Pipeline MLOps DVC** | `uv run dvc repro` (5 stages automatizados) |
| **Containerização** | `docker compose up --build` |
| **Interface MLflow** | `http://localhost:5000` |

---

## Referências

- Requisitos Oficiais: `docs/Tech Challenge Fase 02.pdf` / `docs/pdf_extract.txt`  
- Auditoria do Projeto: `AUDITORIA_TECH_CHALLENGE.md`  
- Arquivo de Métricas: `metrics.json`  
