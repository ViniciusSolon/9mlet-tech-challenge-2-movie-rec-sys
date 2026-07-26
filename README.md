# 🎬 MovieLens & E-Commerce Recommendation System (`movie-rec-sys`)

Uma plataforma completa e de nível de produção para **Sistemas de Recomendação Personalizados em E-Commerce**, combinando modelos profundos em PyTorch, pipeline de dados MLOps reprodutível com DVC, rastreamento de experimentos com MLflow, containerização com Docker e governança através do MLflow Model Registry.

---

## 📌 1. Visão Geral e Contexto de Negócio

Em plataformas modernas de e-commerce, apresentar o produto certo ao cliente no momento ideal é um fator decisivo para aumentar a conversão, a taxa de retenção e o valor do tempo de vida do cliente (LTV). 

Esta plataforma implementa uma solução de **Filtragem Colaborativa Neural** treinada com interações reais e enriquecida com metadados de catálogo.

### 🔄 Mapeamento de Domínio para E-Commerce

| Conceito no Pipeline | Equivalente no E-Commerce | Descrição Técnica |
|---|---|---|
| `userId` | **Cliente / Usuário** | Identificador único do cliente na plataforma. |
| `movieId` | **Produto / SKU** | Item do catálogo elegível para recomendação. |
| `rating` (0.5 a 5.0) | **Avaliação / Engajamento** | Feedback explícito de satisfação e preferência do cliente. |
| `timestamp` | **Data / Hora da Interação** | Marca temporal utilizada para particionamento sequencial sem vazamento de futuro (*leakage*). |
| `movie_metadata.parquet` | **Catálogo de Produtos** | Metadados enriquecidos via API do TMDB (gênero, popularidade, sinopse). |

---

## 🏗️ 2. Arquitetura de Software e Design Patterns

A aplicação segue uma arquitetura em camadas bem definida, priorizando o desacoplamento de responsabilidades e a extensibilidade.

```text
       [data/raw/ (ratings.csv, movies.csv, links.csv)]
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Stage 1: preprocess      │ ──► Strategy Pattern (Explicit/Implicit)
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Stage 2: enrich_metadata │ ──► Join Parquet (TMDB Catalog)
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Stage 3: feature_eng     │ ──► Contiguous Indexing (User/Item)
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Stage 4: train           │ ──► PyTorch MLP + Early Stopping
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Stage 5: evaluate        │ ──► Baselines + MLflow Registry
                 └────────────┬─────────────┘
```

### 🧩 Padrões de Projeto (Design Patterns)

1. **Factory Pattern ([src/models/factory.py](src/models/factory.py)):**
   - Centraliza a instanciação dinâmica de modelos de recomendação (`create_model("torch_mlp")`, `create_model("most_popular")`, `create_model("sklearn_baseline")`), permitindo a inclusão de novas arquiteturas sem modificar o código do pipeline de treinamento.

2. **Strategy Pattern ([src/data/preprocessors/](src/data/preprocessors/)):**
   - Abstrai os algoritmos de pré-processamento de interações (`ExplicitPreprocessor` para notas/estrelas e `ImplicitPreprocessor` para cliques/visualizações).

### 📁 Estrutura de Módulos (`src/`)

```text
src/
├── domain/            # Entidades puras de negócio (User, Movie, Rating)
├── data/              # Leitura I/O, splits temporais e pré-processadores (Strategy)
├── models/            # Implementações em PyTorch, Scikit-Learn e Factory
├── evaluation/        # Métricas de ranking/rating, runner, champion selector e MLflow Registry
├── training/          # Loop de treino PyTorch e fixação de sementes globais (seeds)
├── serving/           # Aplicação FastAPI, Web Dashboard e endpoints de recomendação
└── utils/             # Resolução dinâmica de caminhos e logging
```

---

## 🤖 3. Modelo Central Neural PyTorch & Desempenho

### 🧠 Arquitetura `TorchMLPRecommender`

O modelo principal é uma **Rede Neural Perceptron Multicamadas (MLP)** desenvolvida em PyTorch:
- **Embeddings Densos:** Aprende vetores latentes para usuários ($\mathbf{e}_u \in \mathbb{R}^{32}$) e produtos ($\mathbf{e}_i \in \mathbb{R}^{32}$).
- **Concatenação:** Unifica as representações ($\mathbf{x} = [\mathbf{e}_u \,||\, \mathbf{e}_i] \in \mathbb{R}^{64}$).
- **Camadas Densas:** 
  $$\text{Linear}(64 \rightarrow 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(128 \rightarrow 64) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(64 \rightarrow 1)$$
- **Treinamento & Regularização:** Otimizador Adam, Perda Quadrática Média (MSE), ajuste de taxa de aprendizado via `ReduceLROnPlateau` e **Early Stopping** monitorando a perda na validação (`patience=3`).

### 📊 Desempenho do Modelo vs. Baselines (`metrics.json`)

Avaliação realizada em conjunto de teste com particionamento temporal estrito:

| Modelo | RMSE ↓ | MAE ↓ | R² ↑ | Status no MLflow Registry |
|---|---|---|---|---|
| 🏆 **`torch_mlp` (PyTorch Neural MLP)** | **0.8984** | **0.6990** | **0.1906** | **Production** |
| 🥈 `most_popular` (Baseline) | 0.9625 | 0.7620 | 0.0709 | Baseline |
| 🥉 `sklearn_random_forest` (Baseline) | 0.9975 | 0.7959 | 0.0022 | Baseline |
| 4️⃣ `sklearn_knn` (Baseline) | 1.0155 | 0.8122 | -0.0340 | Baseline |

---

## 💻 4. Guia de Instalação e Execução Local

### 🛠️ Pré-requisitos
- **Python 3.11+**
- **uv** (gerenciador de dependências e ambiente) ou `pip`
- **Git**
- **Docker & Docker Compose** (opcional para ambiente containerizado)

---

### 🚀 Executando o Projeto

#### 1. Clonar o Repositório
```bash
git clone https://github.com/ViniciusSolon/9mlet-tech-challenge-2-movie-rec-sys.git
cd 9mlet-tech-challenge-2-movie-rec-sys
```

#### 2. Instalar Dependências com `uv`
```bash
uv sync --extra dev
```

#### 3. Configurar Variáveis de Ambiente
Crie o arquivo `.env` a partir do arquivo de exemplo:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

#### 4. Validar o Ambiente
```bash
uv run python scripts/validate_env.py
```

#### 5. Baixar o Dataset de Dados
```bash
uv run python scripts/download_dataset.py --variant small
```

#### 6. Executar o Pipeline MLOps Reprodutível (DVC)
```bash
uv run dvc repro
```

#### 7. Executar Suíte de Testes Automatizados
```bash
uv run pytest tests/unit
```

#### 8. Verificar Conformidade de Código (Linter)
```bash
uv run ruff check .
```

---

## 🌐 5. Aplicação Web & REST API Servidora (FastAPI)

A aplicação conta com uma camada de serviço desenvolvida em **FastAPI** (`src/serving/app.py`), combinando uma **interface gráfica interativa** e uma **API REST para inferência em tempo real**.

### 🎨 Recursos Disponíveis

1. **Dashboard Web Interativo (`GET /`):**
   - Painel gráfico em Dark Mode para demonstrações.
   - Permite selecionar o **Cliente (User ID)** e a quantidade de recomendações (**Top-K**).
   - Realiza inferência ao vivo na Rede Neural PyTorch e exibe os cards dos produtos recomendados com nota prevista em estrelas (ex: `★ 4.85`), taxa de match (`96.8% Match`), gêneros e **latência da API em milissegundos** (`⚡ 8 ms`).

2. **Documentação Swagger OpenAPI (`GET /docs`):**
   - Interface padrão do FastAPI para navegação e teste interativo das rotas HTTP da API REST.

3. **Endpoints Principais da API REST:**
   - `GET /health` — Retorna o estado de saúde do servidor, dispositivo PyTorch (`cpu`/`cuda`), modelo ativo (`torch_mlp`) e estágio no MLflow (`production`).
   - `GET /api/v1/recommend/{user_id}?top_k=10` — Gera os Top-K produtos recomendados para o cliente informado.
   - `GET /api/v1/metrics` — Retorna as métricas oficiais registradas em `metrics.json`.

### 🚀 Como Executar o Servidor da API Localmente

Com o seu ambiente virtual ativado (`.venv`), execute:

```bash
python -m uvicorn src.serving.app:app --reload --port 8000
```

*Nota: Se estiver usando `uv` global instalado no sistema, você também pode usar `uv run uvicorn src.serving.app:app --reload --port 8000`.*

Abra no seu navegador:
- **Painel Visual Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Documentação Swagger API:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 🧪 Exemplos de Requisições cURL

```bash
# 1. Checar saúde do serviço e modelo ativo
curl http://localhost:8000/health

# 2. Obter recomendações Top-5 para o Cliente #42
curl http://localhost:8000/api/v1/recommend/42?top_k=5

# 3. Consultar métricas de avaliação do modelo
curl http://localhost:8000/api/v1/metrics
```

---

## 🐳 6. Containerização e MLflow Tracking Server

Toda a infraestrutura de treinamento, serviço web e monitoramento pode ser inicializada via Docker Compose:

### 1. Subir os Serviços
```bash
docker compose up --build
```

### 2. Acessar a Interface do MLflow
Abra o navegador em: **`http://localhost:5000`**

Na interface do MLflow é possível inspecionar:
- Experimentos e curva de perda por época do treinamento neural.
- Métricas comparativas com os modelos baselines.
- A aba **Model Registry** onde a versão do modelo `torch_mlp` é promovida para **Production**.

---

## 📊 7. Governança MLOps & Model Card

Para garantir a transparência, rastreabilidade e uso responsável do modelo:
- **Model Card:** Consulte [docs/MODEL_CARD.md](docs/MODEL_CARD.md) para documentação sobre escopo, limitações e análises éticas de viés de popularidade.

---

## ☁️ 8. Deploy em Nuvem (Render)

A aplicação conta com suporte a deploy via container Docker:
- **URL do Serviço:** `https://9mlet-tech-challenge-2-movie-rec-sys.onrender.com`

---

## 👥 Contribuidores (Contributors)

Agradecimentos aos desenvolvedores que contribuíram para a construção deste projeto:

- **Eduardo Aleixo** ([@rm373692](https://github.com/rm373692))
- **Fernando Azevedo** ([@FernandoAzve](https://github.com/FernandoAzve))
- **Vinicius Solon** ([@ViniciusSolon](https://github.com/ViniciusSolon))
- **Vítor Luís da Silva** ([@vitorvls](https://github.com/vitorvls))

---
*Licença MIT.*
