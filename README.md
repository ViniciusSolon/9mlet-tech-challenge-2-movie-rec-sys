# MovieLens 20M — Recommendation System (FIAP Tech Challenge 02)

Sistema de recomendação personalizada com **PyTorch**, **DVC**, **MLflow** e **Docker**, usando o dataset [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

## Status do scaffold

- **Blocos 0–1:** estrutura `src/`, Factory, Strategy, Ruff, pytest, Docker mínimo
- **Etapa de scraping TMDB:** concluída — [docs/GUIA_SCRAPING_E_PIPELINE.md](docs/GUIA_SCRAPING_E_PIPELINE.md) (leigos) · [docs/PRE_ETAPA_METADADOS.md](docs/PRE_ETAPA_METADADOS.md) (comandos)

## Estrutura

```
src/
  domain/       # UserId, MovieId, Rating, RecommendationList
  data/         # preprocessors (Strategy), external/ (TMDB)
  features/     # feature_eng (Bloco 4)
  models/       # Factory + stubs PyTorch/sklearn
  training/     # seeds, loops (Bloco 5)
  evaluation/   # métricas @K (Bloco 5)
  serving/      # inferência (opcional)
configs/        # YAML
data/           # raw / processed (não versionar CSV brutos no Git)
scripts/        # hello_train, fetch TMDB, relatórios de metadados
tests/          # espelha src/
```

## Pré-requisitos

Para executar este projeto, você precisará ter instalado em sua máquina:

1.  **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (recomendado para Windows/macOS) ou **Docker + Docker Compose** (Linux).
2.  **Git** para clonar o repositório.
3.  **Dataset MovieLens 20M**: Devido ao tamanho (600MB+), os arquivos brutos não estão no Git.
    *   Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).
    *   Extraia os arquivos `.csv` na pasta `data/raw/` do projeto. Você deve ter pelo menos:
        *   `data/raw/rating.csv`
        *   `data/raw/movie.csv`
        *   `data/raw/link.csv`

---

## Como Executar (Guia Rápido)

Siga estes passos para treinar o modelo e visualizar os resultados sem precisar configurar um ambiente Python local:

### 1. Clonar e Configurar
```bash
git clone <url-do-repositorio>
cd 9mlet-tech-challenge-2-movie-rec-sys
# Certifique-se de que os dados estão em data/raw/
```

### 2. Primeiro Treinamento (Forçado)
Como os ambientes Docker são isolados, na primeira execução precisamos forçar o "pipeline" de dados a rodar completamente:
```bash
docker-compose run train python -m dvc repro -f
```
*Este comando vai: Limpar dados antigos -> Processar Ratings -> Enriquecer com Metadados -> Treinar o Modelo PyTorch.*

### 3. Subir o ambiente completo
Após o treinamento, inicie os serviços para persistir os resultados e abrir a interface visual:
```bash
docker-compose up
```

### 4. Acompanhar Experimentos (MLflow)
Com os containers rodando, abra o navegador e acesse:
👉 **[http://localhost:5000](http://localhost:5000)**

Lá você encontrará:
*   Métricas de erro (MSE) por época.
*   Parâmetros utilizados (Learning Rate, Batch Size, etc).
*   O modelo treinado pronto para download (`model.pth`).

---

## Desenvolvimento local (Opcional para Devs)

## Plano de execução

Ver [TODO.md](TODO.md), [docs/](docs/) e contexto em [.cursor/context/](.cursor/context/).

## Licença

MIT — ver [LICENSE](LICENSE).
