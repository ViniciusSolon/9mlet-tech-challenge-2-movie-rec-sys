# MovieLens 20M Recommendation System

Sistema de recomendação desenvolvido para o Tech Challenge da FIAP.

O projeto usa o dataset MovieLens 20M como uma analogia de e-commerce: `userId` representa o cliente, `movieId` representa o produto e `rating` representa a interação com aquele item. A partir disso, o pipeline prepara os dados, treina modelos de recomendação, compara com baselines, registra experimentos no MLflow e versiona o fluxo com DVC.

## O que este projeto demonstra

Este repositório foi estruturado para mostrar um fluxo completo de engenharia de software e MLOps:

- organização do código com responsabilidades separadas;
- pipeline reprodutível com DVC;
- treino de modelo neural em PyTorch;
- comparação com baselines de Scikit-Learn;
- rastreamento de métricas, parâmetros e artefatos com MLflow;
- documentação técnica para a entrega do desafio;
- preparação para apresentação em vídeo e deploy na nuvem.

## Como o projeto funciona

O fluxo principal é o seguinte:

1. Os arquivos CSV brutos do MovieLens ficam em `data/raw/`.
2. O stage `preprocess` limpa e normaliza as interações.
3. O stage `enrich_metadata` junta os dados do MovieLens com os metadados do TMDB já versionados no projeto.
4. O stage `feature_eng` monta a base final de treino.
5. O stage `train` treina o modelo PyTorch e salva o checkpoint.
6. O stage `evaluate` compara o modelo com baselines, calcula métricas e registra o campeão no MLflow Registry.

### Snapshot atual do repositório

Os CSVs versionados em `data/raw/` formam um snapshot enxuto de smoke test. Ele mantém o pipeline reproduzível no repositório, mas não representa o MovieLens 20M completo.

Se você quiser executar a versão final com o dataset completo, substitua os CSVs por uma cópia integral do MovieLens 20M e regenere o `dvc.lock` nesse novo contexto.

## Arquitetura

```text
data/raw
  -> scripts/preprocess.py
  -> data/processed/preprocessed_ratings.parquet
  -> scripts/enrich_metadata.py
  -> data/processed/enriched_metadata.parquet
  -> scripts/feature_engineering.py
  -> data/processed/features_ratings.parquet
  -> scripts/train.py
  -> models/model.pth
  -> scripts/evaluate.py
  -> metrics.json + MLflow Registry
```

### Principais pastas

- `src/data/`: leitura, split e preprocessadores.
- `src/models/`: Factory para `MostPopular`, baselines e modelos PyTorch.
- `src/evaluation/`: métricas de ranking e rating, além do Registry.
- `src/training/`: seeds e utilitários de reprodutibilidade.
- `configs/`: settings carregados do `.env`.
- `scripts/`: stages do pipeline e utilitários de validação.
- `tests/`: cobertura de domínio, métricas e componentes centrais.

## Pré-requisitos

O projeto roda em Windows, macOS e Linux. Você precisa de:

- Python 3.11+
- `uv` instalado, ou Docker Desktop se preferir rodar em container
- Git
- DVC CLI para reproduzir o pipeline localmente
- Dados MovieLens 20M em `data/raw/`

A instalação limpa já foi validada em uma venv nova do workspace, com `scripts/validate_env.py` passando em 25/25 checks.

Arquivos esperados em `data/raw/`:

- `ratings.csv` ou `rating.csv`
- `movies.csv` ou `movie.csv`
- `links.csv` ou `link.csv`

Se você quiser regenerar os metadados TMDB, também vai precisar de `TMDB_API_KEY` no `.env`.

## Setup local

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd 9mlet-tech-challenge-2-movie-rec-sys
```

### 2. Instalar dependências

```bash
uv sync
```

### 3. Criar o arquivo `.env`

Use este comando cross-platform para copiar o exemplo:

```bash
python -c "from pathlib import Path; Path('.env').write_text(Path('.env.example').read_text(encoding='utf-8'), encoding='utf-8')"
```

Depois, edite o `.env` apenas se precisar alterar chaves, paths ou integrações externas.

### 4. Validar o ambiente

```bash
uv run python scripts/validate_env.py
```

## Ordem recomendada de execução

Se você estiver começando do zero, siga esta sequência:

1. Coloque os CSVs do MovieLens em `data/raw/`.
2. Rode `uv sync`.
3. Crie o `.env` a partir do `.env.example`.
4. Execute `uv run python scripts/validate_env.py`.
5. Se quiser um smoke test rápido, gere dados sintéticos com `uv run python scripts/create_dummy_data.py`.
6. Rode `uv run dvc repro` para executar o pipeline completo.
7. Rode `uv run python scripts/evaluate.py` se quiser executar a avaliação manualmente.
8. Se estiver usando Docker, suba os serviços de treino e MLflow.

## Pipeline DVC

O pipeline completo está definido em `dvc.yaml` e segue esta ordem:

`preprocess -> enrich_metadata -> feature_eng -> train -> evaluate`

Para reproduzir tudo:

```bash
uv run dvc repro
```

## Docker e MLflow

O projeto também pode ser executado com Docker:

```bash
docker compose run --rm train
docker compose up mlflow
```

O serviço `train` executa o pipeline e o serviço `mlflow` expõe a interface de rastreamento.

Fora do Docker, o projeto usa por padrão um backend SQLite local em `mlflow.db`, o que facilita a execução em máquina limpa sem depender de um servidor externo.

## Execução manual

Se você quiser inspecionar o treino ou repetir etapas específicas, use:

```bash
uv run python scripts/train.py --model-type torch_mlp --epochs 10
uv run python scripts/evaluate.py
```

## Deploy na nuvem

O deploy público no Render ainda será publicado.

Quando o serviço estiver no ar, o link ficará neste formato:

```text
https://<seu-projeto>.onrender.com
```

Depois do deploy, substitua esse placeholder pela URL real.

## Documentação útil

| Doc | O que você encontra |
|-----|---------------------|
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | performance, limitações e vieses |
| [docs/AUDITORIA_DESAFIO.md](docs/AUDITORIA_DESAFIO.md) | checklist vs. enunciado do desafio |
| [docs/IMPLEMENTACAO_MLP_PYTORCH.md](docs/IMPLEMENTACAO_MLP_PYTORCH.md) | detalhes do treino neural |
| [docs/DOCUMENTACAO_ETAPA2.md](docs/DOCUMENTACAO_ETAPA2.md) | dependências, settings e validação |
| [docs/GUIA_SCRAPING_E_PIPELINE.md](docs/GUIA_SCRAPING_E_PIPELINE.md) | etapa TMDB |

## Licença

MIT. Veja [LICENSE](LICENSE).
