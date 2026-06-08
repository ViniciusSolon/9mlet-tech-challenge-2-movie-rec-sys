# Etapa 2 — Ambiente e Dependências

> Disciplina 02 — Reprodutibilidade  
> Branch: `feat/step-2`  
> Responsável principal: Vini

---

## O que foi feito

### 2.1 + 2.2 — `pyproject.toml` atualizado com todas as dependências

Arquivo: [`pyproject.toml`](../pyproject.toml)

O `pyproject.toml` foi reestruturado para incluir **todas as dependências necessárias até a Etapa 4**, separadas por propósito:

| Grupo | Pacotes | Para qual etapa |
|-------|---------|-----------------|
| HTTP / I/O | `httpx>=0.27`, `pandas>=2.2`, `pyarrow>=18.0`, `pyyaml>=6.0` | Etapa de scraping + pipeline |
| Config | `pydantic-settings>=2.6` | Etapa 2 |
| Ciência de dados | `numpy>=1.26`, `scipy>=1.13`, `scikit-learn>=1.5` | Etapa 4 — baselines |
| Rede neural | `torch>=2.3` (CPU wheel via índice `pytorch-cpu`) | Etapa 4 — PyTorch |
| Rastreamento | `mlflow>=2.18` | Etapa 3 — experimentos + registry |
| Versionamento de dados | `dvc>=3.56` | Etapa 3 — pipeline DVC |
| Engenharia de features | `bertopic>=0.16`, `sentence-transformers>=3.0` | Etapa 3 — `feature_eng` stage |
| Dev | `pytest`, `pytest-cov`, `pytest-mock`, `ruff`, `pre-commit` | desenvolvimento |
| Opcional `s3` | `dvc[s3]` | Etapa 3 — remote S3 |
| Opcional `gdrive` | `dvc[gdrive]` | Etapa 3 — remote GDrive |

**Índice PyTorch CPU** (`[tool.uv.sources]`):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Isso garante que o Docker e máquinas sem GPU usem a wheel CPU (≈ 250 MB vs ≈ 2 GB). Quem tiver GPU pode sobrescrever localmente com `pip install torch --index-url https://download.pytorch.org/whl/cuXXX`.

---

### 2.3 — Lock file gerado e commitado

Arquivo: [`uv.lock`](../uv.lock)

```bash
uv lock
# Resolved 214 packages
```

O `uv.lock` foi gerado e deve ser commitado junto com o `pyproject.toml`. Ele garante que qualquer colaborador (ou pipeline CI/CD) instale exatamente as mesmas versões.

**Para instalar o ambiente do zero:**

```bash
# Instalar uv (se não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar e sincronizar
git clone <repo>
cd 9mlet-tech-challenge-2-movie-rec-sys
uv sync                   # instala deps de prod + dev
uv sync --extra s3        # + suporte DVC para S3 (opcional)
```

---

### 2.4 — `configs/settings.py` expandido com Pydantic Settings

Arquivo: [`configs/settings.py`](../configs/settings.py)

A classe `Settings` agora cobre todas as variáveis de ambiente do projeto:

| Campo | Env var | Padrão | Descrição |
|-------|---------|--------|-----------|
| `tmdb_api_key` | `TMDB_API_KEY` | `""` | Key da API TMDB |
| `omdb_api_key` | `OMDB_API_KEY` | `""` | Key da API OMDb (fallback) |
| `tmdb_language` | `TMDB_LANGUAGE` | `"en-US"` | Idioma das respostas TMDB |
| `tmdb_min_interval_sec` | `TMDB_MIN_INTERVAL_SEC` | `0.26` | Rate limit (req/s) |
| `tmdb_max_retries` | `TMDB_MAX_RETRIES` | `4` | Tentativas antes de desistir |
| `mlflow_tracking_uri` | `MLFLOW_TRACKING_URI` | `"http://localhost:5000"` | Servidor MLflow |
| `mlflow_experiment_name` | `MLFLOW_EXPERIMENT_NAME` | `"movielens-recommender"` | Experimento padrão |
| `dvc_remote_url` | `DVC_REMOTE_URL` | `"./dvc-storage"` | Remote DVC local/S3 |
| `data_dir` | `DATA_DIR` | `./data` | Raiz de dados |
| `raw_data_path` | `RAW_DATA_PATH` | `./data/raw` | Dados brutos |
| `pythonhashseed` | `PYTHONHASHSEED` | `42` | Seed Python |
| `torch_seed` | `TORCH_SEED` | `42` | Seed PyTorch |
| `numpy_seed` | `NUMPY_SEED` | `42` | Seed NumPy |
| `top_k` | `TOP_K` | `10` | K para métricas @K |
| `implicit_rating_threshold` | `IMPLICIT_RATING_THRESHOLD` | `4.0` | Limiar feedback implícito |

Validação de `top_k > 0` implementada com `@field_validator`.

---

### 2.5 — `scripts/validate_env.py` criado

Arquivo: [`scripts/validate_env.py`](../scripts/validate_env.py)

Script de diagnóstico que valida o ambiente antes de rodar qualquer pipeline:

```bash
python scripts/validate_env.py
```

**Verificações realizadas:**

| Check | O que valida |
|-------|-------------|
| Python version | `>= 3.11` |
| Pacotes obrigatórios | `pandas`, `numpy`, `scipy`, `sklearn`, `torch`, `mlflow`, `dvc`, `httpx`, `pydantic_settings`, `pyarrow`, `bertopic`, `sentence_transformers` |
| Versões mínimas | Confronta com as versões do `pyproject.toml` |
| CUDA (opcional) | Informa se GPU está disponível (não falha se não tiver) |
| Estrutura de diretórios | `data/`, `data/raw/`, `data/processed/`, `data/logs/`, `models/`, `configs/`, `src/` |
| `.env` / `.env.example` | Verifica existência do arquivo de configuração |
| `Settings` carregável | Testa `load_settings()` sem erros |

**Saída esperada (ambiente OK):**

```
=== validate_env.py — environment sanity check ===

  ✓  Python version  3.11
  ✓  import pandas  2.2.x
  ...
  ✓  CUDA (optional)  not available (CPU mode)
  ✓  path data  ...
  ✓  Settings load  OK

  14/14 checks passed.

All checks passed. Environment is ready.
```

---

### 2.6 — `src/training/seeds.py` atualizado

Arquivo: [`src/training/seeds.py`](../src/training/seeds.py)

A função `set_global_seeds(seed)` agora define seeds para **Python stdlib, NumPy e PyTorch**:

```python
from training.seeds import set_global_seeds

set_global_seeds(42)  # chame isso no topo de todo script de treino
```

Implementação com `try/except ImportError` para que o módulo continue funcional mesmo antes de `torch` ou `numpy` estarem instalados (útil em ambientes mínimos).

Para PyTorch, também configura:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

### 2.7 — `.env.example` revisado

Arquivo: [`.env.example`](../.env.example)

Reorganizado com seções comentadas para cada etapa do projeto:

- Seção TMDB / OMDb (scraping)
- Seção MLflow (Etapa 3)
- Seção DVC (Etapa 3)
- Seção data paths
- Seção seeds de reprodutibilidade (Etapa 4)
- Seção model / evaluation (Etapa 4)
- Seção opcional PostgreSQL (comentada)

---

## Como reproduzir do zero (máquina nova)

```bash
# 1. Clonar
git clone <repo-url>
cd 9mlet-tech-challenge-2-movie-rec-sys
git checkout feat/step-2

# 2. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sincronizar dependências (usa uv.lock para versões exatas)
uv sync

# 4. Copiar .env e preencher as chaves
cp .env.example .env
# editar .env com TMDB_API_KEY, etc.

# 5. Validar
python scripts/validate_env.py
```

---

## Decisões técnicas

| Decisão | Justificativa |
|---------|---------------|
| **uv em vez de Poetry** | Mais rápido, compatível com `[project]` PEP 621, lock file nativo |
| **Torch CPU no índice pytorch-cpu** | Imagem Docker ≈ 250 MB (vs 2 GB GPU); GPU users podem sobrescrever |
| **BERTopic na deps de prod** | Será consumido no stage `feature_eng` do DVC (Etapa 3); não é só dev |
| **`dvc[s3]` como extra opcional** | Evita instalar `boto3` desnecessariamente; ativa com `uv sync --extra s3` |
| **Validação de `top_k`** | Previne erro silencioso em métricas @K — falha rápido e cedo |
| **Seeds com try/except** | `seeds.py` importável em qualquer estágio, inclusive sem torch |

---

## Arquivos modificados / criados

| Arquivo | Ação |
|---------|------|
| `pyproject.toml` | Atualizado — deps prod + dev + extras + índice PyTorch |
| `uv.lock` | Criado — lock file com 214 pacotes resolvidos |
| `configs/settings.py` | Atualizado — MLflow, DVC, seeds, model/eval settings |
| `.env.example` | Atualizado — seções organizadas para todas as etapas |
| `scripts/validate_env.py` | Criado — script de diagnóstico do ambiente |
| `src/training/seeds.py` | Atualizado — NumPy + PyTorch seeds |
| `docs/DOCUMENTACAO_ETAPA2.md` | Criado — este documento |
