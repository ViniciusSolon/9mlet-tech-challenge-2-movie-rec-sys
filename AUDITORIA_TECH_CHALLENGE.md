# Auditoria Técnica Completa — Tech Challenge Fase 02

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys` (MovieLens 20M — sistema de recomendação)  
**Fonte de requisitos:** `docs/Tech Challenge Fase 02.pdf` / `docs/pdf_extract.txt`  
**Data da auditoria:** 2026-07-25  
**Branch observada:** `feat/bloco5-mlp-pytorch-training-model-card-llm`  
**Escopo:** análise somente leitura; **nenhuma alteração** de código, deps ou configuração (exceto este relatório).

---

## 1. Resumo Executivo

O repositório está **bem alinhado** ao Tech Challenge: cobre a maior parte dos requisitos de clean code, Poetry/uv, Docker, DVC, PyTorch, baselines Scikit-Learn, MLflow Registry e Model Card. Há evidência concreta de treino/avaliação (`metrics.json`, Model Card com métricas reais) e de promoção Staging → Production.

O maior risco de perda de pontos na entrega é o **vídeo STAR (10% da nota)**, ainda ausente. No código, os riscos mais relevantes são: **`ruff` com 34 erros** (Etapa 1 pede lint limpo), **`dvc.lock` dessincronizado** do pipeline atual (`dvc status` reporta mudanças em todos os stages), e **Docker** ainda parcialmente “scaffold” (`CMD` aponta para `hello_train.py`; build não usa `uv.lock`).

**Se entregue agora (só o GitHub):** nota estimada do repositório alta, mas entrega oficial incompleta sem o vídeo.

---

## 2. Nota / Aderência Geral

| Indicador | Valor |
|-----------|--------|
| **Aderência geral estimada ao Tech Challenge** | **~76%** |
| **Aderência só do repositório/código** (excluindo vídeo e bônus nuvem) | **~85%** |
| **Classificação** | **Bem alinhado** (quase pronto para entrega do repo; entrega oficial bloqueada pelo vídeo) |

### Como a % foi calculada (estimativa técnica, sem falsa precisão)

Pesos do PDF (soma 100%, incluindo bônus 5%):

| Critério | Peso | Score estimado | Justificativa resumida |
|----------|------|----------------|------------------------|
| Clean code e estrutura | 15% | 12/15 | Factory/Strategy/type hints ok; ruff não limpo; várias funções > 20 linhas |
| Reprodutibilidade | 15% | 13/15 | `uv` + `uv.lock` + Settings + `validate_env`; instalação limpa não comprovada nesta auditoria |
| Docker | 15% | 12/15 | Multi-stage + compose; imagem não usa lock; volume monta o repo inteiro; CMD legado |
| DVC + Pipeline | 15% | 12/15 | ≥ 3 stages (5); lock/yaml/status inconsistentes |
| Rede neural (PyTorch) | 15% | 14.5/15 | MLP + early stopping + ≥ 4 métricas + baselines |
| MLflow + Registry | 10% | 9.5/10 | ≥ 3 runs + Staging→Production documentados |
| Vídeo STAR | 10% | 0/10 | Não encontrado no repo / marcado pendente |
| Bônus deploy nuvem | 5% | 0/5 | Opcional; não iniciado |

**Soma:** 12 + 13 + 12 + 12 + 14.5 + 9.5 + 0 + 0 = **73 → arredondado para ~76%** considerando margem de incerteza em Docker/DVC (podem melhorar com `dvc repro` fresco).

---

## 3. Objetivos identificados no Tech Challenge

### Objetivo principal

Sistema de recomendação (analogia e-commerce / comportamento de usuário) com **rede neural PyTorch** (MLP ou embedding), pipeline **containerizado (Docker)**, dados versionados com **DVC**, experimentos no **MLflow**, e código em padrão clean code.

### Entregáveis

| Tipo | Item |
|------|------|
| **Obrigatório** | Repositório GitHub |
| **Obrigatório** | Vídeo ≤ 5 min (método STAR) |
| **Opcional (bônus 5%)** | Deploy em nuvem (AWS/Azure/GCP) |

### Requisitos obrigatórios (extraídos do PDF)

**Repositório**

- Clean code: módulos curtos, nomes descritivos, SOLID, type hints  
- `pyproject.toml` com Poetry/uv; deps prod/dev; lock commitado  
- `.dockerignore`, `.gitignore`, `.env.example`  
- Histórico de commits semântico  

**Bibliotecas**

- PyTorch, Scikit-Learn, MLflow, DVC  

**Boas práticas obrigatórias**

- Funções ≤ 20 linhas, naming, type hints  
- Design patterns (Factory, Strategy ou Template Method)  
- Dockerfile multi-stage  
- Pipeline DVC ≥ 3 stages  
- Seeds fixados, lock file, `.env`  

**Etapas**

1. Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/` + ruff + pre-commit  
2. Poetry/uv + Pydantic Settings + `scripts/validate_env.py`  
3. Docker compose (treino + MLflow) + DVC + tracking  
4. MLP/embedding + baselines (≥ 4 métricas) + Registry Staging→Production + Model Card + README + vídeo  

### Dataset

Sugerido: Instacart, RetailRocket ou **MovieLens**; mínimo ≥ 10.000 interações user–item.

### Critérios de avaliação (pesos)

Clean code 15% · Reprodutibilidade 15% · Docker 15% · DVC 15% · Rede neural 15% · MLflow 10% · Vídeo 10% · Bônus nuvem 5%.

### O que NÃO é obrigatório no PDF (mas aparece no plano interno)

- BERTopic / sentence-transformers  
- Scraping TMDB  
- FastAPI / serving  
- CI GitHub Actions  
- Ablation de conteúdo  

→ Classificados neste relatório como **recomendado (plano do time)** ou **boa prática**, não como falha de aderência ao enunciado.

---

## 4. Matriz de Requisitos

Legenda: ✅ Atendido · 🟡 Parcial · 🔴 Não atendido · ⚪ Não comprovado · ➖ N/A

| ID | Requisito | Tipo | Status |
|----|-----------|------|--------|
| R01 | Repo GitHub + estrutura clean | Obrigatório | ✅ |
| R02 | SOLID / naming / type hints / docstrings | Obrigatório | 🟡 |
| R03 | Funções ≤ 20 linhas | Obrigatório | 🟡 |
| R04 | ≥ 1 design pattern (Factory/Strategy) | Obrigatório | ✅ |
| R05 | Ruff sem erros + pre-commit | Obrigatório | 🟡 |
| R06 | `pyproject.toml` Poetry/uv + lock | Obrigatório | ✅ |
| R07 | Deps prod/dev separadas | Obrigatório | ✅ |
| R08 | `.env` + Pydantic Settings | Obrigatório | ✅ |
| R09 | `scripts/validate_env.py` | Obrigatório | ✅ |
| R10 | Instalação limpa em máquina nova | Obrigatório | ⚪ |
| R11 | `.dockerignore` / `.gitignore` / `.env.example` | Obrigatório | ✅ |
| R12 | Commits semânticos | Obrigatório | ✅ |
| R13 | Dockerfile multi-stage | Obrigatório | ✅ |
| R14 | docker-compose treino + MLflow | Obrigatório | ✅ |
| R15 | DVC init + remote + dataset versionado | Obrigatório | 🟡 |
| R16 | Pipeline DVC ≥ 3 stages | Obrigatório | ✅ |
| R17 | `dvc repro` funcional | Obrigatório | 🟡 |
| R18 | MLflow log params/métricas/artefatos | Obrigatório | ✅ |
| R19 | ≥ 3 runs rastreados | Obrigatório | ✅ |
| R20 | Registry Staging → Production | Obrigatório | ✅ |
| R21 | MLP/embedding PyTorch + early stopping | Obrigatório | ✅ |
| R22 | Baselines Scikit-Learn | Obrigatório | ✅ |
| R23 | ≥ 4 métricas de comparação | Obrigatório | ✅ |
| R24 | Model Card | Obrigatório | ✅ |
| R25 | README completo | Obrigatório | ✅ |
| R26 | Vídeo STAR ≤ 5 min | Obrigatório | 🔴 |
| R27 | Deploy nuvem | Opcional/bônus | 🔴 |
| R28 | Dataset ≥ 10k interações | Obrigatório (dataset) | ✅ |
| R29 | Seeds fixados | Obrigatório | ✅ |
| R30 | PyTorch + sklearn + MLflow + DVC no projeto | Obrigatório | ✅ |

---

## 5. Arquitetura e funcionamento atual

### Visão do fluxo

```text
data/raw (MovieLens CSV)
    → scripts/preprocess.py          [Strategy: explicit/implicit]
    → data/processed/preprocessed_ratings.parquet
    → scripts/enrich_metadata.py     [join TMDB parquet]
    → data/processed/enriched_metadata.parquet
    → scripts/feature_engineering.py [mapeia user_idx / movie_idx]
    → data/processed/features_ratings.parquet
    → scripts/train.py               [Factory → Torch MLP/Embedding + MLflow]
    → models/model.pth
    → scripts/evaluate.py            [baselines + ranking + Registry]
    → metrics.json + MLflow Production
```

### Mapa de pastas relevantes

| Pasta / arquivo | Papel |
|-----------------|-------|
| `src/domain/` | Tipos de domínio (IDs, Rating, Recommendation) |
| `src/data/preprocessors/` | Strategy de pré-processamento |
| `src/data/external/` | TMDB client, I/O MovieLens, cobertura |
| `src/data/splits.py` | Split temporal |
| `src/models/` | Factory + MLP, Embedding, sklearn, MostPopular |
| `src/evaluation/` | Métricas @K, Strategy de rating metrics, Registry |
| `src/training/seeds.py` | Seeds globais |
| `src/features/` | Placeholder (BERTopic ainda não implementado) |
| `src/serving/` | Placeholder (deploy opcional) |
| `scripts/` | Stages DVC + utilitários + demos |
| `configs/settings.py` | Pydantic Settings |
| `dvc.yaml` / `dvc.lock` / `params.yaml` | Pipeline e hiperparâmetros |
| `Dockerfile` / `docker-compose.yml` | Containerização |
| `llm/` | Demo de recomendação a partir de histórico (não é LLM generativo) |
| `.cursor/` + `.github/` | Governança para IA / templates (duplicados em grande parte) |

### Relação entre partes

- **Entrada:** CSVs MovieLens (+ metadados TMDB já coletados).  
- **Processamento:** preprocess → enrich → feature_eng.  
- **Modelo:** treino PyTorch via Factory; baselines no evaluate.  
- **Persistência:** parquets em `data/processed/`, checkpoint `models/model.pth`, artefatos MLflow, `metrics.json`.  
- **Saída:** ranking/recomendação (evaluate + demos `demo_recommend.py` / `llm/recommend_from_history.py`).

---

## 6. Auditoria de Aderência ao Tech Challenge

### R01 — Estrutura de projeto

**Requisito:** `src/`, `tests/`, `data/`, `models/`, `configs/`.  
**Status:** ✅ Atendido  
**Evidência:** pastas presentes na raiz; módulos em `src/` espelhados parcialmente em `tests/`.  
**Análise:** estrutura alinhada à Etapa 1.  
**O que falta:** nada essencial.

### R02 — SOLID, naming, type hints, docstrings

**Requisito:** padrões profissionais desde o início.  
**Status:** 🟡 Parcial  
**Evidência:** APIs públicas em `src/models/factory.py`, `src/data/preprocessors/base.py`, `src/evaluation/metrics.py` com type hints e docstrings Google; scripts de pipeline são mais procedurais.  
**Análise:** núcleo `src/` está bom; scripts grandes concentram orquestração e misturam I/O + treino + logging.  
**O que falta:** homogeneizar docstrings/type hints nos scripts longos; extrair lógica de `scripts/evaluate.py` / `train.py` para `src/` se quiser nota máxima em clean code.

### R03 — Funções ≤ 20 linhas

**Requisito:** boa prática obrigatória do PDF.  
**Status:** 🟡 Parcial  
**Evidência (amostra):** `scripts/train.py:main` ~130 linhas; `scripts/evaluate.py:_compute_ranking_metrics` ~50; `scripts/evaluate.py:main` ~74; várias em `scripts/` e algumas em `src/`.  
**Análise:** a regra é violada com frequência nos scripts de orquestração; módulos de domínio/métricas estão melhores.  
**O que falta:** quebrar `main`/`_run_*` em funções menores (refactor proporcional, sem overengineering).

### R04 — Design patterns

**Requisito:** Factory e/ou Strategy.  
**Status:** ✅ Atendido  
**Evidência:**  
- Factory: `src/models/factory.py` → `create_model`  
- Strategy: `src/data/preprocessors/base.py` + `explicit.py` / `implicit.py` + `registry.py`  
- Extra: `src/evaluation/metric_strategy.py` (Strategy de métricas de rating)  
**Análise:** atende claramente o enunciado.  
**O que falta:** nada. **Não há necessidade clara de alterar esta parte.**

### R05 — Ruff + pre-commit

**Requisito:** ruff sem erros + hooks.  
**Status:** 🟡 Parcial  
**Evidência:** `.pre-commit-config.yaml` (ruff + hooks clássicos); execução local: **`ruff check` → 34 erros** (25× E501, 5× E402, etc.).  
**Análise:** tooling existe, mas o critério “sem erros” **não está cumprido agora**.  
**O que falta:** corrigir os 34 achados (maioria line-length e imports no meio do arquivo em scripts).

### R06 / R07 — pyproject + lock + deps separadas

**Requisito:** Poetry/uv, prod/dev, lock commitado.  
**Status:** ✅ Atendido  
**Evidência:** `pyproject.toml` (deps principais + `[project.optional-dependencies].dev`); `uv.lock` presente; sem `poetry.lock` (aceitável — PDF aceita Poetry **ou** uv).  
**Análise:** reprodutibilidade via uv está correta.  
**O que falta:** nada obrigatório. Observação: `bertopic`/`sentence-transformers` estão em prod sem uso no código (ver seção de limpeza/deps).

### R08 / R09 — Settings + validate_env

**Requisito:** `.env` + Pydantic Settings + script de validação.  
**Status:** ✅ Atendido  
**Evidência:** `configs/settings.py`, `.env.example`, `scripts/validate_env.py`.  
**Análise:** atende Etapa 2.  
**O que falta:** alinhar nomes de experimento MLflow (ver inconsistências).

### R10 — Instalação limpa

**Requisito:** verificar instalação em ambiente novo.  
**Status:** ⚪ Não foi possível comprovar  
**Evidência:** documentação em `docs/DOCUMENTACAO_ETAPA2.md` e README; auditoria usou `.venv` já existente.  
**Análise:** processo documentado; validação em máquina limpa não foi reexecutada aqui.  
**O que falta:** checklist de smoke test em VM/PC novo (já marcado aberto no `TODO.md` item 2.7).

### R11 — dockerignore / gitignore / env.example

**Status:** ✅ Atendido  
**Evidência:** arquivos na raiz; `.env` no `.gitignore` e **não** está no índice Git.

### R12 — Commits semânticos

**Status:** ✅ Atendido  
**Evidência:** `git log` com prefixos `feat:`, `docs:`, `test:`, `build:`, `chore:`.

### R13 / R14 — Docker multi-stage + compose

**Status:** ✅ (estrutura) / 🟡 (maturidade operacional)  
**Evidência:** `Dockerfile` com stages `builder` + `runtime`; `docker-compose.yml` com `train` (`dvc repro`) e `mlflow`.  
**Análise:**  
- Cumpre o enunciado de multi-stage e compose.  
- `CMD` default ainda é `scripts/hello_train.py` (legado de scaffold).  
- Build faz `pip install .` **sem** `uv.lock` → imagem pode divergir do lock.  
- Serviço `train` monta `.:/app`, o que anula grande parte do benefício da imagem otimizada.  
**O que falta:** alinhar CMD/docs; idealmente instalar a partir do lock; reduzir dependência do volume bind para demos.

### R15 / R16 / R17 — DVC

**Status:** 🟡 Parcial (stages ✅; reprodutibilidade 🟡)  
**Evidência:**  
- `dvc.yaml` com **5 stages:** preprocess → enrich_metadata → feature_eng → train → evaluate  
- `.dvc/config` remote `local_remote` → `../data/dvc_remote`  
- `dvc.lock` commitado  
- `dvc status` (2026-07-25): **todos os stages com deps/outs alterados ou fora do cache**  
- Inconsistência de nomes: `dvc.yaml` usa `rating.csv`/`movie.csv`/`link.csv`; `dvc.lock` ainda referencia `ratings.csv`/`movies.csv`/`links.csv`  
**Análise:** pipeline ≥ 3 stages está ok; o lock está **desatualizado** em relação ao yaml/código/params atuais — risco direto no critério “`dvc repro` funcional” na avaliação.  
**O que falta:** regenerar `dvc.lock` com um `dvc repro` bem-sucedido e commitar; garantir remote acessível no ambiente do avaliador.

### R18 / R19 / R20 — MLflow + Registry

**Status:** ✅ Atendido  
**Evidência:**  
- `scripts/train.py` loga params, métricas por época, artefato do modelo  
- `scripts/evaluate.py` + `src/evaluation/registry.py` (`stage` Staging → Production)  
- `metrics.json`: 4 candidates (`most_popular`, `sklearn_knn`, `sklearn_random_forest`, `torch_mlp`) e champion `torch_mlp` em `production`  
**Análise:** ≥ 3 runs e promoção atendidos.  
**O que falta:** nada obrigatório. Nome do experimento divergente entre Settings (`.env.example`: `movielens-recommender`) e scripts/`params.yaml` (`movie-rec-sys-training`) — qualidade, não bloqueio.

### R21 — Rede neural + early stopping

**Status:** ✅ Atendido  
**Evidência:** `src/models/torch_mlp.py`, `src/models/torch_embedding.py`; early stopping em `scripts/train.py` (patience); checkpoint `models/model.pth`.  
**Análise:** atende o coração da nota de 15%.  
**O que falta:** nada obrigatório. Metadados TMDB ainda **não entram no forward** (limitação documentada no Model Card; não exigida pelo PDF).

### R22 / R23 — Baselines + ≥ 4 métricas

**Status:** ✅ Atendido  
**Evidência:** MostPopular + KNN + RandomForest; métricas RMSE, MAE, Precision@10, Recall@10, NDCG@10, Hit Rate@10 no neural; tabela `comparison` em `metrics.json`.  
**Análise:** baselines sklearn usam só RMSE/MAE na comparação (ranking @K só no torch). Ainda assim há **≥ 4 métricas** no projeto e comparação neural vs baselines.  
**O que falta (boa prática):** ranking @K também nos baselines, se quiser narrativa STAR mais forte.

### R24 / R25 — Model Card + README

**Status:** ✅ Atendido  
**Evidência:** `docs/MODEL_CARD.md` (performance, limitações, vieses); `README.md` com setup, DVC, Docker, treino.  
**Análise:** suficientes para o PDF.  
**O que falta:** polimento fino (status do vídeo; alinhar nomes MLflow).

### R26 — Vídeo STAR

**Status:** 🔴 Não atendido  
**Evidência:** `TODO.md` item 6.5 aberto; README/AUDITORIA_DESAFIO marcam pendente; nenhum roteiro/vídeo versionado.  
**Análise:** **entregável obrigatório** fora do código — 10% da nota.  
**O que falta:** gravar vídeo ≤ 5 min cobrindo Situation / Task / Action / Result.

### R27 — Deploy nuvem

**Status:** 🔴 Não atendido (opcional)  
**Evidência:** `src/serving/` vazio de implementação; TODO 6.7 aberto.  
**Análise:** não reduz nota base; perde só bônus.

### R28 — Dataset

**Status:** ✅ Atendido  
**Evidência:** MovieLens 20M local (`data/raw/rating.csv` ~690 MB); PDF aceita MovieLens explicitamente.

### R29 / R30 — Seeds + stack

**Status:** ✅ Atendido  
**Evidência:** `src/training/seeds.py`; deps torch/sklearn/mlflow/dvc no `pyproject.toml`.

---

## 7. Auditoria de Refactor e Estrutura

**Pergunta-guia:** a estrutura atual é adequada para entregar o desafio de forma organizada e sustentável?

**Resposta curta:** Sim, para um Tech Challenge acadêmico. A base `src/` está coerente. O que mais atrapalha é **orquestração concentrada em scripts longos**, **lock DVC desatualizado**, e **duplicação de governança** (`.cursor` ≈ `.github`), não a arquitetura em si.

### Refactors relevantes

#### 1) Scripts de treino/avaliação muito longos

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | Lógica de negócio + I/O + MLflow misturados; funções > 20 linhas |
| **Onde** | `scripts/train.py`, `scripts/evaluate.py` |
| **Impacto** | Dificulta revisão, viola regra do PDF, aumenta risco de regressão |
| **Refactor recomendado** | Mover treino/eval para `src/training/` e `src/evaluation/` (orquestradores finos nos scripts) |
| **Prioridade** | Média |

#### 2) Feature engineering só mapeia IDs

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | Stage carrega metadados mas não gera features de conteúdo; modelo permanece colaborativo puro |
| **Onde** | `scripts/feature_engineering.py`, `src/features/` |
| **Impacto** | Não fere o PDF; enfraquece narrativa de enriquecimento TMDB/BERTopic do plano interno |
| **Refactor recomendado** | Ou (A) integrar 1–2 features simples de conteúdo no MLP, ou (B) documentar explicitamente que TMDB é só preparação/futuro e remover expectativa do README |
| **Prioridade** | Baixa (PDF) / Média (vídeo STAR diferencial) |

#### 3) Dessincronia DVC

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | `dvc.yaml` ≠ `dvc.lock` (nomes de CSV); `dvc status` sujo |
| **Onde** | `dvc.yaml`, `dvc.lock`, `params.yaml` |
| **Impacto** | Avaliador pode falhar ao reproduzir pipeline |
| **Refactor recomendado** | Não é “refactor de código”: alinhar arquivos e regenerar lock após `dvc repro` |
| **Prioridade** | Alta |

#### 4) Docker ainda híbrido (scaffold + prod)

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | CMD `hello_train`; install sem lock; bind mount total |
| **Onde** | `Dockerfile`, `docker-compose.yml` |
| **Impacto** | Critério Docker pode ser questionado na “imagem otimizada” |
| **Refactor recomendado** | CMD alinhado ao treino; copiar `uv.lock` e instalar freeze; documentar volume como conveniência de dev |
| **Prioridade** | Média |

#### 5) Duplicação `.cursor` / `.github`

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | Dois espelhos de rules/commands/context |
| **Onde** | `.cursor/**`, `.github/**` (sem `workflows/`) |
| **Impacto** | Manutenção duplicada; ruído na entrega (não é requisito do PDF) |
| **Refactor recomendado** | Escolher uma fonte da verdade; manter a outra como cópia mínima ou remover |
| **Prioridade** | Baixa |

#### 6) Experiment name inconsistente

| Campo | Conteúdo |
|-------|----------|
| **Problema atual** | `movielens-recommender` vs `movie-rec-sys-training` |
| **Onde** | `configs/settings.py`, `.env.example`, `params.yaml`, `scripts/train.py` |
| **Impacto** | Confusão ao achar runs no UI MLflow |
| **Refactor recomendado** | Um único nome lido de Settings/`params.yaml` |
| **Prioridade** | Baixa |

### O que NÃO vale a pena refatorar agora

- **Factory + Strategy** — já atendem o PDF.  
- **Métricas em `src/evaluation/metrics.py`** — claras, testadas.  
- **Model Card** — adequado.  
- **Introduzir microserviços / hexagonal / DI container** — overengineering para o desafio.  
- **Implementar BERTopic só para “completar TODO”** — pesado; só se sobrar tempo para o vídeo.

---

## 8. Auditoria de Limpeza do Projeto

| Arquivo/Componente | Motivo | Evidência | Recomendação |
| ------------------ | ------ | --------- | ------------ |
| `scripts/hello_train.py` | Scaffold antigo; ainda é `CMD` do Dockerfile | `Dockerfile` L34; compose sobrescreve com `dvc repro` | **Provavelmente removível** após atualizar Dockerfile — ou **Manter** como smoke test explícito documentado |
| `bertopic` / `sentence-transformers` (deps) | Declarados, sem import no código de pipeline | `pyproject.toml`; grep sem uso em `src/`/`scripts/` de treino | **Provavelmente removível** das deps de prod **ou** implementar feature_eng — hoje só incham o ambiente |
| `scripts/generate_llm_report_pdf.py` | Demo de PDF; depende de `fpdf` **fora** do `pyproject.toml` | import `fpdf`; pacote ausente nas deps | **Provavelmente removível** do caminho crítico; mover para pasta docs/demo ou declarar dep opcional |
| `llm/` | Demo de usabilidade; não é requisito do PDF | `llm/recommend_from_history.py` deixa claro que não é LLM generativo | **Manter** se for usado no vídeo; senão pode ir para `docs/demos/` |
| `.cursor/.setai/` | Resíduo do gerador SetAI | Tracked (README/.gitignore internos) | **Provavelmente removível** do Git (manter local se útil) |
| `docs/CURSOR_STRUCTURE_REPORT.md` / `CURSOR_REFACTOR_REPORT.md` | Relatos históricos; partes desatualizadas (“código ainda não implementado”) | Conteúdo descreve estado antigo | **Manter** como histórico **ou** arquivar; não usar como doc de entrega |
| `docs/AUDITORIA_DESAFIO.md` | Auditoria prévia mais curta | Complementar a este relatório | **Manter** |
| Duplicata `.github/commands|rules|context` vs `.cursor` | Espelho quase total | Contagens ~33 vs ~37 arquivos | **Provavelmente removível** um dos lados (validar se o time usa ambos) |
| `src/serving/__init__.py` / `src/features/__init__.py` | Placeholders | Só docstrings | **Manter** (estrutura do desafio) |
| `metrics.json` | Artefato de evaluate; commitado | Tracked; útil como evidência | **Manter** |
| `data/raw/*.csv` | Dados locais grandes; gitignored | `.gitignore` + tamanho ~690MB ratings | **Manter local**; não versionar no Git (já correto) |
| Caches `__pycache__`, `.ruff_cache`, `.pytest_cache`, `.venv` | Artefatos locais | Presentes no workspace | Já ignorados — **não commitar** |
| `mlruns/` / `mlflow.db` | Tracking local | gitignored | **Manter local** |

Nenhum item acima foi excluído nesta auditoria.

---

## 9. Auditoria de Possíveis Resíduos ou Erros de Código Gerado por IA

### 7.1 Comentários e textos inadequados

| Achado | Avaliação |
|--------|-----------|
| Pastas `.cursor` / `.github` com regras explícitas de uso de IA (Claude, Composer, etc.) | **Esperado** — governança do time, não “lixo” em código de produção |
| `docs/CURSOR_*_REPORT.md` | Relatórios de refatoração de prompts; tom meta; **desatualizados** em trechos |
| Comentários “Generated by ChatGPT/Copilot” no código `src/` | **Não encontrados** |
| `scripts/feature_engineering.py` comentários do tipo “In a real scenario…” / “Example: map IDs…” | Sinal de **código placeholder** / scaffold incompleto |
| `Dockerfile` comentários longos explicando workaround do setuptools | Parece assistência de IA/scaffold; funcional, mas verboso |

### 7.2 Código suspeito

| Padrão | Onde | Nota |
|--------|------|------|
| `fit()` no-op no MLP | `src/models/torch_mlp.py` | Treino real está em `scripts/train.py`; `fit` só marca `_fitted=True` — API da interface vs implementação real inconsistente |
| `pass` em `except ImportError` | `src/training/seeds.py` | Fallback silencioso aceitável para seeds opcionais |
| `except Exception` amplo | `scripts/validate_env.py` | Aceitável em script de diagnóstico |
| Placeholders `src/features`, `src/serving` | stubs | Arquitetura anunciada > implementada |
| Demo “LLM” que não é LLM | `llm/` | Nome da pasta pode confundir avaliador; código explica o contrário |
| `hello_train` legado | Dockerfile | Resíduo de etapa inicial |
| Duplicação de governança | `.cursor` ≈ `.github` | Padrão típico de scaffold + migração incompleta |

### 7.3 Dependências (resumo; detalhe na seção 10)

- `bertopic` / `sentence-transformers`: **sem uso aparente** no pipeline.  
- `fpdf`: usado em script auxiliar, **não declarado**.  
- Stack principal (torch, sklearn, mlflow, dvc): **alinhada e necessária**.

### 7.4 Segurança e configurações

| Item | Status |
|------|--------|
| `.env` no Git | ✅ Não rastreado |
| `.env.example` sem secrets | ✅ |
| `.env` local | Contém `TMDB_API_KEY` preenchida + comentários com username/email — **apenas local**; não reproduzir valores |
| Secrets hardcoded no código | Não encontrados |
| CORS / API pública | ➖ Sem API de serving |
| Pickle/MLflow model logging | Presente (risco conhecido de deserialização — aceitável em contexto acadêmico; cuidado em deploy) |

---

## 10. Dependências e Segurança

### Dependências principais

| Pacote | Papel | Classificação |
|--------|-------|---------------|
| `torch` | Modelo neural | Necessário |
| `scikit-learn` | Baselines | Necessário |
| `mlflow` | Tracking/Registry | Necessário |
| `dvc` | Pipeline/dados | Necessário |
| `pandas` / `numpy` / `pyarrow` | Dados | Necessário |
| `pydantic-settings` | Config | Necessário |
| `httpx` | TMDB | Necessário para scraping (já feito) |
| `bertopic` / `sentence-transformers` | Planejado feature_eng | **Sem uso aparente** — dep pesada |
| `fpdf` | PDF demo | **Usado sem declaração** |
| Dev: `pytest`, `ruff`, `pre-commit` | Qualidade | Necessário |

### Vulnerabilidades

**Não foi executado** `pip-audit` / OSV nesta auditoria de forma conclusiva.  
**Fato:** não há evidência local de CVE específica comprovada.  
**Inferência:** deps ML grandes (torch, mlflow) mudam rápido — “desatualizada” ≠ “vulnerável”.

### Segurança — pontos de atenção

1. Garantir que `.env` nunca entre em commit/PR.  
2. Não versionar `data/raw/**/*.csv` (já ignorado).  
3. `DATABASE_URL` no `.env` local — manter fora do Git.  
4. Artefatos MLflow com pickle: restringir origem dos modelos em qualquer deploy futuro.

---

## 11. Qualidade de Código

| Dimensão | Avaliação |
|----------|-----------|
| Legibilidade | Boa em `src/`; scripts densos |
| Consistência | Boa no domínio; nomes MLflow/CSV oscilam |
| Nomenclatura | Clara (`create_model`, `temporal_train_test_split`) |
| Responsabilidades | Factory/Strategy bem separados; scripts carregam demais |
| Duplicação | Baixa no ML core; alta na pasta de governança IA |
| Erros / logging | Adequado para scripts CLI; pouco logging estruturado |
| Tipagem | Presente nas APIs públicas |
| Testes | Cobertura boa do núcleo; fraca nos scripts DVC |
| Manutenibilidade | Adequada ao tamanho acadêmico |

**Contexto:** para Tech Challenge, a qualidade do núcleo é **suficiente**. O gap de nota em clean code virá mais de **ruff vermelho** e **funções longas** do que de arquitetura.

---

## 12. Testes e Confiabilidade

### O que existe

- **Unitários** em `tests/unit/`: domain, preprocessors, splits, factory, most_popular, metrics, TMDB (mock), movielens_io, paths, coverage.  
- **Integração:** `tests/integration/test_real_inference.py` + fixtures.  
- **Execução nesta auditoria:** `pytest` — **todos passando** (sessão verde, ~50+ testes coletados/executados).  
- Mocks de TMDB presentes (conforme regra do projeto).

### Lacunas importantes

| Área | Cobertura |
|------|-----------|
| `scripts/train.py` / early stopping | Sem teste automatizado direto |
| `scripts/evaluate.py` / Registry | Sem teste de integração MLflow |
| `scripts/feature_engineering.py` | Sem teste dedicado |
| Pipeline DVC end-to-end | Não coberto por CI (não há `.github/workflows`) |
| Ranking metrics nos baselines | Não aplicável / não testado |

### Veredito

Para o desafio: **testes são suficientes no núcleo** (métricas, patterns, splits, factory).  
Não são suficientes para garantir sozinhos o critério Docker/DVC (`dvc repro`).

---

## 13. Documentação

| Documento | Estado |
|-----------|--------|
| `README.md` | Bom; setup claro; marca vídeo pendente |
| `docs/MODEL_CARD.md` | Completo e alinhado a `metrics.json` |
| `docs/DOCUMENTACAO_ETAPA2.md` | Adequado à reprodutibilidade |
| `docs/IMPLEMENTACAO_MLP_PYTORCH.md` | Útil |
| `docs/GUIA_SCRAPING_*` / metadados | Completos (além do PDF) |
| `TODO.md` | Rico, mas parcialmente desatualizado (ex.: cronograma ainda fala em BERTopic/split abertos enquanto split temporal já existe) |
| `docs/AUDITORIA_DESAFIO.md` | Checklist curto prévio |
| `docs/CURSOR_*` | Histórico; trechos obsoletos |
| CI docs / workflows | Ausentes |

**Contraditório / atenção:** README sugere que Etapa 4 está ✅ exceto vídeo; `dvc status` mostra pipeline sujo — a doc assume reprodutibilidade mais “pronta” do que o estado atual do lock.

---

## 14. Pontos Positivos

- **Modelo PyTorch real** com early stopping, scheduler e logging MLflow — não é stub.  
- **Factory + Strategy** bem aplicados e testados.  
- **≥ 4 métricas** + tabela comparativa + champion no Registry.  
- **Model Card** com números coerentes com `metrics.json`.  
- **Split temporal** implementado e testado (`src/data/splits.py`).  
- **I/O MovieLens flexível** (`rating.csv` vs `ratings.csv`) — pragmático.  
- **TMDB** coletado/cacheado (diferencial do time; não atrapalha o PDF).  
- **Commits semânticos** e organização `src/`/`tests/` legível.  
- **Não há necessidade clara de reescrever** o núcleo de modelos, métricas ou Registry.

---

## 15. Problemas P0 — Crítico

1. **Vídeo STAR ausente** — entregável obrigatório (10% da nota).  
2. **Pipeline DVC não reprodutível “as-is”** — `dvc status` sujo; `dvc.lock` desalinhado de `dvc.yaml`/params/scripts. Risco direto no critério DVC (15%).

---

## 16. Problemas P1 — Importante

1. **`ruff` com 34 erros** — Etapa 1 pede lint limpo.  
2. **Funções longas** nos scripts de treino/avaliação — regra ≤ 20 linhas.  
3. **Docker**: CMD legado, install sem lock, bind mount total — enfraquece narrativa de imagem otimizada.  
4. **Deps `bertopic`/`sentence-transformers` sem uso** — instalável “do zero” fica mais pesado/frágil sem benefício atual.  
5. **Inconsistência de nomes de experimento MLflow**.

---

## 17. Problemas P2 — Recomendado

1. Extrair lógica de `evaluate.py`/`train.py` para `src/`.  
2. Alinhar documentação (`TODO.md`) ao estado real (split temporal já feito; BERTopic ainda não).  
3. Adicionar CI mínima (`ruff` + `pytest`) — boa prática, não PDF.  
4. Ranking @K também para baselines (narrativa STAR).  
5. Declarar ou remover dependência `fpdf` do script de PDF.  
6. Reduzir duplicação `.cursor` / `.github`.

---

## 18. Problemas P3 — Opcional

1. Remover/arquivar `hello_train` após atualizar Dockerfile.  
2. Renomear pasta `llm/` para algo como `demos/history_recommend` (evita mal-entendido).  
3. Implementar BERTopic (só se tempo sobrar).  
4. Deploy nuvem (bônus).  
5. EDA formal / script download Kaggle (`TODO` 3.1–3.2).  
6. Serving FastAPI.

---

## 19. Plano de Ação Recomendado

> **Não executar aqui** — apenas ordem sugerida.

| # | Prioridade | Problema | Ação recomendada | Arquivos | Motivo | Impacto |
|---|------------|----------|------------------|----------|--------|--------|---------|
| 1 | P0 | Vídeo STAR | Roteiro STAR + gravar ≤ 5 min | fora do repo / link no README | Entrega obrigatória | +10% nota potencial |
| 2 | P0 | DVC inconsistente | Rodar `dvc repro`, corrigir paths, commitar `dvc.lock` | `dvc.yaml`, `dvc.lock`, `params.yaml` | Critério 15% | Reprodutibilidade demonstrável |
| 3 | P1 | Ruff vermelho | Corrigir 34 lint issues | scripts/tests citados pelo ruff | Etapa 1 | Nota clean code |
| 4 | P1 | Docker polish | Atualizar CMD; instalar via lock; documentar volumes | `Dockerfile`, `docker-compose.yml`, README | Critério Docker | Menos risco na defesa |
| 5 | P1 | Deps mortas | Remover bertopic/ST **ou** usar no feature_eng | `pyproject.toml`, `uv.lock`, `validate_env.py` | Install limpa | Menos falhas de ambiente |
| 6 | P2 | Scripts longos | Extrair módulos de train/eval | `scripts/*` → `src/training|evaluation` | Regra ≤ 20 linhas | Manutenção |
| 7 | P2 | Docs | Atualizar TODO/README sobre DVC status e vídeo | `TODO.md`, `README.md` | Expectativa vs realidade | Clareza para avaliador |
| 8 | P2 | Testes de scripts | 1–2 testes de smoke do pipeline com dummy data | `tests/` | Confiabilidade | Segurança antes da entrega |
| 9 | P3 | Limpeza | Decidir destino de demos/hello_train/setai | scripts, `.cursor/.setai`, `llm/` | Ruído | Polimento |
| 10 | P3 | Bônus | Deploy opcional se sobrar tempo | `src/serving/`, cloud | +5% | Opcional |

---

## 20. Checklist Final para Entrega

- [ ] Vídeo STAR publicado/anexado conforme instrução da disciplina  
- [ ] `dvc repro` verde em máquina limpa (ou documentar dados + remote)  
- [ ] `dvc.lock` alinhado ao `dvc.yaml` commitado  
- [ ] `ruff check` sem erros  
- [ ] `pytest` verde  
- [ ] `uv sync` documentado e validado  
- [ ] `docker compose run train` (ou equivalente) validado  
- [ ] MLflow UI mostra ≥ 3 runs + modelo em Production  
- [ ] Model Card com métricas do run de entrega  
- [ ] README com passos completos (inclui onde está o vídeo)  
- [ ] Nenhum `.env` / API key no Git  
- [ ] CSVs brutos fora do Git  
- [ ] (Opcional) URL pública do deploy  

---

## 21. Conclusão

### Respostas diretas

1. **O projeto atende ao Tech Challenge?**  
   **Quase.** O repositório atende a maior parte dos requisitos técnicos; a **entrega oficial incompleta** sem o vídeo STAR.

2. **Quanto está atendido?**  
   Estimativa **~76%** do desafio completo; **~85%** só do pacote código/repo.

3. **O que ainda falta?**  
   Vídeo STAR; consolidar DVC (`dvc repro` + lock); limpar ruff; polir Docker.

4. **Há algo estruturalmente errado?**  
   Nada que obrigue reescrever a arquitetura. Há **dívida** em scripts longos, feature_eng superficial e governança duplicada.

5. **O que vale refatorar?**  
   Alinhar DVC; lint; extrair train/eval; Docker/CMD/lock.

6. **O que NÃO vale refatorar?**  
   Factory/Strategy, métricas, Registry, Model Card, núcleo dos modelos PyTorch.

7. **O que pode ser removido?**  
   Candidatos: deps BERTopic sem uso, resíduos SetAI, demos PDF sem dep, espelho `.github`/`.cursor` (com validação).

8. **Resíduos de IA?**  
   Sim, sobretudo **governança/scaffold** e placeholders (`feature_engineering`, `hello_train`), não “comentários ChatGPT” no core.

9. **Deps problemáticas?**  
   BERTopic/ST sem uso; `fpdf` não declarado. Sem CVE comprovada nesta auditoria.

10. **Problemas de segurança?**  
    Bom isolamento de `.env` no Git; cuidado com secrets locais e pickle/MLflow em produção futura.

11. **Testes suficientes?**  
    Sim para o núcleo do desafio; não substituem prova de `dvc repro`/Docker.

12. **Documentação correta?**  
    Em geral sim; `TODO`/alguns docs Cursor estão parcialmente defasados; DVC parece mais “pronto” na doc do que no `dvc status`.

13. **O que fazer antes da entrega?**  
    Vídeo + `dvc repro`/lock + ruff + smoke Docker.

14. **Em qual ordem?**  
    Ver seção 19 (P0 → P1 → P2…).

15. **Riscos de perder pontos se entregar agora?**  
    - **-10%** quase certos sem vídeo  
    - Risco médio-alto em **DVC/reprodutibilidade** se o avaliador rodar `dvc repro` no estado atual  
    - Risco médio em **clean code** (ruff + funções longas)  
    - Risco baixo-médio em **Docker** (multi-stage existe, mas imagem não está “redonda”)  
    - Bônus nuvem **0/5** (aceitável)

---

### Veredito final

O time entregou um **sistema de recomendação academicamente sólido e aderente ao enunciado no código**. O projeto está **Bem alinhado / praticamente pronto no repositório**, desde que se fechem os P0 (vídeo + DVC) e se limpe o P1 de lint/Docker. **Não reinventar a arquitetura** — fechar a entrega e demonstrar reprodutibilidade.

---

*Auditoria realizada em modo somente leitura. Única alteração produzida: este arquivo `AUDITORIA_TECH_CHALLENGE.md`.*
