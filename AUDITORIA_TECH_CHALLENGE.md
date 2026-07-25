# Auditoria Técnica Completa — Tech Challenge Fase 02

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys` (sistema de recomendação MovieLens 20M)  
**Fonte de requisitos:** `docs/Tech Challenge Fase 02.pdf` (texto extraído em `docs/pdf_extract.txt`)  
**Data da auditoria:** 2026-07-25  
**Branch analisada:** `feat/bloco5-mlp-pytorch-training-model-card-llm` (`6d1f093`)  
**Escopo:** análise somente leitura do repositório + execução de checagens não destrutivas (`pytest` unitário, `ruff check`).  
**Regra seguida:** nenhum código, dependência, configuração ou estado de dados foi alterado; este arquivo Markdown é a única entrega desta tarefa.

---

## 1. Resumo Executivo

O repositório **já cobre a maior parte dos requisitos técnicos obrigatórios do PDF**: estrutura `src/`/`tests/`/`configs/`, Factory + Strategy, `pyproject.toml` + `uv.lock`, `.env` + Pydantic Settings, `validate_env.py`, Dockerfile multi-stage, `docker-compose` (treino + MLflow), pipeline DVC com 5 stages, MLP PyTorch com early stopping, baselines Scikit-Learn, ≥ 4 métricas, tracking MLflow, Model Card e README.

Os maiores riscos de nota na entrega atual são:

1. **Vídeo STAR (10% da nota) — não encontrado no repositório** (pendência explícita no `TODO.md`).
2. **`dvc.lock` / dados versionados ainda refletem um snapshot de smoke test** — os CSVs checked-in são pequenos e não representam o MovieLens 20M completo, o que limita a narrativa de reprodutibilidade do dataset final.
3. **Instalação limpa ainda não foi provada em um ambiente realmente zerado** — a validação rodou na `.venv` do workspace, mas não em uma máquina fresca.
4. **Clean code estrutural ainda tem margem de melhoria** — os scripts de orquestração continuam mais longos do que o ideal, embora o lint agora esteja limpo.

**Veredito curto:** o projeto está **bem alinhado** ao desafio no código e na documentação alinhada ao run atual; o principal gap obrigatório que ainda falta é o **vídeo STAR**.

---

## 2. Nota / Aderência Geral

### Estimativa técnica de aderência: **~80%**

Esta porcentagem **não é uma nota oficial da FIAP**. É uma estimativa ponderada pelos critérios do PDF (sem contar o bônus de nuvem como obrigatório).

| Critério (PDF) | Peso | Estimativa de atendimento | Contribuição |
|----------------|------|---------------------------|--------------|
| Clean code e estrutura | 15% | ~70% | 10,5 |
| Reprodutibilidade | 15% | ~88% | 13,2 |
| Docker | 15% | ~85% | 12,8 |
| DVC + Pipeline | 15% | ~82% | 12,3 |
| Rede neural (PyTorch) | 15% | ~88% | 13,2 |
| MLflow + Registry | 10% | ~90% | 9,0 |
| Vídeo STAR | 10% | ~0% | 0,0 |
| **Subtotal obrigatório** | **95%** | — | **~74 / 95 ≈ 78%** |
| Bônus cloud (opcional) | 5% | ~0% | 0 |

**Ajuste qualitativo (+2 pp):** documentação técnica rica, testes unitários passando (52), Model Card sincronizada, pipeline completo em código → **~80%**.

### Classificação

**Bem alinhado** (próximo de “muito bem alinhado” no código; falta principalmente o vídeo STAR para fechar a entrega).

**Principais motivos:**

- Quase todos os módulos exigidos pelo PDF existem e se conectam no fluxo DVC.
- O gap obrigatório mais claro é o **vídeo STAR**.
- Gaps técnicos de entrega: **lock DVC representativo do dataset final**, **instalação limpa em máquina zerada**, **scripts longos**.

---

## 3. Objetivos identificados no Tech Challenge

### Objetivo principal (obrigatório)

Construir um **sistema de recomendação** (analogia e-commerce) com:

- modelo central **rede neural PyTorch** (MLP ou embedding-based);
- pipeline **containerizado (Docker)**;
- dados versionados com **DVC**;
- experimentos rastreados no **MLflow**;
- código com **clean code** e padrões profissionais.

### Entregáveis

| Entregável | Natureza |
|------------|----------|
| Repositório GitHub | **Obrigatório** |
| Vídeo ≤ 5 min (método STAR) | **Obrigatório** |
| Deploy em nuvem (AWS/Azure/GCP) | **Opcional** (+5% bônus) |

### Dataset

Sugerido: e-commerce (Instacart, RetailRocket) **ou MovieLens**; mínimo **≥ 10.000 interações user–item**.  
**Fato observado:** o projeto escolheu MovieLens 20M (aceitável pelo PDF).

### Diferenciação usada nesta auditoria

- **Obrigatório:** pedido explicitamente no PDF.
- **Recomendado:** sugerido no PDF (“boas práticas”, dataset sugerido, etc.) sem peso isolado.
- **Boa prática / plano interno:** itens do `TODO.md`, `.cursor/`, BERTopic, ablation, FastAPI — **não** exigidos literalmente pelo PDF.

---

## 4. Matriz de Requisitos

Legenda: ✅ Atendido · 🟡 Parcial · 🔴 Não atendido · ⚪ Não comprovado · ➖ N/A

| # | Requisito | Tipo | Status |
|---|-----------|------|--------|
| R01 | Repo GitHub com estrutura limpa (`src/`, `tests/`, `data/`, `models/`, `configs/`) | Obrigatório | ✅ |
| R02 | Clean code: módulos curtos, nomes descritivos, SOLID, type hints | Obrigatório | 🟡 |
| R03 | Funções ≤ 20 linhas | Obrigatório (boas práticas) | 🟡 |
| R04 | ≥ 1 design pattern (Factory / Strategy / Template) | Obrigatório | ✅ |
| R05 | Type hints + docstrings Google em APIs públicas | Obrigatório | 🟡 |
| R06 | Ruff sem erros + pre-commit | Obrigatório | ✅ |
| R07 | `pyproject.toml` Poetry/uv; deps prod/dev; lock commitado | Obrigatório | ✅ |
| R08 | `.dockerignore`, `.gitignore`, `.env.example` | Obrigatório | ✅ |
| R09 | Commits semânticos | Obrigatório | ✅ |
| R10 | `.env` + Pydantic Settings | Obrigatório | ✅ |
| R11 | `scripts/validate_env.py` | Obrigatório | ✅ |
| R12 | Instalação limpa (`poetry install` / equivalente `uv sync`) | Obrigatório | 🟡 |
| R13 | Dockerfile multi-stage | Obrigatório | ✅ |
| R14 | `docker-compose` treino + MLflow | Obrigatório | ✅ |
| R15 | DVC init + remote + dataset versionado | Obrigatório | 🟡 |
| R16 | Pipeline DVC ≥ 3 stages (`preprocess` → `feature_eng` → `train` → `evaluate`) | Obrigatório | ✅ |
| R17 | MLflow: params, métricas, artefatos | Obrigatório | ✅ |
| R18 | ≥ 3 runs rastreados | Obrigatório (critério) | ✅ |
| R19 | Model Registry Staging → Production | Obrigatório | ✅ |
| R20 | MLP/embedding PyTorch para recomendação | Obrigatório | ✅ |
| R21 | Early stopping | Obrigatório (critério rede neural) | ✅ |
| R22 | Baselines Scikit-Learn + comparação ≥ 4 métricas | Obrigatório | ✅ |
| R23 | Model Card (performance, limitações, vieses) | Obrigatório | ✅ |
| R24 | README com instruções completas | Obrigatório | ✅ |
| R25 | Vídeo STAR ≤ 5 min | Obrigatório | 🔴 |
| R26 | Deploy nuvem | Opcional | 🔴 |
| R27 | PyTorch, sklearn, MLflow, DVC | Obrigatório (libs) | ✅ |
| R28 | Seeds fixados | Obrigatório (boas práticas) | ✅ |
| R29 | BERTopic / features de conteúdo | Plano interno (não PDF) | 🔴 / ➖ |
| R30 | CI GitHub Actions | Boa prática interna | 🔴 / ➖ |

---

## 5. Arquitetura e funcionamento atual

### Fluxo observado (entrada → saída)

```text
data/raw/{ratings,movies,links}.csv
        │
        ▼
scripts/preprocess.py  ──Strategy──► data/processed/preprocessed_ratings.parquet
        │
        ▼
scripts/enrich_metadata.py + movie_metadata.parquet (TMDB cache)
        │
        ▼
data/processed/enriched_metadata.parquet
        │
        ▼
scripts/feature_engineering.py  ──► features_ratings.parquet (user_idx, movie_idx)
        │
        ▼
scripts/train.py (PyTorch MLP/embedding + early stopping + MLflow)
        │
        ▼
models/model.pth
        │
        ▼
scripts/evaluate.py (baselines + ranking@K + Registry + metrics.json)
```

### Mapa de componentes

| Área | Caminho | Papel |
|------|---------|--------|
| Domínio | `src/domain/` | IDs, Rating, Recommendation |
| Dados | `src/data/` | splits, preprocessors, TMDB/MovieLens I/O |
| Modelos | `src/models/` | Factory, MLP, Embedding, MostPopular, sklearn |
| Avaliação | `src/evaluation/` | métricas, Strategy de rating, Registry |
| Treino | `src/training/seeds.py` | seeds globais |
| Features / Serving | `src/features/`, `src/serving/` | **stubs vazios** |
| Config | `configs/settings.py`, `params.yaml` | env + hiperparâmetros DVC |
| Pipeline | `dvc.yaml`, `scripts/*.py` | stages DVC |
| Demo LLM-ish | `llm/` | recomendação a partir de histórico textual (não é LLM generativo) |
| Governança IA | `.cursor/`, `.github/` (espelho) | regras/comandos para agentes |

### Relação entre partes

O **coração avaliável do PDF** é: Factory de modelos + treino PyTorch + evaluate com baselines/métricas + DVC + Docker + MLflow.  
A coleta TMDB (`src/data/external/`, `scripts/fetch_external_metadata.py`) alimenta `movie_metadata.parquet`, que entra no stage `enrich_metadata`. Porém o stage `feature_eng` **quase não usa** os campos de conteúdo no treino neural — só o mapeamento de IDs. Ou seja: enriquecimento existe no pipeline, mas o modelo atual continua **fortemente colaborativo (user/item embeddings)**.

---

## 6. Auditoria de Aderência ao Tech Challenge

### R01 — Estrutura de pastas

**Status:** ✅ Atendido  
**Evidência:** `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`.  
**Análise:** Atende Etapa 1 do PDF.  
**O que falta:** nada crítico.

### R02 / R03 — Clean code e funções ≤ 20 linhas

**Status:** 🟡 Parcial  
**Evidência:**

- Vários módulos `src/` são curtos e legíveis (`factory.py`, `metrics.py`, preprocessors).
- Funções **acima de 20 linhas** detectadas por AST, por exemplo:
  - `scripts/train.py::main` (~130 linhas)
  - `scripts/evaluate.py::main` (~77), `_run_baselines` (~57), `_compute_ranking_metrics` (~50)
  - `scripts/demo_recommend.py::main` (~76)
  - `scripts/generate_llm_report_pdf.py::build` (~206)

**Análise:** O espírito de clean code está presente, mas a regra explícita de ≤ 20 linhas **não é cumprida de forma consistente**, sobretudo nos scripts de orquestração.  
**O que falta:** extrair helpers nos scripts longos (sem overengineering).

### R04 — Design patterns

**Status:** ✅ Atendido  
**Evidência:**

- Factory: `src/models/factory.py` (`create_model`)
- Strategy (preprocessadores): `src/data/preprocessors/base.py` + `explicit.py` / `implicit.py` + `registry.py`
- Strategy adicional (métricas de rating): `src/evaluation/metric_strategy.py`

**Análise:** Cumpre e até excede o mínimo (≥ 1 pattern).  
**O que falta:** nada obrigatório.

### R05 — Type hints e docstrings Google

**Status:** 🟡 Parcial  
**Evidência:** APIs públicas em `src/` em geral tipadas e com docstrings; scripts nem sempre; alguns métodos (`fit`/`predict` de torch) têm tipagem frouxa (`Any`).  
**O que falta:** homogeneizar scripts de pipeline e interfaces de modelo.

### R06 — Ruff sem erros + pre-commit

**Status:** 🟡 Parcial  
**Evidência:**

- `.pre-commit-config.yaml` com hooks `ruff` + `ruff-format`
- `pyproject.toml` com `[tool.ruff]`
- O `ruff check` foi executado na venv do projeto e terminou sem erros.

**Análise:** Ferramenta configurada e validada; o próximo passo é manter essa checagem no fluxo de PR/CI.

### R07 — pyproject + lock

**Status:** ✅ Atendido  
**Evidência:** `pyproject.toml` (prod + optional `dev`/`s3`/`gdrive`), `uv.lock` commitado.  
**Nota:** o PDF menciona `poetry install` no entregável da Etapa 2; o projeto usa **uv** (também citado no PDF em “Poetry/uv”). Aceitável, desde que o README deixe isso explícito (já deixa).

### R08 — dockerignore / gitignore / env.example

**Status:** ✅ Atendido  
**Evidência:** `.dockerignore`, `.gitignore` (inclui `.env`), `.env.example`.

### R09 — Commits semânticos

**Status:** ✅ Atendido  
**Evidência:** histórico local com prefixos `feat:`, `fix:`, `docs:`, `test:`, `build:`, `chore:`.

### R10 / R11 — Settings + validate_env

**Status:** ✅ Atendido  
**Evidência:** `configs/settings.py` (Pydantic Settings), `scripts/validate_env.py`.  
**Observação:** `validate_env.py` importa `packaging.version`, que **não está declarado diretamente** em `pyproject.toml` (provavelmente transitivo). Funciona no `.venv` atual, mas é frágil.

### R12 — Instalação limpa

**Status:** 🟡 Parcial / ⚪ parcialmente comprovado  
**Evidência:** README + `docs/DOCUMENTACAO_ETAPA2.md` descrevem `uv sync`.  
**Não foi possível comprovar** instalação em máquina 100% limpa nesta auditoria (`uv` não estava no PATH do shell; usou-se `.venv` existente).  
**O que falta:** validar em VM/PC limpo e documentar o resultado.

### R13 / R14 — Docker

**Status:** ✅ Atendido (com ressalvas de robustez)  
**Evidência:** `Dockerfile` (builder + runtime), `docker-compose.yml` com serviços `train` e `mlflow`.  
**Ressalvas:**

- `CMD` default da imagem é `validate_env.py` (ok para smoke; treino vem do compose).
- Compose monta `.:/app` (depende do host ter dados/`dvc.yaml`).
- Imagem não copia `dvc.yaml`/`params.yaml` no build (mitigado pelo volume).

### R15 / R16 — DVC

**Status:** 🟡 Parcial / ✅ stages  
**Evidência:**

- `dvc.yaml`: 5 stages (`preprocess`, `enrich_metadata`, `feature_eng`, `train`, `evaluate`) — **≥ 3 ✅**
- `.dvc/config`: remote `local_remote` → `../data/dvc_remote`
- `dvc.lock` presente

**Problemas:**

- Neste ambiente, `../data/dvc_remote` **não existia** (`Test-Path` → False).
- Em `dvc.lock`, deps de CSV têm tamanhos minúsculos (ex.: `ratings.csv` size **123** bytes) — típico de **dummy data**, não MovieLens 20M.

**O que falta:** garantir remote acessível + `dvc.lock` coerente com o dataset de entrega (ou documentar claramente o caminho dummy vs full).

### R17 / R18 — MLflow tracking e ≥ 3 runs

**Status:** ✅ Atendido (pelo código)  
**Evidência:** `scripts/train.py` e `scripts/evaluate.py` logam params/métricas/artefatos; evaluate cria runs para MostPopular + KNN + RandomForest + torch (≥ 3).  
**Observação:** presença de `mlruns/`/`mlflow.db` locais é esperada; `.gitignore` ignora `mlruns/` e `mlflow.db`.

### R19 — Registry Staging → Production

**Status:** 🟡 Parcial  
**Evidência:** `src/evaluation/registry.py::promote_champion` faz Staging e, se `validate_metrics` passar, Production.  
**Fato observado em `metrics.json` (versionado):**

```json
"champion": {
  "name": "sklearn_random_forest",
  "stage": "staging"
}
```

**Causa provável (fato + inferência):** métricas incluem `"r2": NaN`; `validate_metrics` exige `math.isfinite(...)`, então **não promove para Production**.  
**Conflito documental:** `docs/MODEL_CARD.md` afirma Production com `torch_mlp`.  
**O que falta:** regenerar evaluate no dataset real, excluir NaN da validação (ou não logar R² inválido), alinhar Model Card e `metrics.json`.

### R20 / R21 — PyTorch MLP + early stopping

**Status:** ✅ Atendido  
**Evidência:** `src/models/torch_mlp.py`, `src/models/torch_embedding.py`, early stopping em `scripts/train.py` (`patience`, checkpoint `models/model.pth`).

### R22 — Baselines + ≥ 4 métricas

**Status:** ✅ Atendido  
**Evidência:** MostPopular + KNN + RandomForest; métricas RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate (`src/evaluation/metrics.py`, `scripts/evaluate.py`).

### R23 — Model Card

**Status:** 🟡 Parcial  
**Evidência:** `docs/MODEL_CARD.md` completo em estrutura (dados, métricas, limitações, vieses).  
**Problema:** números/champion **não batem** com `metrics.json` atual do repo.  
**O que falta:** sincronizar com o run oficial de entrega.

### R24 — README

**Status:** ✅ Atendido  
**Evidência:** `README.md` com setup, pipeline, Docker, links de docs. Deploy cloud ainda placeholder (aceitável por ser opcional).

### R25 — Vídeo STAR

**Status:** 🔴 Não atendido  
**Evidência:** ausência no repo; `TODO.md` item 6.5 marcado `[ ]`.  
**O que falta:** gravar e anexar/linkar o vídeo (fora do código).

### R26 — Deploy nuvem

**Status:** 🔴 Não atendido (opcional)  
**Evidência:** README (“ainda será publicado”); `src/serving/` vazio.

### R29 — BERTopic (plano interno)

**Status:** 🔴 Não atendido / ➖ não obrigatório no PDF  
**Evidência:** deps em `pyproject.toml`; `scripts/feature_engineering.py` **não importa** BERTopic; `src/features/__init__.py` é stub.  
**Análise:** não deve ser tratado como falta do PDF; é gap do plano interno/`TODO.md`.

---

## 7. Auditoria de Refactor e Estrutura

### A estrutura atual é adequada?

**Sim, em linhas gerais.** A organização por camadas (`domain`, `data`, `models`, `evaluation`, `training`) é proporcional ao desafio acadêmico e facilita a demonstração STAR. Não há necessidade de microserviços, CQRS, etc.

### Refactors relevantes (recomendados, não executados)

| Problema atual | Onde | Impacto | Refactor recomendado | Prioridade |
|----------------|------|---------|----------------------|------------|
| Scripts orquestradores monolíticos | `scripts/train.py`, `scripts/evaluate.py` | Dificulta manutenção e viola ≤20 linhas | Extrair loaders, ranking eval, logging MLflow para `src/training/` / `src/evaluation/` | Alta |
| `feature_eng` só mapeia IDs; metadados TMDB quase não entram no modelo | `scripts/feature_engineering.py` | Pipeline “enriquece” sem efeito no treino | Ou usar features de conteúdo, ou documentar honestamente que o modelo é só colaborativo | Alta (honestidade/STAR) |
| Pacotes vazios | `src/features/`, `src/serving/` | Confusão arquitetural | Implementar o mínimo ou remover da narrativa até existir código | Média |
| Duplicação de governança | `.cursor/` espelhado em `.github/` | Drift documental | Manter uma fonte da verdade; espelhar só o necessário (CI) | Baixa |
| `fit()` no-op nos modelos torch | `torch_mlp.py`, `torch_embedding.py` | API confusa (`fit` não treina) | Treinar via método real ou renomear/documentar que o treino é externo | Média |
| Champion só por RMSE | `scripts/evaluate.py` | Pode eleger baseline sklearn e contradizer narrativa neural | Critério composto (RMSE + NDCG) alinhado ao Model Card | Alta |
| Validação de métricas rejeita NaN | `registry.py` + R² | Impede Production | Não incluir R² instável na validação de promoção | Alta |
| Comentários placeholder em feature_eng | `feature_engineering.py` | Cheiro de scaffold incompleto | Completar ou limpar comentários | Média |

### O que **não** vale a pena refatorar agora

- Reescrever Factory/Strategy (já atendem o PDF).
- Trocar uv por Poetry só por formalismo.
- Criar API FastAPI completa se o bônus de nuvem não for perseguido.
- Introduzir BERTopic pesado só para “completar TODO”, se isso atrasar o vídeo STAR.

> **Não há necessidade clara de alterar** a estrutura base `src/models` + `src/evaluation/metrics.py` + `src/data/preprocessors` — estão adequados ao desafio.

---

## 8. Auditoria de Limpeza do Projeto

| Arquivo/Componente | Motivo | Evidência | Recomendação |
|--------------------|--------|-----------|--------------|
| `scripts/hello_train.py` | Scaffold antigo (“hello train”) | Só imprime paths; não referenciado em `dvc.yaml`/compose | **Seguro para remover** (após confirmar que ninguém usa) |
| `src/features/__init__.py` / `src/serving/__init__.py` | Pacotes vazios | Apenas docstring; sem imports no código | **Manter** como placeholder **ou** remover se confundir a banca — validar narrativa |
| `bertopic` / `sentence-transformers` em prod deps | Pesados e não usados no código de treino | Nenhum `import bertopic` em `scripts/`/`src/` de pipeline | **Provavelmente removível** das deps de prod (mover para optional) — validar plano do time |
| Docs `docs/CURSOR_STRUCTURE_REPORT.md`, `docs/CURSOR_REFACTOR_REPORT.md` | Relatos históricos da pasta `.cursor` | Descrevem estado antigo / scaffold SetAI | **Provavelmente removível** do material de entrega (arquivar) |
| Espelho `.github/commands` ≈ `.cursor/commands` | Duplicação | Contagens similares (~33–36 arquivos) | **Manter** um; limpar o outro com cuidado |
| `.cursor/.setai/` | Resíduo do gerador SetAI | README próprio; relatório Cursor sugere arquivar | **Provavelmente removível** |
| `metrics.json` atual | Artefato de run do snapshot | Champion RF em Production | **Manter arquivo** e versionar o run oficial quando houver o dataset final |
| `llm/` + `scripts/generate_llm_report_pdf.py` | Demo/extra | Não exigido pelo PDF; útil para apresentação | **Manter** se for usado no STAR; senão marcar como extra |
| `scripts/demo_recommend.py`, `create_dummy_data.py` | Utilitários de demo | Úteis para smoke test | **Manter** |
| `AUDITORIA_DESAFIO.md` / auditorias antigas | Checklists anteriores | Complementares | **Manter** ou consolidar nesta auditoria |
| `data/processed/*.parquet` locais grandes | Gerados | `.gitignore` ignora (exceto `movie_metadata.parquet`) | **Manter política**; não commitar features/ratings |
| `movie_metadata.parquet` (~7,8 MB) | Cache TMDB versionado | Tracked + dep do DVC | **Manter** (evita refetch) |
| `.env` local | Secrets | gitignored; chaves presentes localmente | **Manter fora do Git** |
| `mlflow.db` / `mlruns/` locais | Runtime | gitignored | **Manter localmente**; não versionar |

---

## 9. Auditoria de Possíveis Resíduos ou Erros de Código Gerado por IA

### 9.1 Comentários e textos

Sinais observados (não prova de autoria por IA, mas padrões típicos de geração assistida):

- Comentários didáticos / placeholder em `scripts/feature_engineering.py` (“Simple feature engineering…”, “In a real scenario…”).
- Documentação massiva de governança Cursor/SetAI (`docs/CURSOR_*`, `.cursor/.setai`).
- Pasta `llm/examples/prompt_avaliacao_imparcial.md` menciona ChatGPT/Claude como **ferramenta de avaliação humana** — uso consciente, não “código gerado escondido”.
- Relatórios com tom de agente (`docs/CURSOR_REFACTOR_REPORT.md`: “*Gerado após refatoração…*”).

Não foram encontrados comentários do tipo `Generated by ChatGPT` dentro de `src/`.

### 9.2 Código suspeito

| Sinal | Evidência | Risco |
|-------|-----------|-------|
| API `fit()` que não treina | `TorchMLPRecommender.fit` só seta `_fitted=True` | Confusão; evaluate chama `fit([], None)` só para liberar `predict` |
| Dois caminhos de métricas | `evaluation/metrics.py` (ranking) vs `metric_strategy.py` (rating sklearn) | Ok se documentado; parece acúmulo de iterações |
| `feature_eng` incompleto vs docs que prometem BERTopic/embeddings | docs vs script | Narrativa desalinhada |
| Champion sklearn no artefato vs neural no Model Card | `metrics.json` vs `MODEL_CARD.md` | Risco alto na avaliação oral/vídeo |
| JSON com `NaN` não-padrão | `metrics.json` | Pode quebrar parsers estritos |
| Exceções amplas | `validate_env.py` `except Exception` | Aceitável em script de diagnóstico |

### 9.3 Dependências (ver também seção 10)

- Dependências pesadas **declaradas e não usadas** no pipeline (`bertopic`, `sentence-transformers`).
- `packaging` usado sem declaração direta.
- Stack principal (torch, sklearn, mlflow, dvc) está alinhada ao PDF.

### 9.4 Segurança e configurações

- `.env` **não** está versionado (bom).
- `.env` local contém `TMDB_API_KEY` **preenchida** (comprimento 32) e `DATABASE_URL` preenchida — **não reproduzidos aqui**.
- `.env.example` sem segredos (bom).
- Compose usa `env_file: .env.example` (sem chave TMDB) — ok para treino colaborativo; scraping exigiria `.env` real.
- Sem evidência de secret commitado no Git nesta auditoria.
- Não há CI; logo não há vazamento via logs de Actions (tampouco há automação).

---

## 10. Dependências e Segurança

### Dependências obrigatórias do PDF

| Lib | Presente? | Evidência |
|-----|-----------|-----------|
| PyTorch | ✅ | `pyproject.toml`, `src/models/torch_*.py` |
| Scikit-Learn | ✅ | baselines + metric strategies |
| MLflow | ✅ | train/evaluate/registry |
| DVC | ✅ | `dvc.yaml`, deps |

### Problemas de dependências

| Item | Classificação | Motivo |
|------|---------------|--------|
| `bertopic`, `sentence-transformers` | Possivelmente **desnecessárias hoje** | Sem uso no código de pipeline |
| `packaging` | Declarativa incompleta | Importado em `validate_env.py`, não listado no `pyproject.toml` |
| Versões pinned via lock | Bom | `uv.lock` |
| Vulnerabilidades CVE | ⚪ Não comprovado | Não foi executado audit de CVE dedicado nesta sessão |

**Desatualizada ≠ vulnerável.** Nenhuma biblioteca foi marcada como vulnerável sem evidência de scanner.

### Segurança — resumo

| Tema | Status |
|------|--------|
| Secrets no Git | ✅ Parece ok (`.env` ignorado) |
| Secrets locais | ⚠️ Existem no `.env` da máquina (esperado) |
| CORS / API pública | ➖ Sem API de serving |
| Pickle/torch load | ⚠️ `torch.load` em evaluate/tests — risco clássico; aceitável se artefato for do próprio time |
| Dataset bruto no Git | ✅ CSVs raw ignorados |

---

## 11. Qualidade de Código

**Pontos fortes**

- Separação clara domínio / dados / modelos / avaliação.
- Type hints e docstrings na maior parte de `src/`.
- Seeds centralizados (`src/training/seeds.py`).
- Split temporal com assert (`src/data/splits.py`).
- Métricas de ranking implementadas e testadas com casos conhecidos.

**Pontos fracos**

- Scripts longos concentrando orquestração + I/O + MLflow.
- Regra ≤ 20 linhas frequentemente violada.
- Scripts longos concentrando orquestração + I/O + MLflow.
- `feature_eng` superficial frente ao restante do MLOps.
- `feature_eng` superficial frente ao restante do MLOps.

**Contexto acadêmico:** a qualidade é **suficiente para o desafio**; o foco agora é concluir o vídeo e decidir se o dataset final será o snapshot atual ou o MovieLens completo.

---

## 12. Testes e Confiabilidade

### O que existe

- **Unitários** em `tests/unit/`: métricas, factory, preprocessors, splits, TMDB client (mock), metadata fetch, paths, MostPopular, coverage report, domain IDs.
- **Integração** `tests/integration/test_real_inference.py`: depende de `models/model.pth` + fixture JSON (skip se ausente).
- Nesta auditoria: **`pytest tests/unit` → 52 testes OK**.

### Lacunas importantes

| Área | Cobertura | Risco |
|------|-----------|-------|
| `scripts/train.py` / early stopping | Baixa/ausente | Regressão de treino |
| `scripts/evaluate.py` / Registry promote | Ausente | Bug NaN → Staging passa despercebido |
| Pipeline DVC ponta a ponta | Ausente em CI | `dvc repro` quebra em máquina limpa |
| Ranking metrics no evaluate com modelo real | Parcial (integração separada) | — |
| Serving/API | N/A | — |

### CI

**Não há** `.github/workflows/` — boa prática interna, não requisito literal do PDF.

**Conclusão:** testes são **bons para o núcleo de métricas/factory**, mas **não protegem** o critério de Registry Production nem a reprodutibilidade DVC full.

---

## 13. Documentação

| Documento | Avaliação |
|-----------|-----------|
| `README.md` | Bom e atual; deploy cloud ainda placeholder |
| `docs/MODEL_CARD.md` | Estrutura excelente; agora sincronizado com `metrics.json` |
| `docs/DOCUMENTACAO_ETAPA2.md` | Sólida para uv/settings/validate_env |
| `docs/IMPLEMENTACAO_MLP_PYTORCH.md` | Útil para STAR |
| `docs/AUDITORIA_DESAFIO.md` | Checklist curto; parcialmente desatualizado vs artefatos |
| `TODO.md` | Plano rico; ainda marca vídeo aberto; alguns itens internos (BERTopic) misturam obrigação e desejo |
| Docs TMDB (`FLUXO_*`, `GUIA_SCRAPING_*`) | Boas; prometem uso futuro em features |
| Docs Cursor (`CURSOR_*`) | Históricos; pouco úteis para banca |
| PDF do desafio | Presente em `docs/` |

**Observação:** o run atual ficou consistente entre Model Card, `metrics.json` e Registry.

---

## 14. Pontos Positivos

1. **Factory + Strategy reais e usados** no pipeline — não são “classes enfeite”.
2. **Pipeline DVC com 5 stages** bem acima do mínimo de 3.
3. **Early stopping + scheduler + seeds** no treino PyTorch.
4. **Comparação neural vs baselines** com tabela em `metrics.json`.
5. **Coleta TMDB cacheada** (parquet versionado) — alinhada à ideia de não chamar API a cada `dvc repro`.
6. **Model Card** bem escrito para apresentação.
7. **Testes de métricas com oráculos conhecidos** — adequado ao enunciado de recomendação.
8. **README executável** (`uv sync`, `validate_env`, `dvc repro`, compose).

> **Não há necessidade clara de alterar** a implementação das métricas em `src/evaluation/metrics.py` nem o padrão Strategy dos preprocessors — estão corretos para o desafio.

---

## 15. Problemas P0 — Crítico

1. **Vídeo STAR ausente** — 10% da nota; entregável obrigatório do PDF.
2. **`dvc.lock` / dados versionados ainda refletem o snapshot de smoke test** — risco de narrativa se a entrega final exigir MovieLens completo.

---

## 16. Problemas P1 — Importante

1. **`dvc.lock` com CSVs minúsculos** — sugere lock de smoke/dummy; prejudica “`dvc repro` funcional” na narrativa de entrega final.
2. **Remote DVC `../data/dvc_remote` ausente neste ambiente** — reprodutibilidade frágil.
3. **`feature_eng` não consome metadados TMDB no modelo** — docs/TODO prometem mais do que o código entrega.
4. **Funções ≫ 20 linhas** nos scripts principais.
5. **Champion por RMSE apenas** — pode eleger baseline e enfraquecer o discurso do MLP.

---

## 17. Problemas P2 — Recomendado

1. Declarar `packaging` (ou remover dependência) em `validate_env.py`.
2. Mover BERTopic/sentence-transformers para extras opcionais até existir código.
3. Cobrir Registry/evaluate com testes unitários (NaN, promoção).
4. Adicionar CI mínima (`ruff` + `pytest`).
5. Limpar stubs/`hello_train.py` e docs Cursor obsoletos.
6. Validar instalação limpa documentada.

---

## 18. Problemas P3 — Opcional

1. Deploy cloud / FastAPI (`src/serving/`).
2. Ablation BERTopic/TMDB no forward.
3. Unificar espelho `.cursor`/`.github`.
4. Remover pasta `llm/` se não entrar no vídeo.
5. Polimento de nomenclatura `fit()` nos modelos torch.

---

## 19. Plano de Ação Recomendado

> **Não executar automaticamente.** Ordem sugerida para maximizar nota com menos risco.

| # | Prioridade | Problema | Ação recomendada | Arquivos | Motivo | Impacto esperado |
|---|------------|----------|------------------|----------|--------|------------------|
| 1 | P0 | Vídeo STAR | Roteiro STAR + gravar ≤5 min + link no README | `README.md`, material externo | Entregável obrigatório | +10% potencial |
| 2 | P0 | Vídeo STAR | Roteiro STAR + gravar ≤5 min + link no README | `README.md`, material externo | Entregável obrigatório | +10% potencial |
| 3 | P1 | DVC lock/remote | Regenerar `dvc.lock` com dados de entrega; documentar remote | `dvc.lock`, `.dvc/config`, README | Reprodutibilidade | Etapa 3 |
| 4 | P1 | Narrativa feature_eng | Usar metadados **ou** declarar modelo colaborativo puro | `feature_engineering.py`, docs | Evitar overclaim | STAR/Action |
| 5 | P1 | Scripts longos | Extrair funções ≤20 linhas | `train.py`, `evaluate.py` | Clean code | Manutenção |
| 6 | P2 | Deps mortas | Optional-deps BERTopic | `pyproject.toml` | Instalação mais leve | Reprodutibilidade |
| 7 | P2 | Testes de registry | Unit tests NaN/promote | `tests/unit/` | Evitar regressão | Confiabilidade |
| 8 | P3 | Limpeza | Remover `hello_train`, docs Cursor velhos | scripts/docs | Menos ruído | Polimento |
| 9 | P3 | Bônus cloud | Só se sobrar tempo | `serving/`, deploy | +5% | Opcional |

---

## 20. Checklist Final para Entrega

- [ ] Vídeo STAR ≤ 5 min publicado/linkado
- [ ] `metrics.json` e Model Card revisados no mesmo run de entrega final, se houver dataset completo
- [ ] Modelo no MLflow Registry em **Production** (evidência screenshot/run id)
- [x] `ruff check` = 0 erros
- [ ] `pytest` verde (unit + integração com `model.pth` de entrega)
- [ ] `uv sync` + `validate_env.py` ok em máquina limpa
- [ ] `dvc repro` ok com dataset documentado (full ou amostra explícita)
- [ ] Docker: `docker compose run --rm train` e UI MLflow acessível
- [ ] README revisado (sem links quebrados / placeholders críticos)
- [ ] Confirmar que nenhum `.env` com secrets será commitado
- [ ] Decidir se BERTopic entra na fala do vídeo (hoje: código não usa)
- [ ] (Opcional) URL pública do deploy

---

## 21. Conclusão

### O projeto atende ao Tech Challenge?

**Sim, no código e na documentação técnica atual, com uma pendência obrigatória ainda aberta: o vídeo STAR.**  
Tecnicamente, a base pedida pelo PDF está implementada e os artefatos principais agora estão sincronizados.

### Quanto está atendido?

Estimativa: **~80%** de aderência técnica ponderada pelos critérios do PDF (vídeo ainda zerando 10 pp).

### O que ainda falta (obrigatório)?

1. Vídeo STAR  
2. Artefatos DVC/métricas do dataset final, caso a entrega seja com MovieLens completo  
3. Evidência de instalação limpa em máquina zerada

### Riscos de perder pontos se entregar agora

| Critério | Risco |
|----------|-------|
| Vídeo STAR (10%) | Perda quase certa se não entregar |
| DVC (15%) | Desconto se `dvc repro`/lock não refletirem o dataset apresentado |
| Rede neural (15%) | Menor risco de código; a narrativa precisa explicar o champion do snapshot atual |
| Docker / Reprodutibilidade | Risco moderado (compose depende de volume/dados) |
| Bônus cloud (5%) | Não pontua (opcional) |

### Resposta direta às 15 perguntas da missão

1. **Atende?** Quase — código sim em larga medida; entrega completa ainda não.  
2. **Quanto?** ~74% (estimativa).  
3. **Falta?** Vídeo e, se houver entrega com dataset final, o lock DVC correspondente.  
4. **Estrutura errada?** Não estruturalmente; scripts longos e `feature_eng` ainda poderiam ser refinados.
5. **Vale refatorar?** Sim: evaluate/registry, feature_eng/narrativa, quebrar scripts.  
6. **Não vale?** Trocar stack, inventar microserviços, forçar BERTopic às pressas.  
7. **Remover?** `hello_train.py`, docs Cursor históricos, deps BERTopic até uso real.  
8. **Resíduos de IA?** Placeholders/scaffold e docs SetAI; sem “Generated by ChatGPT” no `src/`.  
9. **Deps problemáticas?** Pesadas sem uso (BERTopic); `packaging` indireto; CVE não auditado.  
10. **Segurança?** `.env` local com chave (ok se não commitada); sem API aberta.  
11. **Testes?** Bons no núcleo; fracos no Registry/pipeline.  
12. **Docs?** Boas, mas Model Card contradiz `metrics.json`.  
13. **Antes da entrega?** STAR + decidir dataset final + revisar o lock DVC.  
14. **Ordem?** Ver Plano de Ação (seção 19).  
15. **Riscos de nota?** Vídeo, reprodutibilidade DVC se o dataset final não estiver versionado, e a narrativa do modelo se o snapshot atual for usado sem explicação.

---

*Fim da auditoria. Nenhuma alteração de código foi realizada além da criação deste relatório.*
