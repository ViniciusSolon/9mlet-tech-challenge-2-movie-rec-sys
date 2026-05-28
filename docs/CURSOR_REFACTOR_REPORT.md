# Relatório de Refatoração — Pasta `.cursor`

**Projeto:** FIAP Tech Challenge 02 — MovieLens 20M + MLOps  
**Data:** 27 de maio de 2026  
**Referência:** `CURSOR_STRUCTURE_REPORT.md`, `TODO.md`  
**Escopo:** Alinhamento completo Python / PyTorch / DVC / MLflow / Docker

---

## Sumário executivo

A pasta `.cursor` foi **refatorada de scaffold Node.js/CLI (SetAI) para governança MLOps Python** focada em sistema de recomendação MovieLens 20M. Foram alterados **24 arquivos**, **criados 10 novos** e adicionados **3 rules nativas Cursor (`.mdc`)** + **`AGENTS.md`** na raiz.

---

## Arquivos alterados

| Arquivo | Ação |
|---------|------|
| `.cursor/README.md` | Reescrito |
| `.cursor/context/tech-stack.md` | Reescrito |
| `.cursor/context/architecture.md` | Reescrito |
| `.cursor/context/deployment.md` | Reescrito |
| `.cursor/context/project-goals.md` | Reescrito (listas corrigidas) |
| `.cursor/context/ml-pipeline.md` | **Criado** |
| `.cursor/rules/code-style.md` | Reescrito |
| `.cursor/rules/testing-rules.md` | Reescrito |
| `.cursor/rules/security-rules.md` | Reescrito |
| `.cursor/rules/ai-usage-rules.md` | Reescrito |
| `.cursor/rules/business-rules.md` | Reescrito |
| `.cursor/rules/git-rules.md` | Atualizado (ruff/pytest) |
| `.cursor/rules/python-ml.mdc` | **Criado** |
| `.cursor/rules/mlops.mdc` | **Criado** |
| `.cursor/rules/testing.mdc` | **Criado** |
| `.cursor/libs/allowed-libs.md` | Reescrito |
| `.cursor/libs/forbidden-libs.md` | Reescrito |
| `.cursor/libs/ai-models.md` | Reescrito |
| `.cursor/commands/*.md` (10 existentes) | Reescritos |
| `.cursor/commands/_project-context.md` | **Criado** (referência) |
| `.cursor/commands/dvc-pipeline-review.md` | **Criado** |
| `.cursor/commands/mlflow-experiment-design.md` | **Criado** |
| `.cursor/commands/recommendation-metrics.md` | **Criado** |
| `.cursor/commands/model-card-generator.md` | **Criado** |
| `.cursor/commands/feature-engineering-review.md` | **Criado** |
| `AGENTS.md` | **Criado** (raiz) |

**Não alterados (intencional):** `.cursor/.setai/*` — config do gerador SetAI; manter fora do Git com `.gitignore` raiz.

---

## Mudanças executadas (por objetivo)

### 1. Stack tecnológica
**Antes:** Node.js, ESLint, Prettier, TypeScript, Vitest, npm publish, Commander.js.  
**Depois:** Python 3.11+, Poetry/uv, PyTorch, sklearn, MLflow, DVC, Pandas, NumPy, PostgreSQL, Ruff, Pytest, Docker, Pydantic, FastAPI (opcional), sentence-transformers/BERTopic (opcional).

### 2. Arquitetura
**Antes:** Layered REST + PostgreSQL SoT + framework “Nenhum” + placeholders.  
**Depois:** Clean architecture ML (`src/domain` … `serving`), pipeline DVC, diagramas Mermaid, Factory/Strategy, fluxo raw → preprocess → feature_eng → train → evaluate → registry → inference.

### 3. Deployment
**Antes:** `npm publish`, `NODE_ENV`, release npm.  
**Depois:** Docker multi-stage, docker-compose, MLflow server, DVC remote, `.env`, `poetry install`, `dvc repro`, `docker compose up`.

### 4. Rules
**Antes:** ESLint/TypeScript obrigatórios; exemplos TS; Prisma/node-postgres.  
**Depois:** Ruff, Pytest, type hints, Google docstrings, SOLID, seeds, métricas Recall@K/NDCG@K/RMSE, regras cold start/top-K/leakage.

### 5. Libs
**Antes:** Whitelist Node CLI.  
**Depois:** torch, sklearn, mlflow, dvc, pandas, fastapi, sqlalchemy, psycopg2, bertopic, etc.; TensorFlow e stack Node proibidos.

### 6. Commands
**Antes:** ~15 linhas duplicadas MovieLens por arquivo + stack Node.  
**Depois:** Prompts curtos referenciando `context/`; 5 commands ML novos.

### 7. Rules nativas Cursor
- `python-ml.mdc` — `alwaysApply: true`, globs `**/*.py`
- `mlops.mdc` — `alwaysApply: true`, globs DVC/Docker/CI
- `testing.mdc` — `alwaysApply: true`, globs `tests/**`

### 8. Governança
- `context/ml-pipeline.md` — stages, ablations, padrões
- `AGENTS.md` — Data Engineer, ML Engineer, MLOps Engineer, Reviewer

### 9. Problemas corrigidos
| Problema | Status |
|----------|--------|
| Framework “Nenhum” | Removido |
| `{{TEST_COVERAGE}}` | Substituído por tabela de cobertura |
| Placeholders `[To be defined]` em architecture | Preenchidos |
| Listas de usuários quebradas | Corrigidas em project-goals |
| Duplicação massiva em commands | Reduzida via referências |
| Referências npm/Node | Removidas dos arquivos principais |
| REST genérico como centro | Substituído por pipeline batch ML |

---

## Arquitetura final (resumo)

```
MovieLens 20M (DVC)
    → preprocess (Strategy)
    → enrich_metadata (TMDB cache)
    → feature_eng (BERTopic opcional)
    → train (PyTorch Factory + sklearn baselines)
    → evaluate (Recall@K, NDCG@K, RMSE, …)
    → MLflow Registry
    → inference (batch / FastAPI opcional)
```

**Governança Cursor:** `.mdc` alwaysApply + `context/` + `rules/` + `commands/` + `AGENTS.md`.

---

## Aderência ao Tech Challenge

| Critério | Antes (`.cursor`) | Depois |
|----------|-------------------|--------|
| PyTorch | Só em goals | architecture + rules + pipeline |
| DVC ≥3 stages | Texto | `ml-pipeline.md` + command review |
| MLflow | Texto | deployment + commands + mlops.mdc |
| Docker | Ausente/errado | deployment.md |
| Ruff + pytest | Ausente | code-style + testing.mdc |
| Factory/Strategy | Ausente | architecture + business |
| Métricas ≥4 | Ausente | business-rules + recommendation-metrics |
| Clean Code/SOLID | Genérico | code-style + python-ml.mdc |

**Nota estimada de aderência documental:** ~4,5/10 → **~9/10** (código `src/` ainda a implementar conforme `TODO.md`).

---

## Aderência ao MovieLens 20M

| Aspecto | Cobertura |
|---------|-----------|
| Arquivos CSV (ratings, movies, tags, links) | `ml-pipeline.md` |
| Analogia e-commerce | `ml-pipeline.md`, `business-rules.md` |
| TMDB enrich + cache DVC | pipeline + security |
| BERTopic em feature_eng | tech-stack + pipeline |
| Ablation 4 runs | `ml-pipeline.md` + `mlflow-experiment-design.md` |
| Escala 20M | architecture (chunking, dev subset) |

---

## Riscos restantes

| ID | Risco | Mitigação |
|----|-------|-----------|
| R1 | Código `src/` ainda não existe | Seguir `TODO.md` |
| R2 | `.setai/config.json` no repo | Adicionar ao `.gitignore` raiz |
| R3 | Rules `.md` + `.mdc` duplicam conteúdo | Manter `.mdc` conciso; `.md` como referência longa |
| R4 | Cursor pode não carregar `.md` em subpastas antigas | Priorizar `.mdc` alwaysApply |
| R5 | TDD 100% business logic vs ML | testing-rules flexibiliza para pipeline |

---

## Próximos passos (fora de `.cursor`)

1. Implementar estrutura `src/`, `tests/`, `dvc.yaml`, `pyproject.toml`, `Dockerfile`  
2. Adicionar na raiz: `.gitignore` (`data/raw/`, `mlruns/`, `.env`, `.cursor/.setai/config.json`)  
3. Executar Bloco 0–1 do `TODO.md`  
4. Validar que Cursor aplica as 3 rules `.mdc` em sessão de teste  
5. Opcional: remover ou arquivar `.cursor/.setai/` após scaffold estável  

---

## Antes / depois (exemplo representativo)

### `tech-stack.md`
```diff
- ESLint, Prettier, TypeScript, Vitest, pnpm
+ Ruff, Pytest, PyTorch, MLflow, DVC, Poetry/uv
```

### `deployment.md`
```diff
- npm publish, npm version, NODE_ENV
+ docker compose up, dvc repro, poetry install, MLFLOW_TRACKING_URI
```

### `allowed-libs.md`
```diff
- Commander.js, Inquirer, tsup
+ torch, mlflow, dvc, pandas, sqlalchemy, bertopic
```

---

## Conclusão

A pasta `.cursor` está **alinhada ao Tech Challenge Fase 02** e ao domínio **MovieLens 20M + MLOps**. A IA do Cursor passa a receber regras nativas (`.mdc`), contexto de pipeline e commands específicos para DVC, MLflow e métricas de recomendação, eliminando o desvio para stack Node.js.

**Próximo marco:** implementação do repositório Python conforme `TODO.md`, usando esta governança como contrato.

---

*Gerado após refatoração executada em todos os arquivos listados.*
