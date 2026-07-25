# Auditoria Técnica Completa — Tech Challenge Fase 02

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys` (sistema de recomendação MovieLens)  
**Fonte de requisitos:** `docs/Tech Challenge Fase 02.pdf` (texto em `docs/pdf_extract.txt`)  
**Data da auditoria:** 2026-07-25  
**Branch analisada:** `feat/bloco5-mlp-pytorch-training-model-card-llm` (`c528344`)  
**Escopo:** análise somente leitura + checagens não destrutivas (`ruff check`, `pytest` unitário, inspeção de artefatos).  
**Regra seguida:** nenhum código, dependência, configuração ou estado de dados foi alterado; este arquivo Markdown é a única entrega desta tarefa.

---

## 1. Resumo Executivo

O repositório **já implementa a maior parte dos requisitos técnicos obrigatórios do PDF**: estrutura limpa, Factory + Strategy, `pyproject.toml` + `uv.lock`, Settings + `.env`, Dockerfile multi-stage, docker-compose (treino + MLflow), pipeline DVC com 5 stages, MLP/embedding PyTorch com early stopping, baselines Scikit-Learn, ≥ 4 métricas, MLflow tracking + Registry Staging→Production, Model Card e README.

**Veredito curto:** o projeto está **bem alinhado** ao desafio no código. O gap obrigatório mais claro para a entrega é o **vídeo STAR (10% da nota)**. Há também um risco técnico importante de nota: o que está **versionado no Git (`dvc.lock` + `metrics.json`)** ainda reflete um **smoke test minúsculo**, enquanto a máquina local já possui MovieLens em escala real (~19,7M ratings) e um `model.pth` treinado — ou seja, a evidência “oficial” do repositório e o workspace local **não estão sincronizados**.

Principais riscos se entregarem o estado atual do Git:

1. **Vídeo STAR ausente** (obrigatório).
2. **`metrics.json` / Model Card** com ranking @K zerado e champion = `sklearn_random_forest` (não o MLP) — narrativa frágil para banca.
3. **`dvc.lock` com CSVs de ~100 bytes** — não prova o MovieLens completo no pipeline versionado.
4. Clean code: **38 funções > 20 linhas** (regra explícita do PDF).

---

## 2. Nota / Aderência Geral

### Estimativa técnica de aderência: **~78%**

Esta porcentagem **não é a nota oficial da FIAP**. É uma estimativa ponderada pelos critérios do PDF (bônus de nuvem fora do obrigatório).

| Critério (PDF) | Peso | Estimativa de atendimento | Contribuição |
|----------------|------|---------------------------|--------------|
| Clean code e estrutura | 15% | ~68% | 10,2 |
| Reprodutibilidade | 15% | ~90% | 13,5 |
| Docker | 15% | ~82% | 12,3 |
| DVC + Pipeline | 15% | ~72% | 10,8 |
| Rede neural (PyTorch) | 15% | ~85% | 12,8 |
| MLflow + Registry | 10% | ~88% | 8,8 |
| Vídeo STAR | 10% | ~0% | 0,0 |
| **Subtotal obrigatório** | **95%** | — | **~68,4 / 95 ≈ 72%** |
| Bônus cloud (opcional) | 5% | ~0% | 0 |

**Ajuste qualitativo (+6 pp):** testes unitários verdes (52), ruff limpo, Model Card honesta sobre o snapshot, instalação limpa documentada/validada → **~78%**.

### Classificação

**Bem alinhado** (código quase pronto para entrega técnica; falta principalmente o vídeo STAR e sincronizar artefatos de evidência no repositório).

**Critério usado:** pesos do PDF × grau de comprovação no código/artefatos versionados (não só o que existe na máquina do desenvolvedor).

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

Sugerido: e-commerce **ou MovieLens**; mínimo **≥ 10.000 interações user–item**.  
**Fato observado:** o projeto escolheu MovieLens (aceitável pelo PDF).

### Diferenciação usada nesta auditoria

- **Obrigatório:** pedido explicitamente no PDF.
- **Recomendado:** sugerido no PDF sem peso isolado.
- **Boa prática / plano interno:** `TODO.md`, BERTopic, ablation, FastAPI, CI — **não** exigidos literalmente pelo PDF.

---

## 4. Matriz de Requisitos

Legenda: ✅ Atendido · 🟡 Parcial · 🔴 Não atendido · ⚪ Não comprovado · ➖ N/A

| # | Requisito | Tipo | Status |
|---|-----------|------|--------|
| R01 | Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/` | Obrigatório | ✅ |
| R02 | Clean code: SOLID, naming, type hints | Obrigatório | 🟡 |
| R03 | Funções ≤ 20 linhas | Obrigatório (boas práticas) | 🟡 |
| R04 | ≥ 1 design pattern (Factory / Strategy) | Obrigatório | ✅ |
| R05 | Type hints + docstrings Google | Obrigatório | 🟡 |
| R06 | Ruff sem erros + pre-commit | Obrigatório | ✅ |
| R07 | `pyproject.toml` Poetry/uv + lock | Obrigatório | ✅ |
| R08 | `.dockerignore`, `.gitignore`, `.env.example` | Obrigatório | ✅ |
| R09 | Commits semânticos | Obrigatório | ✅ |
| R10 | `.env` + Pydantic Settings | Obrigatório | ✅ |
| R11 | `scripts/validate_env.py` | Obrigatório | ✅ |
| R12 | Instalação limpa | Obrigatório | ✅ |
| R13 | Dockerfile multi-stage | Obrigatório | ✅ |
| R14 | docker-compose treino + MLflow | Obrigatório | ✅ |
| R15 | DVC init + remote + dataset versionado | Obrigatório | 🟡 |
| R16 | Pipeline DVC ≥ 3 stages | Obrigatório | ✅ |
| R17 | MLflow params/métricas/artefatos | Obrigatório | ✅ |
| R18 | ≥ 3 runs rastreados | Obrigatório (critério) | ✅ |
| R19 | Registry Staging → Production | Obrigatório | ✅ |
| R20 | MLP/embedding PyTorch | Obrigatório | ✅ |
| R21 | Early stopping | Obrigatório (critério rede neural) | ✅ |
| R22 | Baselines sklearn + ≥ 4 métricas | Obrigatório | ✅ |
| R23 | Model Card | Obrigatório | ✅ |
| R24 | README completo | Obrigatório | ✅ |
| R25 | Vídeo STAR ≤ 5 min | Obrigatório | 🔴 |
| R26 | Deploy nuvem | Opcional | 🔴 |
| R27 | Libs PyTorch, sklearn, MLflow, DVC | Obrigatório | ✅ |
| R28 | Seeds fixados | Obrigatório (boas práticas) | ✅ |
| R29 | BERTopic / features de conteúdo no modelo | Plano interno | 🔴 / ➖ |
| R30 | CI GitHub Actions | Boa prática | 🔴 / ➖ |

---

## 5. Arquitetura e funcionamento atual

### Fluxo observado (entrada → saída)

```text
data/raw/{rating(s),movie(s),link(s)}.csv
        │
        ▼
scripts/preprocess.py  ──Strategy──► preprocessed_ratings.parquet
        │
        ▼
scripts/enrich_metadata.py + movie_metadata.parquet (TMDB pré-coletado)
        │
        ▼
enriched_metadata.parquet
        │
        ▼
scripts/feature_engineering.py  ──► features_ratings.parquet (user_idx, movie_idx)
        │
        ▼
scripts/train.py (PyTorch + early stopping + MLflow)
        │
        ▼
models/model.pth
        │
        ▼
scripts/evaluate.py (baselines + @K + Registry + metrics.json)
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
| Demo | `llm/` | recomendação a partir de histórico (não é LLM generativo) |
| Governança IA | `.cursor/`, `.github/` (espelho) | regras/comandos para agentes |

### Relação entre partes

O **coração avaliável do PDF** é: Factory de modelos + treino PyTorch + evaluate com baselines/métricas + DVC + Docker + MLflow.  
A coleta TMDB alimenta `movie_metadata.parquet` e entra em `enrich_metadata`, mas `feature_eng` **quase não usa** campos de conteúdo no treino neural — só mapeia IDs. O modelo atual é **fortemente colaborativo**.

### Divergência workspace vs Git (fato crítico)

| Artefato | No Git / `dvc.lock` | Na máquina local (workspace) |
|----------|--------------------|------------------------------|
| Ratings brutos | `ratings.csv` ~123 bytes (smoke) | `rating.csv` ~690 MB (~20M linhas) |
| Features | ~5 KB no lock | ~217 MB, **19.761.138** linhas |
| `models/model.pth` | ignorado pelo Git; lock aponta smoke | ~21 MB (treino real, val_mse ~0,75) |
| `metrics.json` | smoke: RMSE ~1,7–2,25; @K = 0; champion RF | não regenerado no Git após treino local |

**Inferência:** alguém treinou o MovieLens completo localmente, mas **não regenerou/committou** `dvc.lock` + `metrics.json` coerentes com esse run. A banca que clonar o repo verá o snapshot smoke, não o treino grande.

---

## 6. Auditoria de Aderência ao Tech Challenge

### R01 — Estrutura de pastas

**Status:** ✅ Atendido  
**Evidência:** `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`.  
**O que falta:** nada crítico.

### R02 / R03 — Clean code e funções ≤ 20 linhas

**Status:** 🟡 Parcial  
**Evidência (AST):** 38 funções com > 20 linhas, destacando:

- `scripts/train.py::main` (~130 linhas)
- `scripts/evaluate.py::main` (~77), `_run_baselines` (~57)
- `scripts/generate_llm_report_pdf.py::build` (~206)
- `src/data/external/tmdb_client.py::_request_json` (~60)

**Análise:** espírito de clean code presente em `src/`; a regra explícita ≤ 20 linhas **não é cumprida** de forma consistente nos scripts.  
**O que falta:** extrair helpers nos scripts longos (sem overengineering).

### R04 — Design patterns

**Status:** ✅ Atendido  
**Evidência:**

- Factory: `src/models/factory.py` (`create_model`)
- Strategy: `src/data/preprocessors/` + `src/evaluation/metric_strategy.py`

**O que falta:** nada obrigatório.

### R05 — Type hints e docstrings Google

**Status:** 🟡 Parcial  
**Evidência:** APIs públicas em `src/` tipadas; scripts nem sempre; `fit`/`predict` com tipagem frouxa (`Any`).  
**O que falta:** homogeneizar scripts e interfaces de modelo.

### R06 — Ruff + pre-commit

**Status:** ✅ Atendido  
**Evidência:** `.pre-commit-config.yaml`; `ruff check src scripts tests configs` na `.venv` → **All checks passed**.

### R07 — pyproject + lock

**Status:** ✅ Atendido  
**Evidência:** `pyproject.toml` (prod + `dev`/`s3`/`gdrive`), `uv.lock`.  
**Nota:** PDF cita `poetry install`; o projeto usa **uv** (também permitido no PDF).

### R08 — dockerignore / gitignore / env.example

**Status:** ✅ Atendido  
**Evidência:** arquivos na raiz; `.env` no `.gitignore`.

### R09 — Commits semânticos

**Status:** ✅ Atendido  
**Evidência:** histórico com `feat:`, `fix:`, `docs:`, `test:`, `chore:`.

### R10 / R11 — Settings + validate_env

**Status:** ✅ Atendido  
**Evidência:** `configs/settings.py`, `scripts/validate_env.py`.  
**Observação:** `validate_env.py` importa `packaging.version`, **não declarado diretamente** em `pyproject.toml` (dependência transitiva — frágil).

### R12 — Instalação limpa

**Status:** ✅ Atendido  
**Evidência:** README + `docs/DOCUMENTACAO_ETAPA2.md`; commit `c528344` (“validating clean install in a new venv”); docs afirmam 25/25 checks.

### R13 / R14 — Docker

**Status:** ✅ Atendido (com ressalvas)  
**Evidência:** `Dockerfile` multi-stage; `docker-compose.yml` com `train` + `mlflow`.  
**Ressalvas:**

- Compose monta `.:/app` (depende do host).
- Imagem não copia `dvc.yaml`/`params.yaml` no build (mitigado pelo volume).
- `CMD` default = `validate_env.py` (treino via compose).

### R15 / R16 — DVC

**Status:** 🟡 Parcial (versionamento) / ✅ (≥ 3 stages)  
**Evidência:**

- `dvc.yaml`: 5 stages — **≥ 3 ✅**
- `.dvc/config`: remote `local_remote` → `../data/dvc_remote`
- `dvc.lock` presente

**Problemas:**

- Neste ambiente, `../data/dvc_remote` **não existia**.
- `dvc.lock` aponta CSVs smoke (~123 bytes).
- `dvc.yaml` declara `data/raw/ratings.csv` (plural), enquanto Kaggle local usa `rating.csv` — o código Python resolve ambos (`movielens_io.py`), mas o **stage DVC** espera o nome plural.

**O que falta:** regenerar lock com dataset de entrega + remote acessível + alinhar nomes dos deps no `dvc.yaml`.

### R17 / R18 — MLflow tracking e ≥ 3 runs

**Status:** ✅ Atendido  
**Evidência:** `scripts/train.py` e `scripts/evaluate.py` logam params/métricas/artefatos; `metrics.json` lista 4 candidatos com `run_id` (MostPopular, KNN, RF, torch_mlp).

### R19 — Model Registry Staging → Production

**Status:** ✅ Atendido  
**Evidência:** `src/evaluation/registry.py` → `promote_champion()` (Staging → Production se métricas válidas).  
**Observação de negócio/nota:** no snapshot versionado, o champion em Production é **`sklearn_random_forest`**, não o MLP. O PDF pede rede neural como modelo **central**; ter RF como Production é defensável se o vídeo explicar a comparação, mas pode confundir a banca.

### R20 / R21 — PyTorch + early stopping

**Status:** ✅ Atendido  
**Evidência:** `src/models/torch_mlp.py`, `torch_embedding.py`; early stopping em `scripts/train.py` (`patience`).

### R22 — Baselines + ≥ 4 métricas

**Status:** ✅ Atendido (com ressalva de qualidade do snapshot)  
**Evidência:** MostPopular + KNN + RandomForest; métricas RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate.  
**Ressalva:** no `metrics.json` commitado, baselines têm só RMSE/MAE; ranking @K do torch está **0,0** (smoke).

### R23 / R24 — Model Card + README

**Status:** ✅ Atendido  
**Evidência:** `docs/MODEL_CARD.md` (honesta sobre snapshot); `README.md` com setup, DVC, Docker.

### R25 — Vídeo STAR

**Status:** 🔴 Não atendido  
**Evidência:** nenhum `.mp4`/`.mov` no repo; `TODO.md` item 6.5 aberto.  
**O que falta:** gravar e entregar o vídeo (fora do código).

### R26 — Deploy nuvem

**Status:** 🔴 Não atendido (opcional)  
**Evidência:** README com placeholder Render; `src/serving/` vazio.

### R29 — BERTopic (plano interno)

**Status:** 🔴 / ➖  
**Evidência:** deps em `pyproject.toml`; **zero** `import bertopic` no código de pipeline; `src/features/__init__.py` stub.  
**Análise:** **não** é requisito literal do PDF.

---

## 7. Auditoria de Refactor e Estrutura

A estrutura atual **é adequada** para entregar o Tech Challenge. Não precisa de microserviços nem de reescrita.

| Problema atual | Onde | Impacto | Refactor recomendado | Prioridade |
|----------------|------|---------|----------------------|------------|
| Scripts monolíticos | `scripts/train.py`, `evaluate.py` | Dificulta cumprir ≤ 20 linhas e revisão | Extrair loaders, loops e logging MLflow para funções/módulos pequenos | Alta (nota clean code) |
| `feature_eng` só mapeia IDs | `scripts/feature_engineering.py` | Metadados TMDB não entram no modelo | Ou integrar features de conteúdo, ou documentar honestamente “colaborativo puro” no STAR | Média |
| Stubs vazios | `src/features/`, `src/serving/` | Ruído / expectativa falsa | Preencher ou remover da narrativa de arquitetura | Baixa |
| Champion por RMSE só | `scripts/evaluate.py` L369 | RF vence MLP no smoke; ranking @K ignorado na promoção | Critério multi-métrica ou forçar neural como modelo “oficial” + RF como baseline | Média (narrativa) |
| Duplicação RMSE/MAE | `metrics.py` vs `metric_strategy.py` | Confusão leve | Unificar em um módulo | Baixa |
| Comentários scaffold | `feature_engineering.py` L34–49 | Cheiro de template | Remover “Example:” / “In a real scenario” | Baixa |
| Espelho `.cursor` / `.github` | pastas de governança | Manutenção duplicada | Manter uma fonte; não bloqueia entrega | Baixa |

**O que NÃO vale a pena refatorar agora:**

- Factory/Strategy já existentes.
- Cliente TMDB + cache DVC.
- Seeds + Settings + validate_env.
- Métricas de ranking em `src/evaluation/metrics.py`.
- Model Card atual (já é honesta).

---

## 8. Auditoria de Limpeza do Projeto

| Arquivo/Componente | Motivo | Evidência | Recomendação |
| ------------------ | ------ | --------- | ------------ |
| `scripts/hello_train.py` | Scaffold Docker antigo | Não está no `dvc.yaml`; só imprime paths | **Seguro para remover** (após confirmar que ninguém usa no compose) |
| `src/features/__init__.py` | Stub | Só docstring; sem imports | **Manter** como placeholder OU documentar “futuro” — não remover se o time ainda planeja BERTopic |
| `src/serving/__init__.py` | Stub | Sem FastAPI | **Manter** se bônus cloud for perseguido; senão pode sumir da narrativa |
| `bertopic` / `sentence-transformers` | Deps pesadas sem uso | Nenhum import no pipeline | **Provavelmente removível** das deps de prod (ou `optional`) — validar com o time |
| `docs/CURSOR_STRUCTURE_REPORT.md`, `CURSOR_REFACTOR_REPORT.md` | Históricos de scaffold | Descrevem estado pré-código | **Provavelmente removível** da entrega à banca (ou mover para pasta interna) |
| `.cursor/.setai/` | Resíduo SetAI | README próprio | **Provavelmente removível** |
| `llm/` + `scripts/generate_llm_report_pdf.py` | Demo extra | Fora do DVC; útil para STAR | **Manter** se usado no vídeo; senão opcional |
| Caches `__pycache__`, `.pytest_cache`, `.ruff_cache` | Gerados | Já no `.gitignore` | **Manter ignorados** (não commitar) |
| `.env` local | Secrets | Gitignored; `TMDB_API_KEY` preenchida localmente | **Manter fora do Git** |
| Dados locais `data/raw/*.csv` grandes | Dataset | Gitignored; ok | **Manter local + DVC**; não commitiar CSV |

---

## 9. Auditoria de Possíveis Resíduos ou Erros de Código Gerado por IA

### 9.1 Comentários e textos inadequados

- **Não** há `Generated by ChatGPT` / `Copilot` dentro de `src/`.
- Sinais concretos de scaffold:
  - `scripts/feature_engineering.py`: comentários `# Example:` e `# (In a real scenario, use a specific artifact folder)`.
  - `scripts/hello_train.py`: “hello train (scaffold)”.
- Docs `CURSOR_*` descrevem regeneração via SetAI/Cursor — histórico de governança, não código de produção.
- `llm/examples/prompt_avaliacao_imparcial.md` menciona ChatGPT/Claude como **ferramenta de avaliação humana** — uso consciente, não resíduo escondido.

### 9.2 Código suspeito

- `TorchMLPRecommender.fit()` só seta `_fitted = True` (treino real está em `scripts/train.py`) — interface ABC cumprida de forma superficial.
- Champion selection só por RMSE pode promover baseline sklearn em Production.
- Stubs `features/` e `serving/` criam arquitetura “prometida” sem implementação.
- Fallback silencioso de split aleatório se não houver `timestamp` (`src/data/splits.py`) — ok se documentado; risco se o dataset perder a coluna.

### 9.3 Dependências

Ver seção 10.

### 9.4 Segurança

- `.env` **não** está no Git (confirmado via `git check-ignore`).
- `.env.example` sem secrets.
- Localmente `.env` contém `TMDB_API_KEY` preenchida (comprimento 32) — **valor não reproduzido neste relatório**.
- `DATABASE_URL` local preenchida — garantir que não vaze em screenshots/logs do vídeo.
- Compose usa `.env.example` no serviço `train` — adequado para demo sem secrets.

---

## 10. Dependências e Segurança

| Item | Classificação | Motivo |
|------|---------------|--------|
| `torch`, `scikit-learn`, `mlflow`, `dvc` | Adequadas | Exigidas pelo PDF |
| `bertopic`, `sentence-transformers` | Possivelmente desnecessárias (prod) | Declaradas, não usadas no código |
| `packaging` | Frágil | Usado em `validate_env.py` sem declaração direta |
| Versões no `uv.lock` | Atualizadas o suficiente | Não se classifica como vulnerável sem scan CVE |
| Secrets no Git | Não encontrados | `.env` ignorado; example vazio |
| CI vulnerabilidades | ⚪ Não comprovado | Sem workflow de Dependabot/audit nesta auditoria |

**Desatualizada ≠ Vulnerável.** Nenhum CVE foi comprovado com ferramenta nesta auditoria.

---

## 11. Qualidade de Código

| Aspecto | Avaliação |
|---------|-----------|
| Legibilidade em `src/` | Boa |
| Consistência scripts vs libs | Média (scripts longos) |
| Nomenclatura | Boa e descritiva |
| SOLID / patterns | Adequado ao desafio |
| Tipagem | Boa em `src/`; frouxa em partes |
| Erros / logging | Aceitável (prints + MLflow) |
| Manutenibilidade | Boa para projeto acadêmico |
| Magic numbers | Alguns em `evaluate.py` (`_K=10`, caps de sample) — ok se documentados |

Contexto acadêmico: **não precisa** de padrão enterprise completo. Precisa ser justificável e alinhado ao PDF.

---

## 12. Testes e Confiabilidade

**Fatos observados nesta auditoria:**

- `pytest tests/unit` na `.venv`: **52 testes, todos passando**.
- `ruff check`: **sem erros**.
- Cobertura: `pytest-cov` declarado; **cobertura percentual não medida** nesta auditoria (⚪).

| Área | Cobertura de testes | Observação |
|------|---------------------|------------|
| Métricas @K / RMSE | Sim (`test_metrics.py`) | Alinhado ao PDF |
| Factory / MostPopular | Sim | |
| Preprocessors / splits | Sim | Temporal order testado |
| TMDB client (mock) | Sim | |
| Pipeline DVC end-to-end | Não (integração real limitada) | `tests/integration/test_real_inference.py` existe; não foi o foco desta rodada unitária |
| Registry Staging→Prod | Não unitário dedicado | Lógica em `registry.py` sem teste isolado encontrado |
| train.py / evaluate.py | Pouco / indireto | Maior risco de regressão |

**Conclusão:** testes são **suficientes para o núcleo de domínio/métricas** do desafio, mas **não garantem** sozinhos o `dvc repro` completo nem a qualidade do run de entrega.

---

## 13. Documentação

| Documento | Status | Observação |
|-----------|--------|------------|
| `README.md` | ✅ Bom | Setup, DVC, Docker; admite snapshot smoke |
| `docs/MODEL_CARD.md` | ✅ Bom | Alinhado ao `metrics.json` smoke |
| `docs/DOCUMENTACAO_ETAPA2.md` | ✅ | Reprodutibilidade |
| `docs/AUDITORIA_DESAFIO.md` | 🟡 | Checklist útil; parcialmente desatualizado vs HEAD |
| `TODO.md` | 🟡 | Rico; mistura obrigação PDF e desejos (BERTopic) |
| `docs/CURSOR_*` | 🟡 Obsoleto para banca | Histórico de scaffold |
| Vídeo STAR | 🔴 Ausente | |

**Risco:** docs prometem MovieLens 20M; artefatos Git mostram smoke. O README já avisa — reforçar no vídeo.

---

## 14. Pontos Positivos

- Estrutura `src/` clara e alinhada ao enunciado — **não há necessidade clara de reorganizar pastas**.
- Factory + Strategy realmente implementados e usados.
- Pipeline DVC com 5 stages conectados de ponta a ponta.
- Early stopping + seeds + MLflow logging no treino.
- Comparação neural vs baselines com ≥ 4 métricas no código.
- Registry com promoção Staging→Production automatizada.
- Model Card honesta sobre limitações e snapshot fraco.
- TMDB com cache versionável (não chama API a cada `dvc repro`).
- Ruff + pytest verdes na auditoria.
- Commits semânticos e `.env.example` corretos.
- Analogia e-commerce bem explicada no README.

---

## 15. Problemas P0

1. **Vídeo STAR obrigatório ausente** — pode zerar 10% da nota.  
2. **Artefatos de evidência no Git desalinhados** (`dvc.lock` smoke + `metrics.json` com @K=0 e champion RF) — risco de a banca avaliar um pipeline “de brinquedo” mesmo com treino local grande.

---

## 16. Problemas P1

1. Funções ≫ 20 linhas (regra explícita do PDF / clean code 15%).  
2. `dvc.yaml` fixa `ratings.csv` enquanto dados Kaggle usam `rating.csv`.  
3. Remote DVC `../data/dvc_remote` inexistente neste ambiente.  
4. Champion Production = Random Forest no snapshot versionado (narrativa da rede neural enfraquecida).  
5. Metadados TMDB não entram no forward do modelo (docs/TODO prometem mais).  
6. `packaging` não declarado diretamente.

---

## 17. Problemas P2

1. Deps BERTopic/sentence-transformers sem uso.  
2. Stubs `features/` e `serving/`.  
3. Ausência de CI GitHub Actions (boa prática, não PDF).  
4. Dockerfile sem copiar `dvc.yaml`/`params.yaml` (ok com volume).  
5. Testes fracos para `train`/`evaluate`/Registry.

---

## 18. Problemas P3

1. Remover `hello_train.py` e docs Cursor históricos.  
2. Unificar RMSE/MAE duplicados.  
3. Limpar comentários scaffold em `feature_engineering.py`.  
4. Deploy cloud / FastAPI (bônus opcional).

---

## 19. Plano de Ação Recomendado

| # | Prioridade | Problema | Ação recomendada | Arquivos | Motivo | Impacto |
|---|------------|----------|------------------|----------|--------|--------|
| 1 | P0 | Sem vídeo STAR | Roteiro Situation/Task/Action/Result + gravar ≤ 5 min | fora do repo / link na entrega | 10% da nota | Desbloqueia entrega |
| 2 | P0 | Evidência Git = smoke | Rodar evaluate no dataset de entrega; atualizar `metrics.json`, Model Card e, se aplicável, `dvc.lock` | `metrics.json`, `docs/MODEL_CARD.md`, `dvc.lock` | Banca vê números reais | Credibilidade |
| 3 | P1 | Nomes CSV DVC | Alinhar deps do `dvc.yaml` aos nomes reais (ou gerar symlinks/cópias `ratings.csv`) | `dvc.yaml`, `data/raw/` | `dvc repro` quebra em Kaggle puro | Reprodutibilidade |
| 4 | P1 | Clean code ≤ 20 linhas | Quebrar `train.py` / `evaluate.py` em helpers | `scripts/` | Critério 15% | Nota clean code |
| 5 | P1 | Champion RF | Explicar no STAR **ou** ajustar critério de promoção / destacar MLP como modelo central | `evaluate.py`, Model Card, vídeo | Evitar confusão | Nota neural + vídeo |
| 6 | P2 | Deps mortas | Mover BERTopic para optional ou implementar mínimo | `pyproject.toml` | Instalação mais leve | Reprodutibilidade |
| 7 | P2 | Remote DVC | Documentar criação do remote local ou S3 | `.dvc/config`, README | `dvc pull` funciona | Etapa 3 |
| 8 | P3 | Limpeza | Remover `hello_train`, docs Cursor velhos | scripts/docs | Menos ruído | Polimento |

**IMPORTANTE:** este plano é recomendação. **Nenhuma ação foi executada nesta auditoria.**

---

## 20. Checklist Final para Entrega

- [ ] Vídeo STAR ≤ 5 min gravado e linkado na entrega
- [ ] `metrics.json` + Model Card refletem o run que a banca deve acreditar
- [ ] `dvc repro` funciona em máquina limpa (ou documentar smoke vs full)
- [ ] `uv sync` + `validate_env.py` OK
- [ ] `docker compose` sobe `mlflow` e `train`
- [ ] Ruff + pytest verdes
- [ ] Registry mostra Staging→Production (print no vídeo)
- [ ] README com passos reais (uv, não só poetry)
- [ ] Nenhum `.env` commitado
- [ ] Decidir se BERTopic entra ou sai da narrativa
- [ ] (Opcional) Deploy nuvem + URL pública

---

## 21. Conclusão

### Respostas diretas

1. **O projeto atende ao Tech Challenge?** Quase: **sim no código técnico principal**; **não completa a entrega** sem o vídeo STAR e com artefatos Git ainda em modo smoke.  
2. **Quanto está atendido?** Estimativa **~78%** dos critérios ponderados do PDF.  
3. **O que falta?** Vídeo STAR; sincronizar evidências (`metrics`/`dvc.lock`/narrativa); reforçar clean code nos scripts.  
4. **Algo estruturalmente errado?** Não de forma grave — a pasta `src/` e o pipeline fazem sentido.  
5. **O que vale refatorar?** Scripts longos; alinhamento DVC/nomes CSV; critério de champion / narrativa neural.  
6. **O que NÃO vale?** Reescrever Factory/Strategy, TMDB client, métricas core, Settings.  
7. **Remover?** `hello_train.py`, docs Cursor históricos, deps BERTopic até uso real (validar).  
8. **Resíduos de IA?** Comentários scaffold e docs SetAI; sem “Generated by ChatGPT” no `src/`.  
9. **Deps problemáticas?** BERTopic não usado; `packaging` transitivo — sem CVE comprovado.  
10. **Segurança?** `.env` local com API key (ok se não versionado); não commitir secrets.  
11. **Testes suficientes?** Sim para domínio/métricas; fracos para orquestração completa.  
12. **Docs corretas?** Em geral sim; honestas sobre smoke; TODO mistura desejo vs obrigação.  
13. **Antes da entrega?** Vídeo + evidências coerentes + checagem `dvc repro`/`docker`.  
14. **Ordem?** STAR → sincronizar métricas/lock → DVC nomes/remote → clean code scripts → limpeza.  
15. **Riscos de perder pontos:** ausência do vídeo (10%); clean code parcial (15%); DVC/repro frágil (15%); narrativa com champion RF e @K=0 no snapshot Git.

**Classificação final:** **Bem alinhado** — praticamente pronto no código; **não** “pronto para entregar” até fechar o vídeo STAR e alinhar as evidências versionadas ao run que o grupo quer apresentar.

---

*Auditoria somente leitura. Única alteração permitida e realizada: criação/atualização deste arquivo `AUDITORIA_TECH_CHALLENGE.md`.*
