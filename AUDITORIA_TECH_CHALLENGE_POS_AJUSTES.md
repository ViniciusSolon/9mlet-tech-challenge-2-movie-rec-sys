# Auditoria Técnica Completa — Pós-Ajustes (Tech Challenge Fase 02)

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys`  
**Fonte de requisitos:** `docs/Tech Challenge Fase 02.pdf` (`docs/pdf_extract.txt`)  
**Data da auditoria:** 2026-07-25  
**Escopo:** estado **atual do working tree** após correções pós-auditoria (muitas ainda **não commitadas** no HEAD `c528344`)  
**Checagens não destrutivas:** `ruff check` (OK), `pytest tests/unit` (60 OK)  
**Regra:** nenhum código foi alterado nesta tarefa; única entrega = este arquivo.

**Arquivo irmão:** `AUDITORIA_TECH_CHALLENGE.md` (auditoria anterior) · `docs/RELATORIO_MUDANCAS_POS_AUDITORIA.md` (changelog das correções).

---

## 1. Resumo Executivo

Após os ajustes, o projeto está **muito bem alinhado** ao Tech Challenge no código. Quase todos os requisitos técnicos obrigatórios do PDF estão implementados e comprováveis:

- estrutura clean code + Factory + Strategy;
- `pyproject.toml` + `uv.lock` (Poetry/uv) com deps prod/dev;
- Docker multi-stage + compose (treino + MLflow);
- DVC com 5 stages + remote `./dvc-storage` + aliases Kaggle/GroupLens;
- MLP PyTorch com early stopping, baselines sklearn, ≥ 4 métricas;
- MLflow tracking + Registry Staging→Production com **champion `torch_mlp`**;
- Model Card e README atualizados;
- ruff limpo e 60 testes unitários passando.

**Único gap obrigatório claro de entrega:** o **vídeo STAR (10% da nota)** — não há arquivo de vídeo no repositório.

**Ressalva importante:** o `dvc.lock` versionado ainda descreve um **smoke test minúsculo** (CSVs ~100 bytes), enquanto `metrics.json` e o workspace local refletem evaluate no MovieLens completo. A banca que olhar só o lock pode achar o pipeline “de brinquedo”, mesmo com métricas reais no JSON.

**Veredito:** tecnicamente **praticamente pronto para entrega**; falta gravar/publicar o vídeo STAR e, de preferência, alinhar/commitar `dvc.lock` + mudanças pendentes.

---

## 2. Nota / Aderência Geral

### Estimativa técnica de aderência: **~88%**

Não é a nota oficial da FIAP. É uma estimativa ponderada pelos critérios do PDF (bônus de nuvem fora do obrigatório).

| Critério (PDF) | Peso | Estimativa pós-ajustes | Contribuição |
|----------------|------|------------------------|--------------|
| Clean code e estrutura | 15% | ~85% | 12,8 |
| Reprodutibilidade | 15% | ~95% | 14,3 |
| Docker | 15% | ~90% | 13,5 |
| DVC + Pipeline | 15% | ~85% | 12,8 |
| Rede neural (PyTorch) | 15% | ~95% | 14,3 |
| MLflow + Registry | 10% | ~95% | 9,5 |
| Vídeo STAR | 10% | ~0% | 0,0 |
| **Subtotal obrigatório** | **95%** | — | **~77,2 / 95 ≈ 81%** |
| Bônus cloud | 5% | ~0% | 0 |

**Ajuste qualitativo (+7 pp):** evidências fortes em `metrics.json` (MLP champion, HR@10=0,61), CI GitHub Actions, aliases DVC, validate_env 27/27, limpeza de scaffold → **~88%**.

### Classificação

**Muito bem alinhado** — próximo de **“Praticamente pronto para entrega”** no código; a entrega FIAP completa ainda exige o vídeo STAR.

**Comparação com a auditoria anterior (~78%):** ganho principal em champion neural, métricas reais, deps, remote DVC, Docker completo, testes/CI e limpeza. O teto sem vídeo continua limitado pelos 10% do STAR.

---

## 3. Objetivos identificados no Tech Challenge

### Objetivo principal (obrigatório)

Sistema de recomendação (analogia e-commerce) com:

- rede neural PyTorch (MLP ou embedding);
- Docker;
- DVC;
- MLflow;
- clean code profissional.

### Entregáveis

| Entregável | Natureza |
|------------|----------|
| Repositório GitHub | **Obrigatório** |
| Vídeo ≤ 5 min (STAR) | **Obrigatório** |
| Deploy nuvem | **Opcional** (+5%) |

### Dataset

MovieLens (ou e-commerce) com ≥ 10.000 interações — **atendido** pela escolha MovieLens 20M.

### Diferenciação

- **Obrigatório:** texto explícito do PDF.
- **Recomendado:** dataset sugerido, boas práticas sem peso isolado.
- **Boa prática / plano interno:** BERTopic, FastAPI, ablation — **não** exigidos literalmente.

---

## 4. Matriz de Requisitos

Legenda: ✅ Atendido · 🟡 Parcial · 🔴 Não atendido · ⚪ Não comprovado · ➖ N/A

| # | Requisito | Tipo | Status |
|---|-----------|------|--------|
| R01 | Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/` | Obrigatório | ✅ |
| R02 | Clean code SOLID / naming / type hints | Obrigatório | ✅ |
| R03 | Funções ≤ 20 linhas | Obrigatório (boas práticas) | 🟡 |
| R04 | ≥ 1 design pattern (Factory/Strategy) | Obrigatório | ✅ |
| R05 | Type hints + docstrings Google | Obrigatório | ✅ |
| R06 | Ruff sem erros + pre-commit | Obrigatório | ✅ |
| R07 | `pyproject.toml` Poetry/uv + lock | Obrigatório | ✅ |
| R08 | `.dockerignore` / `.gitignore` / `.env.example` | Obrigatório | ✅ |
| R09 | Commits semânticos | Obrigatório | ✅ |
| R10 | `.env` + Pydantic Settings | Obrigatório | ✅ |
| R11 | `scripts/validate_env.py` | Obrigatório | ✅ |
| R12 | Instalação limpa | Obrigatório | ✅ |
| R13 | Dockerfile multi-stage | Obrigatório | ✅ |
| R14 | docker-compose treino + MLflow | Obrigatório | ✅ |
| R15 | DVC init + remote + dataset versionado | Obrigatório | 🟡 |
| R16 | Pipeline DVC ≥ 3 stages | Obrigatório | ✅ |
| R17 | MLflow params/métricas/artefatos | Obrigatório | ✅ |
| R18 | ≥ 3 runs rastreados | Obrigatório | ✅ |
| R19 | Registry Staging → Production | Obrigatório | ✅ |
| R20 | MLP/embedding PyTorch | Obrigatório | ✅ |
| R21 | Early stopping | Obrigatório | ✅ |
| R22 | Baselines + ≥ 4 métricas | Obrigatório | ✅ |
| R23 | Model Card | Obrigatório | ✅ |
| R24 | README completo | Obrigatório | ✅ |
| R25 | Vídeo STAR | Obrigatório | 🔴 |
| R26 | Deploy nuvem | Opcional | 🔴 |
| R27 | Libs PyTorch, sklearn, MLflow, DVC | Obrigatório | ✅ |
| R28 | Seeds fixados | Obrigatório | ✅ |
| R29 | BERTopic no modelo | Plano interno | ➖ |
| R30 | CI GitHub Actions | Boa prática | ✅ |

---

## 5. Arquitetura e funcionamento atual

### Fluxo (entrada → saída)

```text
data/raw/{rating(s)|movie(s)|link(s)}.csv
        │  prepare_raw_aliases (GroupLens ↔ Kaggle)
        ▼
preprocess (Strategy explicit/implicit)
        ▼
enrich_metadata + movie_metadata.parquet (TMDB cache)
        ▼
feature_eng (user_idx/movie_idx + colunas de conteúdo opcionais)
        ▼
train (Factory → torch_mlp/embedding + early stopping + MLflow)
        ▼
models/model.pth
        ▼
evaluate (baselines + @K + select_champion torch_* + Registry)
        ▼
metrics.json  ·  Staging → Production
```

### Componentes principais

| Área | Caminho | Papel |
|------|---------|--------|
| Domínio | `src/domain/` | IDs, Rating, Recommendation |
| Dados | `src/data/` | splits, preprocessors, TMDB, `raw_aliases` |
| Modelos | `src/models/` | Factory, MLP, Embedding, baselines |
| Treino | `src/training/` | seeds + `loop.py` |
| Avaliação | `src/evaluation/` | métricas, runner, champion, registry |
| Config | `configs/settings.py`, `params.yaml` | env + hiperparâmetros |
| Pipeline | `dvc.yaml`, `scripts/` | stages DVC |
| Demo | `llm/`, `scripts/demo_recommend.py` | demos (não são requisito PDF) |
| CI | `.github/workflows/ci.yml` | ruff + pytest |

### Relação importante

O modelo neural continua **colaborativo** (`user_idx`/`movie_idx`). Metadados TMDB entram no enrich e podem ser anexados no feature parquet, mas o forward do MLP não usa sinopse/BERTopic — e isso **não viola** o PDF.

### Divergência workspace vs `dvc.lock` (fato)

| Artefato | Observado |
|----------|-----------|
| `metrics.json` | Run real: torch_mlp champion, RMSE≈0,78, HR@10=0,61 |
| Workspace | `rating.csv` ~690 MB; features ~217 MB; `model.pth` ~21 MB |
| `dvc.lock` | Ainda smoke: `ratings.csv` size **123** bytes |

**Inferência:** evaluate foi regenerado no dataset completo, mas o lock DVC não foi regenerado/commitado coerente com esse run.

---

## 6. Auditoria de Aderência ao Tech Challenge

### R01 — Estrutura

**Status:** ✅  
**Evidência:** `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`.

### R02 / R05 — Clean code, hints, docstrings

**Status:** ✅ (com ressalva R03)  
**Evidência:** módulos `src/` tipados; Factory/Strategy; helpers novos com docstrings Google (`champion.py`, `loop.py`, `runner.py`).

### R03 — Funções ≤ 20 linhas

**Status:** 🟡 Parcial  
**Evidência (AST):** ainda **38 funções > 20 linhas**, ex.:

- `scripts/generate_llm_report_pdf.py::build` (~206)
- `scripts/demo_recommend.py::main` (~79)
- `scripts/evaluate.py::main` (~49), `train.py::main` (~47)

**Análise:** melhorou vs auditoria anterior (mains de train/evaluate eram ~130/~77), mas a regra explícita do PDF ainda não é 100%.  
**O que falta:** quebrar entrypoints auxiliares (`generate_llm_report_pdf`, demos) se a banca for rigorosa na contagem de linhas.

### R04 — Design patterns

**Status:** ✅  
**Evidência:** `src/models/factory.py`; `src/data/preprocessors/`; `src/evaluation/metric_strategy.py`.

### R06 — Ruff + pre-commit

**Status:** ✅  
**Evidência:** `.pre-commit-config.yaml`; nesta auditoria `ruff check src scripts tests configs` → **All checks passed**.

### R07–R12 — Reprodutibilidade

**Status:** ✅  
**Evidência:** `pyproject.toml` (prod + `dev`/`topics`/`s3`), `uv.lock`, `configs/settings.py`, `.env.example`, `scripts/validate_env.py`, docs de instalação limpa, `[tool.uv] package = true`, `packaging` declarado.

### R13–R14 — Docker

**Status:** ✅  
**Evidência:** `Dockerfile` multi-stage (copia `dvc.yaml`/`params.yaml`/`.dvc`); `docker-compose.yml` com `train` + `mlflow`.  
**Ressalva:** compose monta `.:/app` (depende do host).

### R15–R16 — DVC

**Status:** 🟡 / ✅  
**Evidência:** 5 stages em `dvc.yaml`; remote `./dvc-storage`; aliases via `prepare_raw_aliases`.  
**Parcial em R15:** `dvc.lock` ainda é snapshot smoke e está **desalinhado** do `dvc.yaml` atual (cmds antigos no lock).  
**O que falta:** `dvc repro` + commit de `dvc.lock` coerente (smoke documentado **ou** dataset de entrega).

### R17–R19 — MLflow + Registry

**Status:** ✅  
**Evidência:** 4 candidatos em `metrics.json`; `MLflowRegistryManager`; champion:

```json
"champion": { "name": "torch_mlp", "version": 2, "stage": "production" }
```

`src/evaluation/champion.py` prioriza modelos `torch_*`.

### R20–R22 — Neural + baselines + métricas

**Status:** ✅  
**Evidência:** `torch_mlp` / `torch_embedding`; early stopping em `src/training/loop.py`; MostPopular + KNN + RF; RMSE, MAE, Precision@10, Recall@10, NDCG@10, Hit Rate@10.

### R23–R24 — Model Card + README

**Status:** ✅  
**Evidência:** `docs/MODEL_CARD.md` alinhado ao run; `README.md` com `uv sync --extra dev`, DVC, Docker, MLflow SQLite.

### R25 — Vídeo STAR

**Status:** 🔴  
**Evidência:** nenhum `.mp4`/`.mov` encontrado; `TODO.md` ainda marca vídeo pendente.  
**O que falta:** gravar e entregar o vídeo (fora do código).

### R26 — Deploy nuvem

**Status:** 🔴 (opcional)  
**Evidência:** placeholder no README; `src/serving/` só docstring.

### R29 — BERTopic

**Status:** ➖  
**Evidência:** extra `topics` no `pyproject.toml`; **zero** `import bertopic` no pipeline. Não é requisito do PDF.

### R30 — CI

**Status:** ✅ (boa prática)  
**Evidência:** `.github/workflows/ci.yml` (ruff + pytest).

---

## 7. Auditoria de Refactor e Estrutura

A estrutura **é adequada** ao desafio. Não precisa de microserviços.

| Problema atual | Onde | Impacto | Refactor recomendado | Prioridade |
|----------------|------|---------|----------------------|------------|
| Funções > 20 linhas em demos/PDF | `generate_llm_report_pdf.py`, `demo_recommend.py` | Pode custar pontos de clean code | Extrair helpers ou aceitar como scripts auxiliares fora do núcleo | Média |
| `dvc.lock` desatualizado | `dvc.lock` vs `dvc.yaml`/`metrics.json` | Confunde banca sobre reprodutibilidade | Regenerar lock (smoke ou full) e commitar | Alta |
| Mudanças pós-auditoria não commitadas | working tree | Entrega Git incompleta se push do HEAD antigo | Commit/PR das correções | Alta |
| Stages DVC sem hash dos CSVs brutos | `dvc.yaml` deps | Troca de dados pode não invalidar stage | Documentar `dvc repro -f` (já no README) ou checksum auxiliar | Baixa |
| Conteúdo TMDB não no forward neural | `feature_engineering` / MLP | Expectativa vs realidade | Manter narrativa “colaborativo” no STAR — **não** forçar BERTopic | — |

**Não vale a pena refatorar agora:**

- Factory / Strategy / Registry / seeds / Settings.
- Cliente TMDB + cache.
- Critério de champion neural.
- Layout `src/`.

> Não há necessidade clara de alterar a arquitetura principal do pipeline.

---

## 8. Auditoria de Limpeza do Projeto

| Arquivo/Componente | Motivo | Evidência | Recomendação |
| ------------------ | ------ | --------- | ------------ |
| `scripts/hello_train.py` | Já removido | `Test-Path` → False | **Manter ausente** |
| Docs `CURSOR_*` | Já removidos | deletados no working tree | **Manter ausente** |
| Extra `topics` (BERTopic) | Opcional, sem import | `pyproject.toml` | **Manter** como optional |
| `llm/` + demos | Fora do DVC | Útil para STAR/demo | **Manter** se usados no vídeo |
| `src/features/`, `src/serving/` | Placeholders documentados | só `__init__.py` | **Manter** (narrativa honesta) |
| Caches `__pycache__`, `.pytest_cache` | Gerados | gitignore | **Manter ignorados** |
| `.env` | Secrets locais | gitignored | **Manter fora do Git** |
| `AUDITORIA_TECH_CHALLENGE.md` + este relatório | Documentação de processo | — | **Manter** para o time; opcional na banca |

---

## 9. Auditoria de Possíveis Resíduos ou Erros de Código Gerado por IA

### 9.1 Comentários / textos

- Sem `Generated by AI` / ChatGPT / Copilot em `src/`.
- Comentários scaffold (“In a real scenario”) do `feature_engineering` foram removidos na rodada anterior.
- Menções a Cursor/ChatGPT restantes estão em docs/prompts de avaliação humana (`llm/examples/`), não em código de produção.

### 9.2 Código suspeito

- `TorchMLPRecommender.fit()` ainda é superficial (treino real em `scripts/train.py`) — padrão conhecido, não bloqueante.
- Stages DVC deprecated warnings do MLflow Registry (`transition_model_version_stage`) — API antiga, ainda funcional.
- Fallback de split aleatório sem timestamp — documentado; ok se `timestamp` existir (MovieLens tem).

### 9.3 Dependências

Ver seção 10.

### 9.4 Segurança

- `.env` **não** versionado (`git check-ignore`).
- `.env.example` sem secrets.
- Não reproduzir chaves locais se existirem no `.env` da máquina.

---

## 10. Dependências e Segurança

| Item | Classificação | Motivo |
|------|---------------|--------|
| torch, sklearn, mlflow, dvc | Adequadas | Exigidas pelo PDF |
| packaging | Adequada | Agora direta no `pyproject.toml` |
| bertopic / sentence-transformers | Opcionais | Extra `topics`; sem uso no pipeline |
| Vulnerabilidades CVE | ⚪ Não comprovado | Sem scan de vulnerabilidade nesta auditoria |
| Secrets no Git | Não encontrados | `.env` ignorado |

**Desatualizada ≠ Vulnerável.**

---

## 11. Qualidade de Código

| Aspecto | Avaliação pós-ajustes |
|---------|----------------------|
| Legibilidade `src/` | Boa |
| Modularização train/evaluate | Melhorou muito (`loop`, `runner`, `champion`) |
| Ruff | Verde |
| Tipagem | Boa nas APIs públicas |
| Funções ≤ 20 linhas | Ainda parcial em scripts auxiliares |
| Manutenibilidade | Adequada ao desafio acadêmico |

Contexto acadêmico: o nível atual é **justificável e entregável**.

---

## 12. Testes e Confiabilidade

**Fatos:**

- `pytest tests/unit`: **60 testes, todos passando**.
- Novos: `test_champion.py`, `test_registry.py`, `test_raw_aliases.py`.
- CI configurada em `.github/workflows/ci.yml` (⚪ execução no GitHub Actions não comprovada nesta máquina).

| Área | Testes | Observação |
|------|--------|------------|
| Métricas @K / RMSE | Sim | Alinhado ao PDF |
| Factory / MostPopular / preprocessors / splits | Sim | |
| TMDB (mock) | Sim | |
| Champion neural | Sim | Novo e importante |
| Registry validate_metrics | Sim | Novo |
| Aliases CSV | Sim | Novo |
| `dvc repro` E2E | Não unitário | Depende de dados/ambiente |
| Cobertura % | ⚪ | `pytest-cov` existe; % não medida aqui |

**Conclusão:** testes **suficientes** para o núcleo do desafio; não substituem sozinhos um `dvc repro` de entrega.

---

## 13. Documentação

| Documento | Status | Observação |
|-----------|--------|------------|
| `README.md` | ✅ | Setup uv, DVC, Docker, MLflow SQLite |
| `docs/MODEL_CARD.md` | ✅ | Alinhado ao `metrics.json` atual |
| `docs/RELATORIO_MUDANCAS_POS_AUDITORIA.md` | ✅ | Changelog das correções |
| `docs/AUDITORIA_DESAFIO.md` | ✅ | Checklist atualizado |
| `TODO.md` | 🟡 | Ainda mistura desejo vs obrigação; vídeo aberto |
| Vídeo STAR | 🔴 | Ausente |

Documentação técnica **representa bem** o estado pós-ajustes. Risco residual: `dvc.lock` smoke vs narrativa “MovieLens 20M” / métricas reais.

---

## 14. Pontos Positivos

- Champion Production = **`torch_mlp`**, alinhado ao “modelo central” do PDF.
- Métricas reais coerentes (MLP melhor que baselines em RMSE).
- Factory + Strategy reais e usados — **não há necessidade clara de alterar**.
- Pipeline DVC completo com aliases Kaggle/GroupLens.
- Early stopping + seeds + MLflow logging.
- `pyproject.toml` bem estruturado (prod/dev/topics).
- Ruff + pytest + CI.
- Model Card honesta e atualizada.
- Limpeza de scaffold (`hello_train`, docs Cursor).
- Remote DVC dentro do repo (`./dvc-storage`).

---

## 15. Problemas P0

1. **Vídeo STAR obrigatório ausente** — zera até 10% da nota se não entregue.

---

## 16. Problemas P1

1. **`dvc.lock` desalinhado** do `dvc.yaml` atual e das métricas reais (ainda smoke).  
2. **Correções pós-auditoria ainda não commitadas** no HEAD remoto/local antigo — risco de entregar branch sem os fixes.  
3. **Regra ≤ 20 linhas** ainda violada em vários scripts auxiliares.

---

## 17. Problemas P2

1. Compose depende de volume `.:/app`.  
2. Warnings de API deprecated do MLflow Registry stages.  
3. Testes fracos para orquestração completa `dvc repro`.  
4. Deploy cloud / FastAPI ausentes (só importa se quiserem o bônus).

---

## 18. Problemas P3

1. Scripts de demo/PDF report ainda longos.  
2. Placeholders `features/`/`serving/` (já honestos).  
3. Polimento de docs duplicadas de auditoria.

---

## 19. Plano de Ação Recomendado

| # | Prioridade | Problema | Ação recomendada | Arquivos | Motivo | Impacto |
|---|------------|----------|------------------|----------|--------|--------|
| 1 | P0 | Sem vídeo STAR | Roteiro + gravação ≤ 5 min (usar métricas do `metrics.json`) | entrega externa | 10% da nota | Fecha entrega |
| 2 | P1 | Lock DVC smoke | Regenerar `dvc.lock` (smoke limpo **ou** full) e documentar qual | `dvc.lock`, README | Credibilidade DVC | Nota pipeline |
| 3 | P1 | Fixes não commitados | Commit/PR de todas as mudanças pós-auditoria | working tree | Banca vê o código certo | Entrega Git |
| 4 | P1 | Funções longas em demos | Quebrar ou deixar claro que não são o núcleo avaliado | `scripts/generate_llm_*`, demos | Clean code | Marginal |
| 5 | P2 | Warnings MLflow stages | Migrar para aliases quando couber | `registry.py` | Futuro-proof | Baixo agora |
| 6 | P3 | Polimento | Revisar docs duplicadas de auditoria | `docs/` | Clareza | Cosmético |

**Não executar estas ações nesta auditoria.**

---

## 20. Checklist Final para Entrega

- [ ] Vídeo STAR ≤ 5 min gravado e linkado
- [ ] Commit das correções pós-auditoria na branch de entrega
- [ ] `dvc.lock` coerente com a narrativa (smoke documentado ou full)
- [ ] `uv sync --extra dev` + `validate_env.py` OK
- [ ] `ruff` + `pytest` verdes (já OK no working tree)
- [ ] Print/demo do Registry com `torch_mlp` em Production no vídeo
- [ ] README com passos reais
- [ ] Nenhum `.env` commitado
- [ ] (Opcional) Deploy nuvem + URL

---

## 21. Conclusão

### Respostas diretas

1. **Atende ao Tech Challenge?** No código, **sim, em nível muito alto**; a entrega FIAP completa **ainda não**, sem o vídeo STAR.  
2. **Quanto está atendido?** Estimativa **~88%** (critérios ponderados; vídeo = 0).  
3. **O que falta?** Vídeo STAR; alinhar/commitar `dvc.lock` e as mudanças pendentes.  
4. **Estruturalmente errado?** Não — arquitetura adequada.  
5. **Vale refatorar?** Só lock/commit e, se necessário, scripts auxiliares longos.  
6. **Não vale refatorar?** Factory, Strategy, Registry, TMDB, seeds, champion neural.  
7. **Remover?** Scaffold já removido; demos/`llm/` manter se forem ao STAR.  
8. **Resíduos de IA?** Sem sinais graves em `src/` nesta rodada.  
9. **Deps problemáticas?** Não no núcleo; BERTopic só optional.  
10. **Segurança?** `.env` fora do Git; sem secrets no example.  
11. **Testes suficientes?** Sim para o núcleo (60 unitários + CI).  
12. **Docs corretas?** Sim, alinhadas ao run atual.  
13. **Antes da entrega?** Vídeo + commit + lock coerente.  
14. **Ordem?** STAR → commit dos fixes → `dvc.lock` → polimento.  
15. **Riscos de nota agora:** ausência do vídeo (10%); lock DVC confuso (parte de 15%); clean code parcial se a banca contar linhas à risca; branch sem commit dos fixes.

**Classificação final:** **Muito bem alinhado / praticamente pronto no código.** O projeto pós-ajustes cumpre o espírito e a letra técnica do PDF; o que separa da entrega completa é sobretudo o **vídeo STAR** e a higiene de versionamento (`commit` + `dvc.lock`).

---

*Auditoria somente leitura. Única alteração: criação deste arquivo `AUDITORIA_TECH_CHALLENGE_POS_AJUSTES.md`.*
