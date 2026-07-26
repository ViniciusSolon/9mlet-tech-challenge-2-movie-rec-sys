# Relatório de Mudanças — Correções Pós-Auditoria

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys`  
**Referência:** `AUDITORIA_TECH_CHALLENGE.md`  
**Data:** 2026-07-25  
**Objetivo:** fechar gaps técnicos do Tech Challenge (exceto vídeo STAR) e deixar o repositório pronto para entrega.

---

## 1. Resumo executivo

Foram aplicadas correções nos pontos P0–P3 da auditoria técnica, com foco em:

- reprodutibilidade (`pyproject.toml` / `uv`);
- pipeline DVC e nomes de CSV MovieLens;
- modelo neural como champion no MLflow Registry;
- clean code (extração de helpers);
- evidências reais em `metrics.json` + Model Card;
- limpeza de scaffold e CI básica.

**Resultado do evaluate regenerado (MovieLens completo no workspace):**

| Modelo | RMSE | MAE | P@10 | HR@10 |
|--------|------|-----|------|-------|
| **torch_mlp (champion / Production)** | **0,780** | **0,600** | **0,150** | **0,610** |
| most_popular | 0,887 | 0,691 | — | — |
| sklearn_random_forest | 0,906 | 0,720 | — | — |
| sklearn_knn | 0,948 | 0,760 | — | — |

**Pendência obrigatória restante do PDF:** gravação do vídeo STAR (≤ 5 min).

---

## 2. Motivação

A auditoria apontou aderência estimada de ~78%, com riscos principais:

1. evidências Git em modo smoke (`metrics.json` com @K = 0 e champion Random Forest);
2. `dvc.yaml` rígido a nomes `ratings.csv` (quebra com dump Kaggle `rating.csv`);
3. deps pesadas não usadas (BERTopic) e `packaging` só transitivo;
4. scripts longos e stubs de scaffold;
5. remote DVC apontando para pasta inexistente fora do repo.

---

## 3. Mudanças por área

### 3.1 Dependências (`pyproject.toml` + `uv.lock`)

| Antes | Depois |
|-------|--------|
| `bertopic` e `sentence-transformers` em deps de produção | Extra opcional `topics` (`uv sync --extra topics`) |
| `packaging` não declarado | Dependência direta `packaging>=24.0` |
| Sem `[tool.uv]` | `[tool.uv] package = true` |

**Uso correto validado:**

```bash
uv sync --extra dev          # pipeline + testes/lint
uv sync --extra topics       # só se for usar BERTopic
uv sync --extra s3           # remote DVC S3 (opcional)
```

### 3.2 Pipeline DVC e dados brutos

| Arquivo | Mudança |
|---------|---------|
| `src/data/raw_aliases.py` | **Novo** — cria aliases GroupLens ↔ Kaggle |
| `scripts/prepare_raw_aliases.py` | **Novo** — CLI chamado no início de preprocess/enrich |
| `dvc.yaml` | Stages passam a chamar o prepare; deps de CSV rígidas removidas |
| `.dvc/config` | Remote local `./dvc-storage` (antes `../data/dvc_remote`) |
| `dvc-storage/.gitkeep` | **Novo** — pasta versionável do remote |
| `scripts/create_dummy_data.py` | Gera nomes GroupLens **e** Kaggle |

### 3.3 Treino e avaliação (clean code + champion neural)

| Arquivo | Mudança |
|---------|---------|
| `src/training/loop.py` | **Novo** — loaders, epoch, early stopping, optimizer |
| `scripts/train.py` | Orquestração mais enxuta usando o loop |
| `src/evaluation/runner.py` | **Novo** — baselines, ranking, logging MLflow |
| `src/evaluation/champion.py` | **Novo** — prefere `torch_*` para Production |
| `scripts/evaluate.py` | Usa runner + champion; JSON sem NaN inválido |
| `src/evaluation/metrics.py` | RMSE/MAE delegam ao scikit-learn (fonte única) |

**Critério de champion:** entre candidatos `torch_*`, escolhe o menor RMSE; se não houver neural, cai para o melhor baseline.

**Fix MLflow/sklearn:** `serialization_format="pickle"` (skops rejeitava tipos do KNN).

**Ranking:** scoring em batches + amostra de até 200 usuários (viável no MovieLens 20M).

### 3.4 Feature engineering

| Arquivo | Mudança |
|---------|---------|
| `scripts/feature_engineering.py` | Removidos comentários scaffold; anexa `year` / `vote_average` / `popularity` quando existirem |
| `src/features/__init__.py` | Docstring honesta (BERTopic opcional, fora do PDF) |
| `src/serving/__init__.py` | Docstring honesta (FastAPI = bônus opcional) |

### 3.5 Ambiente, Docker e CI

| Arquivo | Mudança |
|---------|---------|
| `scripts/validate_env.py` | BERTopic opcional; checa `packaging`; cria `dvc-storage`; saída ASCII (Windows) |
| `Dockerfile` | Copia `dvc.yaml`, `params.yaml`, `.dvc`; cria dirs de dados |
| `.env.example` | Documenta SQLite local vs `localhost:5000` no compose |
| `.gitignore` | Ignora conteúdo de `dvc-storage/` (mantém `.gitkeep`) |
| `.github/workflows/ci.yml` | **Novo** — `uv sync --extra dev` + ruff + pytest |

### 3.6 Testes novos

| Arquivo | O que cobre |
|---------|-------------|
| `tests/unit/test_champion.py` | Preferência por modelo neural |
| `tests/unit/test_registry.py` | Validação de métricas do Registry |
| `tests/unit/test_raw_aliases.py` | Aliases Kaggle → GroupLens |

**Validação local após as mudanças:** ruff OK · pytest unitário 60/60 · `validate_env` 27/27.

### 3.7 Limpeza

| Removido | Motivo |
|----------|--------|
| `scripts/hello_train.py` | Scaffold Docker antigo |
| `docs/CURSOR_STRUCTURE_REPORT.md` | Histórico SetAI/Cursor obsoleto |
| `docs/CURSOR_REFACTOR_REPORT.md` | Idem |

### 3.8 Documentação e evidências

| Arquivo | Mudança |
|---------|---------|
| `metrics.json` | Regenerado com run real; champion `torch_mlp` / Production |
| `docs/MODEL_CARD.md` | Tabelas e narrativa alinhadas ao run atual |
| `docs/AUDITORIA_DESAFIO.md` | Checklist atualizado |
| `docs/DOCUMENTACAO_ETAPA2.md` | Extra `topics` documentado |
| `README.md` | `uv sync --extra dev`, aliases DVC, MLflow SQLite |
| `TODO.md` | Status pós-correção; instalação limpa marcada |
| `AUDITORIA_TECH_CHALLENGE.md` | Relatório de auditoria (entrada desta rodada) |

---

## 4. Problemas encontrados durante a regeneração do evaluate

| Tentativa | Erro | Correção |
|-----------|------|----------|
| 1 | MLflow em `http://localhost:5000` sem servidor | Usar / documentar `sqlite:///mlflow.db` |
| 2 | `skops` rejeitou tipos do KNN | `serialization_format="pickle"` |
| 3 | Sucesso | `torch_mlp` → Production; `metrics.json` atualizado |

---

## 5. Arquivos novos (lista)

```text
.github/workflows/ci.yml
dvc-storage/.gitkeep
scripts/prepare_raw_aliases.py
src/data/raw_aliases.py
src/evaluation/champion.py
src/evaluation/runner.py
src/training/loop.py
tests/unit/test_champion.py
tests/unit/test_raw_aliases.py
tests/unit/test_registry.py
docs/RELATORIO_MUDANCAS_POS_AUDITORIA.md   # este documento
```

---

## 6. O que NÃO foi alterado (de propósito)

- Arquitetura geral `src/` (data / models / evaluation / training).
- Factory de modelos e Strategy de preprocessadores.
- Cliente TMDB e cache de metadados.
- Deploy em nuvem / FastAPI (bônus opcional do PDF).
- Integração completa de BERTopic no `feature_eng` (não exigida pelo enunciado).
- Vídeo STAR (entrega humana, fora do código).

---

## 7. Como reproduzir o estado atual

```bash
uv sync --extra dev
python -c "from pathlib import Path; Path('.env').write_text(Path('.env.example').read_text(encoding='utf-8'), encoding='utf-8')"
# Ajuste TMDB_API_KEY apenas se for refetch de metadados
uv run python scripts/validate_env.py
uv run python scripts/prepare_raw_aliases.py
uv run dvc repro
# ou só a avaliação, com modelo já treinado:
uv run python scripts/evaluate.py
```

---

## 8. Checklist pós-mudanças vs Tech Challenge

| Item do PDF | Status após esta rodada |
|-------------|-------------------------|
| Clean code / Factory / Strategy / ruff | ✅ |
| `pyproject` + lock + `.env` + validate_env | ✅ |
| Docker multi-stage + compose | ✅ |
| DVC ≥ 3 stages | ✅ (5 stages) |
| PyTorch + early stopping | ✅ |
| Baselines + ≥ 4 métricas | ✅ |
| MLflow + Registry Staging→Production | ✅ (`torch_mlp`) |
| Model Card + README | ✅ |
| Vídeo STAR | ❌ pendente |
| Deploy nuvem | ➖ opcional |

---

## 9. Conclusão

As mudanças fecham os gaps técnicos críticos da auditoria e alinham as evidências versionadas (`metrics.json`, Model Card, champion neural) ao run real do MovieLens. O repositório fica em estado adequado para a banca no código; a entrega completa do Tech Challenge depende agora apenas do **vídeo STAR**.
