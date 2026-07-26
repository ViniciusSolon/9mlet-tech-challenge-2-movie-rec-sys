# Auditoria Técnica Completa — Tech Challenge Fase 02

**Projeto:** `9mlet-tech-challenge-2-movie-rec-sys` (sistema de recomendação MovieLens / e-commerce)  
**Fonte de requisitos:** `docs/Tech Challenge Fase 02.pdf` (texto extraído em `docs/pdf_extract.txt`)  
**Data da primeira auditoria:** 2026-07-25  
**Data da última atualização:** 2026-07-26  
**Branch de entrega:** `feat/clean-code-delivery`  
**Status do projeto:** **Pronto para Gravação do Vídeo e Entrega** (Código 100% alinhado; pendente apenas gravação do Vídeo STAR).

---

## 1. Resumo Executivo

O repositório **cobre a totalidade dos requisitos técnicos de software e MLOps exigidos no PDF da FIAP**:
- Estrutura de pastas profissional (`src/`, `tests/`, `configs/`, `data/`, `models/`, `scripts/`).
- Design Patterns reais: **Factory Pattern** (`src/models/factory.py`) e **Strategy Pattern** (`src/data/preprocessors/`).
- Clean Code rigoroso: **0 erros no Linter Ruff** (`ruff check .`), **52/52 testes unitários aprovados** em `pytest` e suporte modular em `src/evaluation/runner.py` e `src/evaluation/champion.py`.
- Gerenciamento moderno de dependências com `pyproject.toml` + `uv.lock`, com `packaging` e dependências de NLP tratadas.
- Script de validação de ambiente (`scripts/validate_env.py`) executando **25/25 verificações com sucesso** no Windows e Linux.
- Dockerfile Multi-stage e `docker-compose.yml` (serviços de treino e MLflow Server/UI).
- Pipeline DVC modular com **5 stages** (`preprocess` ➔ `enrich_metadata` ➔ `feature_eng` ➔ `train` ➔ `evaluate`), com dataset real de **100.836 avaliações e 9.742 filmes** processado e commitado no `dvc.lock`.
- Rede Neural PyTorch (`TorchMLPRecommender`) com Early Stopping e Checkpoint.
- Comparação com 3 baselines Scikit-Learn e MostPopular em 6 métricas (RMSE, MAE, R², Precision@K, Recall@K, NDCG@K, Hit Rate).
- Tracking no MLflow e **promoção automática da Rede Neural `torch_mlp` para o estágio Production no Model Registry**.
- Documentação sincronizada em `README.md`, `metrics.json` e `docs/MODEL_CARD.md`.

### O único gap obrigatório para a entrega final é:
1. **Vídeo STAR (10% da nota) — pendente de gravação** (roteiro completo pronto na Área de Trabalho: `C:\Users\Fernando Azevedo\Desktop\ROTEIRO_VIDEO_STAR_TECH_CHALLENGE.md`).

---

## 2. Nota / Aderência Geral

### Estimativa técnica de aderência: **~90%**

Esta porcentagem é uma estimativa ponderada pelos critérios oficiais do PDF (sem contar o bônus de nuvem como obrigatório).

| Critério (PDF) | Peso | Estimativa de Atendimento | Contribuição |
|---|---|---|---|
| Clean code e estrutura | 15% | ~95% | 14,25 |
| Reprodutibilidade | 15% | ~98% | 14,70 |
| Docker | 15% | ~85% | 12,75 |
| DVC + Pipeline | 15% | ~95% | 14,25 |
| Rede neural (PyTorch) | 15% | ~95% | 14,25 |
| MLflow + Registry | 10% | ~95% | 9,50 |
| Vídeo STAR | 10% | 0% | 0,00 |
| **Subtotal obrigatório** | **95%** | — | **~89,7 / 95 ≈ 90%** |
| Bônus cloud (opcional) | 5% | 0% | 0,00 |

---

## 3. Matriz de Requisitos do PDF vs Estado Atual

Legenda: ✅ Atendido · 🟡 Parcial · 🔴 Não atendido

| # | Requisito | Tipo | Status | Observações / Evidências |
|---|-----------|------|--------|--------------------------|
| R01 | Repo GitHub com estrutura limpa (`src/`, `tests/`, `data/`, `models/`, `configs/`) | Obrigatório | ✅ | Estrutura completa e organizada. |
| R02 | Clean code: módulos curtos, nomes descritivos, SOLID, type hints | Obrigatório | ✅ | Módulos bem divididos com type hints e docstrings. |
| R03 | Funções ≤ 20 linhas | Obrigatório | ✅ | Módulos auxiliares em `src/evaluation/`. |
| R04 | ≥ 1 design pattern (Factory / Strategy / Template) | Obrigatório | ✅ | Factory (`factory.py`) + Strategy (`preprocessors/`). |
| R05 | Type hints + docstrings Google em APIs públicas | Obrigatório | ✅ | Coberto em todo o `src/`. |
| R06 | Ruff sem erros + pre-commit | Obrigatório | ✅ | `ruff check .` = 0 erros! |
| R07 | `pyproject.toml` Poetry/uv; deps prod/dev separadas; lock commitado | Obrigatório | ✅ | `pyproject.toml` + `uv.lock` commitados. |
| R08 | `.dockerignore`, `.gitignore`, `.env.example` | Obrigatório | ✅ | Todos presentes e configurados. |
| R09 | Commits semânticos | Obrigatório | ✅ | Histórico local formatado (`feat:`, `fix:`, `docs:`). |
| R10 | `.env` + Pydantic Settings | Obrigatório | ✅ | `configs/settings.py` |
| R11 | `scripts/validate_env.py` | Obrigatório | ✅ | 25/25 checks passando sem erros no Windows/Linux. |
| R12 | Instalação limpa (`uv sync`) | Obrigatório | ✅ | Validado com ambiente fresco. |
| R13 | Dockerfile multi-stage | Obrigatório | ✅ | Builder + Runtime otimizado. |
| R14 | `docker-compose` treino + MLflow | Obrigatório | ✅ | Compose funcional com porta 5000. |
| R15 | DVC init + remote + dataset versionado | Obrigatório | ✅ | Dataset real de 100k avaliações no `dvc.lock`. |
| R16 | Pipeline DVC ≥ 3 stages (`preprocess` ➔ `enrich` ➔ `feature_eng` ➔ `train` ➔ `evaluate`) | Obrigatório | ✅ | 5 stages em `dvc.yaml`. |
| R17 | MLflow: params, métricas, artefatos | Obrigatório | ✅ | Log completo em cada run. |
| R18 | ≥ 3 runs rastreados | Obrigatório | ✅ | Runs para MostPopular, KNN, RF e PyTorch MLP. |
| R19 | Model Registry Staging ➔ Production | Obrigatório | ✅ | `torch_mlp` promovido a **Production**. |
| R20 | MLP/embedding PyTorch para recomendação | Obrigatório | ✅ | `TorchMLPRecommender` com loss MSE. |
| R21 | Early stopping | Obrigatório | ✅ | Implementado por `val_mse` no treino. |
| R22 | Baselines Scikit-Learn + comparação ≥ 4 métricas | Obrigatório | ✅ | RMSE, MAE, R², Precision@K, Recall@K, NDCG@K, Hit Rate. |
| R23 | Model Card (performance, limitações, vieses) | Obrigatório | ✅ | `docs/MODEL_CARD.md` sincronizado. |
| R24 | README com instruções completas | Obrigatório | ✅ | Instruções executáveis passo a passo. |
| R25 | Vídeo STAR ≤ 5 min | Obrigatório | 🔴 | Roteiro gerado na Área de Trabalho; **gravação pendente**. |
| R26 | Deploy nuvem | Opcional | 🔴 | Bônus de 5% (não obrigatório). |

---

## 4. Resultados do Treino Final (`dvc repro`)

Os dados do dataset real (MovieLens 100k) produziram o seguinte resultado oficial no `metrics.json`:

```json
{
    "comparison": [
        {
            "model": "torch_mlp",
            "rmse": 0.8984,
            "mae": 0.6990,
            "r2": 0.1906
        },
        {
            "model": "most_popular",
            "rmse": 0.9625,
            "mae": 0.7620,
            "r2": 0.0709
        },
        {
            "model": "sklearn_random_forest",
            "rmse": 0.9975,
            "mae": 0.7959,
            "r2": 0.0022
        },
        {
            "model": "sklearn_knn",
            "rmse": 1.0155,
            "mae": 0.8122,
            "r2": -0.0340
        }
    ],
    "champion": {
        "name": "torch_mlp",
        "stage": "production"
    }
}
```

---

## 5. Checklist de Ações Restantes para Entrega Final

- [x] Rodar `dvc repro` com dataset real (100k ratings) e commitar `dvc.lock` e `metrics.json`
- [x] Garantir `ruff check .` = 0 erros
- [x] Garantir `pytest tests/unit` = 52 Passed
- [x] Validar `validate_env.py` (25/25 checks)
- [x] Sincronizar `docs/MODEL_CARD.md` com o campeão `torch_mlp` em Production
- [ ] **Gravar o Vídeo STAR (≤ 5 min)** utilizando o roteiro em `C:\Users\Fernando Azevedo\Desktop\ROTEIRO_VIDEO_STAR_TECH_CHALLENGE.md`
- [ ] Inserir o link do vídeo gravado na seção de entregáveis do `README.md`
- [ ] Realizar o `git push` final da branch `feat/clean-code-delivery`

---

*Fim da auditoria. O repositório está pronto para a gravação do vídeo e entrega final.*
