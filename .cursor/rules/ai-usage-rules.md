# AI Usage Rules

## Principles

1. AI does not make final architecture, security, or grading decisions alone  
2. Explicit context (`.cursor/context/`) beats long ad-hoc prompts  
3. Generated code must pass **ruff** and **pytest**  
4. Humans review all ML metrics claims against MLflow runs  

## Where AI May Help

| Area | Allowed |
|------|---------|
| Pipeline design (DVC stages) | Yes — human validates reproducibility |
| PyTorch model code | Yes — with unit/smoke tests |
| MLflow logging boilerplate | Yes |
| pytest fixtures | Yes |
| Model Card / README | Yes — verify metrics |
| Refactoring `src/` | Yes — behavior unchanged |

## Where AI Must Not Act Alone

- Production MLflow Registry promotion  
- Changing `dvc.yaml` stage dependencies without review  
- Deleting DVC-tracked artifacts  
- Security credentials or `.env` values  
- Declaring challenge requirements met without `dvc repro` evidence  

## Mandatory Project Setup Before AI-Assisted Coding

- [ ] `pyproject.toml` + lock file  
- [ ] `ruff` configured  
- [ ] `pytest` configured  
- [ ] `.cursor/context/` and `.mdc` rules present  
- [ ] `dvc.yaml` skeleton (when pipeline work starts)  

**Do not use Node.js/TypeScript tooling as prerequisites — this is a Python ML project.**

## Models for LLM Tasks

See `.cursor/libs/ai-models.md` for Cursor model selection by phase.

## Tests & Quality

- New `src/` modules → corresponding tests  
- Metric functions → known-value unit tests  
- AI never replaces CI (`ruff`, `pytest`, optional `dvc repro`)  

## Related

- `.cursor/libs/ai-models.md`
- `.cursor/rules/python-ml.mdc`
- `AGENTS.md`
