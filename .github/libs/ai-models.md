# AI Models (Cursor / LLM)

## Project Context

**FIAP Tech Challenge 02** — Python, PyTorch, DVC, MLflow, MovieLens 20M.

Use LLMs for code and docs; use PyTorch for the recommender model.

## By Phase

| Phase | Primary | Use when |
|-------|---------|----------|
| Architecture / pipeline | Claude Opus / GPT-5 | DVC design, MLflow strategy |
| Implementation | Cursor Composer + Codex | `src/`, tests, configs |
| Refactoring | Claude Sonnet | Safe refactors with pytest green |
| Debug | Gemini Pro | Tracebacks, DVC/MLflow errors |
| Boilerplate | Gemini Flash | Repetitive pytest fixtures |

## Rules

- Review all generated PyTorch and `dvc.yaml` changes  
- Validate metric claims against real MLflow runs  
- Do not accept Node.js dependencies from LLM suggestions  

## Related

- `.cursor/rules/ai-usage-rules.md`
- `AGENTS.md`
