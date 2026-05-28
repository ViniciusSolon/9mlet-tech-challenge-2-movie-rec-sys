# Testing Rules

## Strategy

- **Unit tests:** domain logic, Factory, Strategy, metrics, encoders  
- **Integration tests:** DVC stage scripts with tiny fixtures  
- **Pipeline smoke:** subset data end-to-end (`dvc repro` on sample)  
- **Pyramid:** many unit, fewer integration, minimal E2E  

TDD is **recommended** for pure Python modules; for ML pipelines prioritize **contract tests** on data schemas and metric functions before full training loops.

## Tools

- **pytest** + **pytest-cov**  
- Fixtures in `tests/conftest.py`  
- Synthetic MovieLens-like samples in `tests/fixtures/` (never commit full 20M)  

## Coverage Targets

| Area | Minimum |
|------|---------|
| Overall | 70% |
| `src/domain/`, `src/evaluation/`, Factory, Strategy | 90%+ |
| Training loops | Smoke + metric assertions on tiny data |

## Pipeline & Data Tests

- Validate schema: required columns (`userId`, `movieId`, `rating`, `timestamp`)  
- Assert temporal split: train timestamps < test timestamps  
- Assert no user leakage across splits when configured  
- After `feature_eng`, matrix dimensions match unique user/item counts  

## Recommendation Metrics Tests

Unit-test metric implementations with known inputs:

- **Recall@K**
- **Precision@K**
- **NDCG@K**
- **RMSE**

## Reproducibility Tests

- Same seed → same metric on fixed tiny dataset (within float tolerance)  

## File Naming

- `tests/unit/test_*.py`  
- `tests/integration/test_*.py`  

## CI

- PR blocked if `pytest` or `ruff` fails  
- Coverage must not decrease on new `src/` code without approval  

## Prohibited

- Production MovieLens data in tests  
- Flaky tests depending on network (mock TMDB)  
- Tests with no assertions  

## Related

- `.cursor/rules/code-style.md`
- `.cursor/rules/testing.mdc`
- `.cursor/commands/test-strategy.md`
