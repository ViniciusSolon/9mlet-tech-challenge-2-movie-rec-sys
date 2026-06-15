# Security Rules

## Secrets

- Store secrets in `.env` only; provide `.env.example` without values  
- Never commit: `.env`, API keys, DVC remote credentials, database passwords  
- Use GitHub Secrets for CI  

## Data & Privacy

- MovieLens IDs are pseudonymous — do not enrich logs with external PII  
- Do not log full rating histories in production logs  

## ML-Specific Risks

- **Pickle:** load only trusted model artifacts from MLflow/DVC outputs  
- **TMDB API:** rate-limit client; cache responses under DVC; never call API on every `dvc repro`  
- **SQL:** SQLAlchemy/psycopg2 with parameterized queries only — no string concatenation  

## Input Validation

- Pydantic models for FastAPI inference requests  
- Validate `user_id` / `top_k` bounds on serving endpoints  
- Reject negative `top_k` or unknown users with clear 4xx responses  

## Dependencies

- Pin versions in lock file  
- Run `pip audit` or GitHub Dependabot  
- Only libraries from `allowed-libs.md` unless Tech Lead approves exception  

## Docker

- Non-root user in runtime image  
- No secrets baked into image layers  
- Multi-stage build to exclude dev dependencies from runtime  

## Prohibited

- Node.js stack references (Prisma, node-postgres) — project is Python  
- Scraping IMDb (ToS); use TMDB API with key  
- Disabling security checks in CI  

## Pre-Deploy Security Checklist

- [ ] No secrets in repo or image  
- [ ] `.env` in `.gitignore`  
- [ ] Dependencies scanned  
- [ ] Model artifacts from trusted pipeline only  

## Related

- `.cursor/context/deployment.md`
- `.cursor/rules/mlops.mdc`
