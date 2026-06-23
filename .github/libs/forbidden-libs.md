# Forbidden Libraries

## Deep Learning

| Library | Reason | Alternative |
|---------|--------|-------------|
| **tensorflow** / **keras** | Project standardizes on **PyTorch** | `torch` |

## Node.js Ecosystem (entire stack)

| Library | Reason |
|---------|--------|
| commander, inquirer, yargs | CLI not project focus |
| eslint, prettier, typescript | Use **ruff** + Python |
| vitest, jest, mocha | Use **pytest** |
| tsup, webpack, rollup | Use Docker + Poetry/uv |
| prisma, pg (node-postgres) | Use SQLAlchemy + psycopg2 |

## ML Anti-Patterns

| Pattern | Reason |
|---------|--------|
| Unpinned `pip install` in Dockerfile without lock | Breaks reproducibility |
| Calling TMDB on every `dvc repro` | Rate limits, non-reproducible |
| `implicit` duplicate of custom torch model without justification | Prefer single primary DL stack |

## Security / Maintenance

- Packages with known CVEs and no fix  
- Unmaintained libs (< 1000 weekly downloads) without approval  
- `pickle` from untrusted sources  

## HTTP

- **Axios** (Node) — forbidden  
- **requests/httpx** — allowed only for TMDB with cached DVC artifact  

## Approval Process

1. Document necessity  
2. Tech Lead approval  
3. Add exception to this file with reason  

## Related

- `.cursor/libs/allowed-libs.md`
