# Business Rules — Recommendation System

## Domain

MovieLens 20M recommendation for FIAP Tech Challenge 02. Optional e-commerce analogy: user = customer, movie = product (SKU).

## Recommendation Output

- Return **top-K** items per user (default K configurable, e.g. 10)  
- **Exclude** items the user has already rated/interacted with in training history (unless evaluating with proper hold-out)  
- Scores must be sortable; ties broken deterministically (e.g. by `movieId`)  

## Data Splitting

- Prefer **temporal split** on `timestamp`: train on past, validate/test on future  
- Document split dates in MLflow params  
- **No leakage:** features for an interaction must use only information available before that timestamp (especially TMDB popularity / ratings)  

## Cold Start

| Case | Behavior |
|------|----------|
| **New user** (no history) | Fallback: popular movies, genre-based, or content-only from last profile hint |
| **New item** (few ratings) | Boost content features (tags, BERTopic, TMDB metadata) |
| Document fallbacks in Model Card |

## Evaluation Metrics (required)

Report at minimum on hold-out set:

- **Recall@K**
- **Precision@K**
- **NDCG@K**
- **RMSE** (rating prediction baseline comparison)

Compare PyTorch model vs ≥ 1 sklearn baseline in MLflow.

## Data Quality

- Drop or impute invalid `movieId` / `userId`  
- Handle missing `links.csv` entries in `enrich_metadata` without failing entire pipeline  
- Minimum interactions per user/item configurable (filter long-tail noise for training)  

## MLflow / Registry

- Every training run logs params, metrics, and model artifact  
- Promote to Production only after evaluate stage passes thresholds defined in `configs/`  

## Non-Goals (do not implement as requirements)

- Real-time streaming recommendations at scale  
- User login / account management  
- Full production frontend  

## Related

- `.cursor/context/project-goals.md`
- `.cursor/context/ml-pipeline.md`
- `.cursor/commands/recommendation-metrics.md`
