# Project Goals

## Problem Statement

Build a **personalized recommendation system** using the **MovieLens 20M** dataset, learning user–movie preference patterns to suggest relevant titles and increase engagement.

The project also demonstrates **MLOps engineering**: reproducible pipelines (DVC), experiment tracking (MLflow), containerization (Docker), and PyTorch-based neural recommenders with sklearn baselines.

## Target Users

- **End users:** viewers receiving personalized movie recommendations  
- **Business:** platform seeking higher engagement and retention  
- **Technical:** ML/MLOps teams requiring versioned data, tracked experiments, and reproducible deploys  

## Business Objectives

- Improve discovery and reduce search time  
- Increase consumption of relevant content  
- Deliver a reproducible ML pipeline aligned with FIAP Tech Challenge criteria  

## Technical Constraints (mandatory)

- PyTorch neural recommender (MLP or embedding-based)  
- Scikit-Learn baselines with ≥ 4 metrics  
- MLflow: parameters, metrics, artifacts, Model Registry  
- DVC: versioned data + pipeline ≥ 3 stages  
- Docker multi-stage + docker-compose (app + MLflow)  
- Poetry or uv with committed lock file  
- Clean Code, SOLID, type hints, Google docstrings  
- Factory (models) and Strategy (preprocessors)  
- Ruff + pytest + pre-commit  

## Business Constraints

- GitHub repository + STAR video (≤ 5 min)  
- Dataset with ≥ 10k user–item interactions (MovieLens 20M satisfies)  
- Metric comparison across experiments documented in MLflow  

## Non-Goals

- Video streaming platform  
- User authentication / full production UI  
- Large-scale real-time serving or distributed training  
- Mandatory cloud deploy (optional bonus only)  

## Related Documentation

- `.cursor/context/architecture.md`
- `.cursor/context/ml-pipeline.md`
- `TODO.md`
