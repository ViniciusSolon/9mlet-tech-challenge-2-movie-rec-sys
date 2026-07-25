"""Integration tests: real model.pth vs JSON golden cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.factory import create_model

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "inference_cases.json"
MODEL_PATH = ROOT / "models" / "model.pth"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _load_model(fixture: dict) -> torch.nn.Module:
    state = torch.load(MODEL_PATH, map_location="cpu")
    n_users, emb_dim = state["user_embedding.weight"].shape
    n_items = state["item_embedding.weight"].shape[0]
    cfg = fixture["model"]
    model = create_model(
        cfg["type"],
        n_users=n_users,
        n_items=n_items,
        embedding_dim=cfg.get("embedding_dim", emb_dim),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(state)
    model.fit([], None)
    model.eval()
    return model


def _predict(model: torch.nn.Module, user_idx: int, movie_idx: int) -> float:
    x = torch.tensor([[user_idx, movie_idx]], dtype=torch.long)
    with torch.no_grad():
        return float(model(x).squeeze().cpu())


@pytest.fixture(scope="module")
def real_model() -> torch.nn.Module:
    if not MODEL_PATH.is_file():
        pytest.skip("models/model.pth não encontrado — rode o treino antes")
    if not FIXTURE.is_file():
        pytest.skip(f"fixture ausente: {FIXTURE}")
    return _load_model(_load_fixture())


@pytest.fixture(scope="module")
def cases() -> dict:
    return _load_fixture()


def test_score_cases_match_golden(real_model: torch.nn.Module, cases: dict) -> None:
    """Predições ficam dentro da tolerância do JSON de expectativa."""
    failures: list[str] = []
    for case in cases["score_cases"]:
        user_idx = case["input"]["user_idx"]
        movie_idx = case["input"]["movie_idx"]
        expected = case["expected"]
        score = _predict(real_model, user_idx, movie_idx)

        if not (expected["min"] <= score <= expected["max"]):
            failures.append(
                f"{case['id']}: score={score:.4f} fora de [{expected['min']}, {expected['max']}]"
            )
            continue

        delta = abs(score - expected["score"])
        if delta > expected["abs_tol"]:
            failures.append(
                f"{case['id']}: score={score:.4f} esperado≈{expected['score']} "
                f"(Δ={delta:.4f} > tol={expected['abs_tol']})"
            )

    assert not failures, "Falhas em score_cases:\n" + "\n".join(failures)


def test_order_cases_preserve_preference(
    real_model: torch.nn.Module,
    cases: dict,
) -> None:
    """O modelo mantém a ordem de preferência capturada no golden JSON."""
    failures: list[str] = []
    for case in cases["order_cases"]:
        user_idx = case["input"]["user_idx"]
        higher_idx = case["input"]["higher_movie_idx"]
        lower_idx = case["input"]["lower_movie_idx"]
        margin = case["expected"]["min_margin"]

        high = _predict(real_model, user_idx, higher_idx)
        low = _predict(real_model, user_idx, lower_idx)
        if high < low + margin:
            failures.append(
                f"{case['id']}: score({higher_idx})={high:.4f} "
                f"não supera score({lower_idx})={low:.4f} com margem {margin}"
            )

    assert not failures, "Falhas em order_cases:\n" + "\n".join(failures)
