#!/usr/bin/env python
"""Roda os casos de tests/fixtures/inference_cases.json contra models/model.pth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "inference_cases.json"
MODEL_PATH = ROOT / "models" / "model.pth"


def main() -> int:
    if not MODEL_PATH.is_file():
        print(f"Erro: {MODEL_PATH} não encontrado. Treine antes.")
        return 1

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
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

    ok = True
    print("=== score_cases ===")
    for case in fixture["score_cases"]:
        u = case["input"]["user_idx"]
        m = case["input"]["movie_idx"]
        with torch.no_grad():
            score = float(model(torch.tensor([[u, m]], dtype=torch.long)).squeeze())
        exp = case["expected"]
        delta = abs(score - exp["score"])
        passed = exp["min"] <= score <= exp["max"] and delta <= exp["abs_tol"]
        ok = ok and passed
        status = "OK" if passed else "FAIL"
        print(
            f"[{status}] {case['id']}: pred={score:.4f} "
            f"esperado≈{exp['score']} (±{exp['abs_tol']})"
        )

    print("=== order_cases ===")
    for case in fixture["order_cases"]:
        u = case["input"]["user_idx"]
        hi_i = case["input"]["higher_movie_idx"]
        lo_i = case["input"]["lower_movie_idx"]
        margin = case["expected"]["min_margin"]
        with torch.no_grad():
            hi = float(model(torch.tensor([[u, hi_i]], dtype=torch.long)).squeeze())
            lo = float(model(torch.tensor([[u, lo_i]], dtype=torch.long)).squeeze())
        passed = hi >= lo + margin
        ok = ok and passed
        status = "OK" if passed else "FAIL"
        print(f"[{status}] {case['id']}: {hi:.4f} vs {lo:.4f} (margem mín. {margin})")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
