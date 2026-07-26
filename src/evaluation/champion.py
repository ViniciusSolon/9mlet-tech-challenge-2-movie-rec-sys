"""Champion selection for MLflow Model Registry promotion."""

from __future__ import annotations


def select_champion(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Pick the Production champion among evaluation candidates.

    The Tech Challenge central model is the neural recommender. Among
    ``torch_*`` candidates, choose the lowest RMSE. If none exist, fall
    back to the overall lowest RMSE (baselines only).

    Args:
        candidates: Payloads with ``name`` and ``metrics`` (must include rmse).

    Returns:
        The selected candidate dictionary.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        msg = "no candidates to select a champion from"
        raise ValueError(msg)

    neural = [item for item in candidates if str(item["name"]).startswith("torch")]
    pool = neural or candidates
    return min(pool, key=lambda item: float(item["metrics"]["rmse"]))  # type: ignore[index]
