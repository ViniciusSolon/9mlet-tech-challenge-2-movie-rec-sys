"""Centralized random seeds for reproducibility."""

from __future__ import annotations

import os
import random


def set_global_seeds(seed: int = 42) -> None:
    """Set Python and hash seeds (numpy/torch in Bloco 2)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
