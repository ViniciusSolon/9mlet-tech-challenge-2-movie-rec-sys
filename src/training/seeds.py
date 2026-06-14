"""Centralized random seeds for reproducibility.

Call ``set_global_seeds()`` once at the beginning of every training script
or notebook to guarantee deterministic results across all libraries.
"""

from __future__ import annotations

import os
import random


def set_global_seeds(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
