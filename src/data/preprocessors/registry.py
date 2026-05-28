"""Registry for preprocessor strategies."""

from __future__ import annotations

from data.preprocessors.base import PreprocessorStrategy
from data.preprocessors.explicit import ExplicitFeedbackPreprocessor
from data.preprocessors.implicit import ImplicitFeedbackPreprocessor

_STRATEGIES: dict[str, type[PreprocessorStrategy]] = {
    "explicit": ExplicitFeedbackPreprocessor,
    "implicit": ImplicitFeedbackPreprocessor,
}


def get_preprocessor(name: str, **kwargs: object) -> PreprocessorStrategy:
    """Instantiate a preprocessor by name.

    Args:
        name: ``explicit`` or ``implicit``.
        **kwargs: Passed to the strategy constructor.

    Returns:
        Configured preprocessor instance.

    Raises:
        ValueError: If name is unknown.
    """
    cls = _STRATEGIES.get(name)
    if cls is None:
        msg = f"unknown preprocessor: {name}"
        raise ValueError(msg)
    return cls(**kwargs)
