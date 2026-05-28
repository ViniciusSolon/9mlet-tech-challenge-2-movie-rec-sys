"""Errors for external metadata providers."""


class TmdbRequestError(Exception):
    """TMDB request failed after bounded retries."""
