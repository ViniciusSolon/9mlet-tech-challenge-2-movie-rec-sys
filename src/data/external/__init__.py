"""External metadata clients (TMDB)."""

from data.external.metadata_fetch import run_fetch
from data.external.movielens_io import build_link_table, load_links
from data.external.tmdb_client import TmdbClient

__all__ = ["TmdbClient", "build_link_table", "load_links", "run_fetch"]
