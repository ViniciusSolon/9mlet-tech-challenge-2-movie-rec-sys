"""Coverage statistics for ``movie_metadata.parquet``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _filled(series: pd.Series) -> pd.Series:
    """True where string field is non-empty."""
    if series.dtype == object:
        return series.notna() & (series.astype(str).str.strip() != "")
    return series.notna()


def build_coverage_frame(metadata: pd.DataFrame) -> dict[str, float | int]:
    """Compute coverage metrics from metadata parquet."""
    total = len(metadata)
    ok = metadata[metadata["fetch_status"] == "ok"]
    ok_n = len(ok)

    def pct(mask: pd.Series) -> float:
        return round(100.0 * mask.sum() / ok_n, 2) if ok_n else 0.0

    return {
        "total_rows": total,
        "ok": ok_n,
        "api_error": int((metadata["fetch_status"] == "api_error").sum()),
        "not_found": int((metadata["fetch_status"] == "not_found").sum()),
        "missing_tmdb_id": int((metadata["fetch_status"] == "missing_tmdb_id").sum()),
        "ok_pct_total": round(100.0 * ok_n / total, 2) if total else 0.0,
        "overview_pct_ok": pct(_filled(ok["overview"])),
        "genres_pct_ok": pct(_filled(ok["genres"])),
        "keywords_pct_ok": pct(_filled(ok["keywords"])),
        "release_year_pct_ok": pct(ok["release_year"].notna()),
        "title_pct_ok": pct(_filled(ok["title"])),
    }


def coverage_to_markdown(
    stats: dict[str, float | int],
    parquet_path: Path | str,
) -> str:
    """Render coverage report as Markdown."""
    total = int(stats["total_rows"])
    path_str = Path(parquet_path).as_posix()

    def share(count: float | int) -> float:
        return round(100.0 * float(count) / total, 2) if total else 0.0

    return f"""# Relatório de cobertura — metadados TMDB (P.8)

> Gerado a partir de `{path_str}`.

## Resumo da coleta (`fetch_status`)

| Status | Quantidade | % do total |
|--------|------------|------------|
| **ok** | {stats["ok"]} | {stats["ok_pct_total"]}% |
| not_found | {stats["not_found"]} | {share(stats["not_found"])}% |
| missing_tmdb_id | {stats["missing_tmdb_id"]} | {share(stats["missing_tmdb_id"])}% |
| api_error | {stats["api_error"]} | {share(stats["api_error"])}% |
| **Total** | {total} | 100% |

## Campos preenchidos (somente filmes com `fetch_status=ok`)

| Campo | % com dado |
|-------|------------|
| title | {stats["title_pct_ok"]}% |
| overview (sinopse) | {stats["overview_pct_ok"]}% |
| genres | {stats["genres_pct_ok"]}% |
| keywords | {stats["keywords_pct_ok"]}% |
| release_year | {stats["release_year_pct_ok"]}% |

## Interpretação

- **overview** alimenta BERTopic / embeddings de texto no `feature_eng`.
- **not_found** / **missing_tmdb_id**: filme segue só com MovieLens (`movie.csv`, tags).
- Cobertura **≥ 95%** em `ok` é adequada para o Tech Challenge.
"""


def build_validation_sample(
    metadata: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    size: int = 10,
) -> pd.DataFrame:
    """Pick stratified sample for manual validation (P.9)."""
    merged = metadata.merge(
        movies[["movieId", "title"]],
        left_on="movie_id",
        right_on="movieId",
        how="left",
        suffixes=("_tmdb", "_ml"),
    )
    if "title_ml" in merged.columns:
        merged = merged.rename(columns={"title_ml": "title_movielens"})
    chunks: list[pd.DataFrame] = []
    per_group = max(1, size // 3)
    for status in merged["fetch_status"].unique():
        group = merged[merged["fetch_status"] == status]
        n = min(len(group), per_group)
        chunks.append(group.sample(n=n, random_state=42))
    sample = pd.concat(chunks).drop_duplicates(subset=["movie_id"]).head(size)
    if len(sample) < size:
        rest = merged[~merged["movie_id"].isin(sample["movie_id"])]
        need = min(size - len(sample), len(rest))
        if need > 0:
            sample = pd.concat([sample, rest.sample(n=need, random_state=42)])
    return sample.head(size)


def validation_sample_to_markdown(sample: pd.DataFrame) -> str:
    """Render P.9 checklist with TMDB links."""
    lines = [
        "# Amostra para validação manual — 10 filmes (P.9)",
        "",
        "Conferir no site TMDB se título e sinopse batem com o esperado.",
        "",
    ]
    for _, row in sample.iterrows():
        tmdb_id = row.get("tmdb_id")
        if pd.notna(tmdb_id):
            link = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
        else:
            link = "—"
        overview = row.get("overview") or ""
        preview = (overview[:120] + "…") if len(str(overview)) > 120 else overview
        tmdb_title = row.get("title_tmdb") or row.get("title") or "—"
        lines.extend(
            [
                f"## movieId={row['movie_id']} · status=`{row['fetch_status']}`",
                "",
                "- [ ] Conferido",
                f"- **MovieLens:** {row.get('title_movielens', '—')}",
                f"- **TMDB (parquet):** {tmdb_title}",
                f"- **Link:** {link}",
                f"- **Sinopse (cache):** {preview or '—'}",
                "",
            ]
        )
    lines.append("**Responsável:** marcar `[x]` após revisão da equipe.")
    return "\n".join(lines)
