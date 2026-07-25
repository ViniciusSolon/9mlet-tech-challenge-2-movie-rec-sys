#!/usr/bin/env python
"""Demo texto: histórico de filmes+notas → top-K com título e sinopse.

Não é um LLM generativo. Usa o checkpoint PyTorch (embeddings de item) +
metadados TMDB já no projeto para montar recomendações personalizadas
quando o usuário ainda não tem ``user_idx`` no treino (cold start).

Uso:
    python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json
    python llm/recommend_from_history.py --input llm/examples/historico_exemplo.json --k 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.factory import create_model  # noqa: E402

_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recomenda filmes a partir de um histórico em texto/JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON com historico: [{titulo, nota}, ...]",
    )
    parser.add_argument("--k", type=int, default=None, help="Quantidade de recomendações.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "model.pth",
        help="Checkpoint PyTorch.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "processed" / "enriched_metadata.parquet",
        help="Parquet com título e sinopse.",
    )
    return parser.parse_args()


def _normalize_title(title: str) -> str:
    text = title.strip().lower()
    text = _YEAR_RE.sub("", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_catalog(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")
    frame = pd.read_parquet(metadata_path)
    frame = frame.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
    frame["title_norm"] = frame["title"].astype(str).map(_normalize_title)
    frame["overview"] = frame["overview"].fillna("Sinopse indisponível.")
    return frame


def _build_movie_index(catalog: pd.DataFrame) -> dict[int, int]:
    """Mesma regra do feature_eng: ordem de aparição em metadata."""
    return {int(mid): i for i, mid in enumerate(catalog["movieId"].tolist())}


def _resolve_title(query: str, catalog: pd.DataFrame) -> pd.Series | None:
    q = _normalize_title(query)
    if not q:
        return None
    exact = catalog.loc[catalog["title_norm"] == q]
    if len(exact) == 1:
        return exact.iloc[0]
    starts = catalog.loc[catalog["title_norm"].str.startswith(q, na=False)]
    if len(starts) == 1:
        return starts.iloc[0]
    contains = catalog.loc[catalog["title_norm"].str.contains(re.escape(q), na=False)]
    if len(contains) >= 1:
        scored = [
            (SequenceMatcher(None, q, row.title_norm).ratio(), idx)
            for idx, row in contains.iterrows()
        ]
        scored.sort(reverse=True)
        return catalog.loc[scored[0][1]]
    best_idx = None
    best_ratio = 0.0
    for idx, row in catalog.iterrows():
        ratio = SequenceMatcher(None, q, row.title_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    if best_idx is not None and best_ratio >= 0.55:
        return catalog.loc[best_idx]
    return None


def _load_model(model_path: Path, n_users: int, n_items: int) -> torch.nn.Module:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    emb_dim = state["user_embedding.weight"].shape[1]
    model = create_model(
        "torch_mlp",
        n_users=n_users,
        n_items=n_items,
        embedding_dim=emb_dim,
        hidden_dim=128,
        dropout=0.2,
    )
    model.load_state_dict(state)
    model.fit([], None)
    model.eval()
    return model


def _match_history(
    history: list[dict],
    catalog: pd.DataFrame,
    movie_index: dict[int, int],
) -> tuple[list[dict], list[str]]:
    matched: list[dict] = []
    missing: list[str] = []
    for row in history:
        title = str(row.get("titulo") or row.get("title") or "").strip()
        rating = float(row.get("nota") or row.get("rating") or 0.0)
        hit = _resolve_title(title, catalog)
        if hit is None:
            missing.append(title)
            continue
        movie_id = int(hit["movieId"])
        if movie_id not in movie_index:
            missing.append(title)
            continue
        matched.append(
            {
                "query": title,
                "movieId": movie_id,
                "movie_idx": movie_index[movie_id],
                "title": str(hit["title"]),
                "rating": rating,
                "overview": str(hit["overview"]),
            }
        )
    return matched, missing


def _user_profile(
    model: torch.nn.Module,
    matched: list[dict],
) -> torch.Tensor:
    """Perfil cold-start = média ponderada dos embeddings dos filmes curtidos."""
    item_weight = model.item_embedding.weight.detach()
    vectors: list[torch.Tensor] = []
    weights: list[float] = []
    for row in matched:
        weight = max(row["rating"] - 2.5, 0.1)
        vectors.append(item_weight[row["movie_idx"]])
        weights.append(weight)
    stacked = torch.stack(vectors)
    w = torch.tensor(weights, dtype=stacked.dtype).unsqueeze(1)
    profile = (stacked * w).sum(dim=0) / w.sum()
    return F.normalize(profile, dim=0)


def _recommend(
    model: torch.nn.Module,
    profile: torch.Tensor,
    catalog: pd.DataFrame,
    movie_index: dict[int, int],
    seen_ids: set[int],
    k: int,
) -> list[dict]:
    item_weight = F.normalize(model.item_embedding.weight.detach(), dim=1)
    scores = item_weight @ profile
    order = torch.argsort(scores, descending=True).tolist()
    reverse = {idx: mid for mid, idx in movie_index.items()}
    id_to_row = catalog.set_index("movieId")

    results: list[dict] = []
    for movie_idx in order:
        movie_id = reverse.get(int(movie_idx))
        if movie_id is None or movie_id in seen_ids:
            continue
        row = id_to_row.loc[movie_id]
        overview = row["overview"]
        if not isinstance(overview, str) or not overview.strip():
            overview = "Sinopse indisponível."
        year = row.get("release_year")
        results.append(
            {
                "rank": len(results) + 1,
                "movieId": int(movie_id),
                "titulo": str(row["title"]),
                "score": float(scores[movie_idx]),
                "sinopse": overview.strip(),
                "ano": None if pd.isna(year) else int(year),
            }
        )
        if len(results) >= k:
            break
    return results


def _print_report(
    user_label: str,
    matched: list[dict],
    missing: list[str],
    recommendations: list[dict],
) -> None:
    print("=" * 72)
    print(f"Recomendações para: {user_label}")
    print("=" * 72)
    print("\nHistórico reconhecido:")
    for row in matched:
        print(f"  • {row['title']}  (nota {row['rating']})  ← buscou '{row['query']}'")
    if missing:
        print("\nNão encontrados no catálogo:")
        for title in missing:
            print(f"  • {title}")

    print(f"\nTop {len(recommendations)} indicações:\n")
    for item in recommendations:
        ano = f" ({item['ano']})" if item["ano"] else ""
        print(f"{item['rank']:02d}. {item['titulo']}{ano}")
        print(f"    score={item['score']:.3f}")
        sinopse = item["sinopse"]
        if len(sinopse) > 280:
            sinopse = sinopse[:277] + "..."
        print(f"    {sinopse}\n")


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    history = payload.get("historico") or payload.get("history") or []
    if not history:
        print("JSON sem 'historico'.")
        return 1
    k = int(args.k or payload.get("k") or 10)
    user_label = str(payload.get("usuario") or payload.get("user") or "usuário")

    catalog = _load_catalog(args.metadata)
    movie_index = _build_movie_index(catalog)
    n_items = len(movie_index)
    # n_users só precisa bater com o checkpoint
    state = torch.load(args.model, map_location="cpu")
    n_users = state["user_embedding.weight"].shape[0]
    model = _load_model(args.model, n_users=n_users, n_items=n_items)

    matched, missing = _match_history(history, catalog, movie_index)
    if not matched:
        print("Nenhum filme do histórico foi encontrado no catálogo.")
        return 1

    profile = _user_profile(model, matched)
    seen = {row["movieId"] for row in matched}
    recommendations = _recommend(
        model, profile, catalog, movie_index, seen, k=k
    )
    _print_report(user_label, matched, missing, recommendations)

    out_path = args.input.with_name(args.input.stem + "_recomendacoes.json")
    out_path.write_text(
        json.dumps(
            {
                "usuario": user_label,
                "historico_resolvido": matched,
                "nao_encontrados": missing,
                "recomendacoes": recommendations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"JSON salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
