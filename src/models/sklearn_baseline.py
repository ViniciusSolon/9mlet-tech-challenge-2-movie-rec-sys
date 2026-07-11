"""Scikit-learn baseline stub (training in Bloco 5)."""

from __future__ import annotations

from typing import Any

from models.base import RecommenderModel


class SklearnBaselineRecommender(RecommenderModel):
    """Placeholder baseline until NMF/SVD is wired."""

    def __init__(self) -> None:
        self._fitted = False

    @property
    def name(self) -> str:
        return "sklearn_baseline"

    def fit(
        self, features: Any, targets: Any | None = None
    ) -> SklearnBaselineRecommender:
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[float]:
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        size = len(features) if hasattr(features, "__len__") else 1
        return [0.0] * size



class MostPopularRecommender(RecommenderModel):
    """Non-personalised baseline that recommends globally popular items.

    Popularity is defined as the count of interactions (ratings) per item
    in the training set.

    Args:
        top_k: Number of top items to store internally.
    """

    def __init__(self, top_k: int = 100) -> None:
        self._top_k = top_k
        self._popular_items: list[int] = []
        self._fitted = False

    @property
    def name(self) -> str:
        """Human-readable model identifier."""
        return "most_popular"

    def fit(
        self, features: pd.DataFrame, targets: Any | None = None
    ) -> MostPopularRecommender:
        """Rank items by interaction count and store the top-K.

        Args:
            features: DataFrame with at least a ``movie_idx`` column.
            targets: Ignored.

        Returns:
            Self.
        """
        counts = features["movie_idx"].value_counts()
        self._popular_items = counts.head(self._top_k).index.tolist()
        self._fitted = True
        return self

    def predict(self, features: Any) -> list[int]:
        """Return the globally popular item list (same for every user).

        Args:
            features: Ignored; included for API compatibility.

        Returns:
            List of top-K item indices ordered by popularity.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        return list(self._popular_items)

    def recommend(self, user_idx: int, seen_items: set[int], k: int) -> list[int]:
        """Return top-K popular items unseen by the user.

        Args:
            user_idx: User index (ignored in non-personalised baseline).
            seen_items: Items already interacted with by the user.
            k: Number of recommendations to return.

        Returns:
            Ordered list of up to K item indices.
        """
        if not self._fitted:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        recs = [i for i in self._popular_items if i not in seen_items]
        return recs[:k]


class NMFRecommender(RecommenderModel):
    """NMF-based collaborative filter using scikit-learn.

    Decomposes the user-item interaction matrix into latent factors and
    predicts unseen ratings as the dot product of user and item factors.

    Args:
        n_components: Number of latent components.
        max_iter: Maximum NMF optimisation iterations.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 50,
        max_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self._n_components = n_components
        self._max_iter = max_iter
        self._random_state = random_state
        self._model = NMF(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        """Human-readable model identifier."""
        return "nmf_baseline"

    def fit(
        self, features: pd.DataFrame, targets: Any | None = None
    ) -> NMFRecommender:
        """Build the user-item matrix and factorise it with NMF.

        Args:
            features: DataFrame with columns ``user_idx``, ``movie_idx``,
                ``rating``.
            targets: Ignored.

        Returns:
            Self.
        """
        n_users = int(features["user_idx"].max() + 1)
        n_items = int(features["movie_idx"].max() + 1)
        rows = features["user_idx"].values
        cols = features["movie_idx"].values
        data = features["rating"].values.astype(float)
        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self._user_factors = self._model.fit_transform(matrix)
        self._item_factors = self._model.components_.T
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> list[float]:
        """Predict ratings for (user, item) pairs.

        Args:
            features: DataFrame with columns ``user_idx`` and ``movie_idx``.

        Returns:
            List of predicted scores.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted or self._user_factors is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        user_f = self._user_factors[features["user_idx"].values]
        item_f = self._item_factors[features["movie_idx"].values]
        return (user_f * item_f).sum(axis=1).tolist()

    def recommend(self, user_idx: int, seen_items: set[int], k: int) -> list[int]:
        """Return top-K unseen items for a user based on predicted scores.

        Args:
            user_idx: Index of the target user.
            seen_items: Items already interacted with.
            k: Number of recommendations.

        Returns:
            Ordered list of up to K item indices.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted or self._item_factors is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        user_vec = self._user_factors[user_idx]
        scores = self._item_factors.dot(user_vec)
        ranked = np.argsort(-scores)
        return [int(i) for i in ranked if i not in seen_items][:k]
