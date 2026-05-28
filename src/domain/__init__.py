"""Domain entities for the recommendation system."""

from domain.ids import MovieId, UserId
from domain.rating import Rating
from domain.recommendation import Recommendation, RecommendationList

__all__ = [
    "MovieId",
    "UserId",
    "Rating",
    "Recommendation",
    "RecommendationList",
]
