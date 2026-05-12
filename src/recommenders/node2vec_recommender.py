from base_recommender import BaseRecommender
from typing import List, Tuple
from ..models import Rating

class Node2VecRecommender(BaseRecommender):
    def __init__(self):
        self.graph = None  # Placeholder na graf
        self.embeddings = None  # Placeholder na wygenerowane wektory

    def fit(self, ratings: List[Rating]) -> None:
        pass

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        pass

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        pass