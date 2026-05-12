from base_recommender import BaseRecommender
from typing import List, Tuple
from ..models import Rating

class SVDRecommender(BaseRecommender):
    def __init__(self, n_components: int = 12):
        self.n_components = n_components
        self.corr_matrix = None
        self.book_indices = None

    def fit(self, ratings: List[Rating]) -> None:
        pass

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        pass

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        pass