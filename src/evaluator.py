from recommenders.base_recommender import BaseRecommender
from typing import List, Dict
from models import Rating


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, recommender: BaseRecommender, test_ratings: List[Rating], top_k: int = 10) -> Dict[str, float]:
        """
        Główna metoda. Przepytuje recommender (używając metody create_ranking)
        i porównuje wyniki z test_ratings. Zwraca słownik z uśrednionymi metrykami.
        """
        pass

    def calculate_ndcg(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Ocenia pozycję i trafność rekomendacji w Top-K."""
        pass

    def calculate_mrr(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Ocenia, jak wysoko znalazło się pierwsze trafienie (Mean Reciprocal Rank)."""
        pass

    def calculate_hits_at_k(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Sprawdza, czy w Top-K znalazła się przynajmniej jedna trafiona książka."""
        pass