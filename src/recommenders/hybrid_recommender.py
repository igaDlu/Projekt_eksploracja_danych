from typing import List, Tuple
from .base_recommender import BaseRecommender
from ..models import Rating


class HybridRecommender(BaseRecommender):
    def __init__(self, model_a: BaseRecommender, model_b: BaseRecommender, weight_a: float = 0.5) -> None:
        """
        model_a: Pierwszy model (np. SVD)
        model_b: Drugi model (np. Node2Vec)
        weight_a: Waga dla pierwszego modelu. Waga drugiego to (1.0 - weight_a).
        """
        # Ustawiamy 'kind' na podstawie pierwszego modelu, żeby zachować spójność trybu
        super().__init__(kind=model_a.kind)
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def fit(self, ratings: List[Rating]) -> None:
        print(f"  -> Trenowanie składowej A: {self.model_a.__class__.__name__}...")
        self.model_a.fit(ratings)
        print(f"  -> Trenowanie składowej B: {self.model_b.__class__.__name__}...")
        self.model_b.fit(ratings)

    def predict(self, user_idx: int, item_idx: int) -> float:
        pred_a = self.model_a.predict(user_idx, item_idx)
        pred_b = self.model_b.predict(user_idx, item_idx)

        # WYRÓWNANIE SKALI
        if self.model_a.__class__.__name__ == "Node2VecRecommender":
            pred_a *= 10.0
        if self.model_b.__class__.__name__ == "Node2VecRecommender":
            pred_b *= 10.0

        final_prediction = (pred_a * self.weight_a) + (pred_b * self.weight_b)
        return float(final_prediction)

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        self.model_a.rate(user_idx, item_idx, score)
        self.model_b.rate(user_idx, item_idx, score)

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if not hasattr(self.model_a, 'pivot_table') or self.model_a.pivot_table is None:
            return []

        if self.kind == "user":
            if user_idx not in self.model_a.pivot_table.index:
                return []
            user_ratings = self.model_a.pivot_table.loc[user_idx]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
        else:
            if user_idx not in self.model_a.pivot_table.columns:
                return []
            user_ratings = self.model_a.pivot_table.loc[:, user_idx]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()

        # Obliczamy hybrydowe predykcje
        predictions = []
        for item_idx in unrated_items:
            pred_score = self.predict(user_idx, item_idx)
            if pred_score > 0.0:
                predictions.append((item_idx, pred_score))

        # Sortujemy malejąco
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]