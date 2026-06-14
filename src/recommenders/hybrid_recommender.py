from typing import List, Tuple
from .base_recommender import BaseRecommender
from ..models import Rating


class HybridRecommender(BaseRecommender):
    def __init__(self, model_a: BaseRecommender, model_b: BaseRecommender, weight_a: float = 0.5) -> None:
        """
        model_a: Pierwszy model (np. SVD lub Node2Vec)
        model_b: Drugi model (np. KNN)
        weight_a: Waga dla pierwszego modelu. Waga drugiego to (1.0 - weight_a).
        """
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
        print(12345)
        pred_b = self.model_b.predict(user_idx, item_idx)

        # WYRÓWNANIE SKALI (Zachowane z Twojego kodu)
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
        # --- ZOPTYMALIZOWANE GENEROWANIE RANKINGU HYBRYDOWEGO ---
        
        # Pobieramy szerszy kontekst z obu modeli (np. top 200), żeby mieć pewność, 
        # że znajdziemy wspólne książki do zsumowania
        candidates_k = max(top_k * 4, 200)
        
        ranking_a = self.model_a.create_ranking(user_idx, top_k=candidates_k)
        ranking_b = self.model_b.create_ranking(user_idx, top_k=candidates_k)
        
        if not ranking_a and not ranking_b:
            return []

        # Słownik na połączone oceny: {isbn: final_score}
        combined_scores = {}

        # Mnożnik skali dla Node2Vec
        scale_a = 10.0 if self.model_a.__class__.__name__ == "Node2VecRecommender" else 1.0
        scale_b = 10.0 if self.model_b.__class__.__name__ == "Node2VecRecommender" else 1.0

        # Przetwarzamy wyniki z modelu A
        for item_idx, score in ranking_a:
            combined_scores[item_idx] = combined_scores.get(item_idx, 0.0) + (score * scale_a * self.weight_a)

        # Przetwarzamy wyniki z modelu B
        for item_idx, score in ranking_b:
            combined_scores[item_idx] = combined_scores.get(item_idx, 0.0) + (score * scale_b * self.weight_b)

        # Sortujemy połączone wyniki malejąco
        predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]