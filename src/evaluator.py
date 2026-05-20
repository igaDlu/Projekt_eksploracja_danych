import numpy as np
from typing import List, Dict, Tuple
from .models import Rating
from .recommenders.base_recommender import BaseRecommender


class Evaluator:
    def __init__(self):
        pass

    def calculate_hits_at_k(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Sprawdza, czy w Top-K znalazła się przynajmniej jedna trafiona książka (1.0 lub 0.0)."""
        top_k_predicted = predicted[:k]
        # Sprawdzamy część wspólną zestawu książek przeczytanych i rekomendowanych
        for item in top_k_predicted:
            if item in actual:
                return 1.0
        return 0.0

    def calculate_mrr(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Ocenia, jak wysoko znalazło się pierwsze trafienie (Mean Reciprocal Rank)."""
        top_k_predicted = predicted[:k]
        for rank, item in enumerate(top_k_predicted, start=1):
            if item in actual:
                return 1.0 / rank  # Im wyższa pozycja (niższy rank), tym większa wartość
        return 0.0

    def calculate_ndcg(self, actual: List[int], predicted: List[int], k: int) -> float:
        """Ocenia pozycję i trafność rekomendacji w Top-K (Normalized Discounted Cumulative Gain)."""
        top_k_predicted = predicted[:k]

        dcg = 0.0
        for rank, item in enumerate(top_k_predicted, start=1):
            if item in actual:
                # Klasyczny wzór na DCG dla binarnego trafienia (w zzbiorze testowym traktujemy pozycję jako 'trafił/nie trafił')
                dcg += 1.0 / np.log2(rank + 1)

        if dcg == 0.0:
            return 0.0

        # Idealne DCG (IDCG)
        idcg = 0.0
        actual_hits_count = min(len(actual), k)
        for rank in range(1, actual_hits_count + 1):
            idcg += 1.0 / np.log2(rank + 1)

        return dcg / idcg

    def evaluate(self, recommender: BaseRecommender, test_ratings: List[Rating], top_k: int = 10) -> Dict[str, float]:
        """
        Główna metoda testowa. Grupuje oceny testowe po użytkownikach,
        generuje dla nich rankingi i wylicza średnie wartości metryk.
        """
        print(f"Uruchamianie ewaluacji modelu dla Top-{top_k}...")

        # 1. Grupujemy rzeczywiste interakcje ze zbioru testowego po użytkownikach
        user_test_profile = {}
        for r in test_ratings:
            if r.user_id not in user_test_profile:
                user_test_profile[r.user_id] = []
            user_test_profile[r.user_id].append(r.isbn)  # r.isbn zawiera tutaj nasz wewnętrzny indeks item_idx

        hits_list = []
        mrr_list = []
        ndcg_list = []

        # 2. Iterujemy po każdym użytkowniku ze zbioru testowego
        total_users = len(user_test_profile)
        processed = 0

        for user_id, actual_items in user_test_profile.items():
            # Generujemy rekomendacje z modelu (bierzemy tylko ID książek z krotek)
            ranking_raw = recommender.create_ranking(user_idx=user_id, top_k=top_k)
            predicted_items = [item_idx for item_idx, _ in ranking_raw]

            if not predicted_items:
                continue

            # Obliczamy metryki dla tego konkretnego użytkownika
            hits_list.append(self.calculate_hits_at_k(actual_items, predicted_items, top_k))
            mrr_list.append(self.calculate_mrr(actual_items, predicted_items, top_k))
            ndcg_list.append(self.calculate_ndcg(actual_items, predicted_items, top_k))

            processed += 1
            if processed % 100 == 0 or processed == total_users:
                print(f"Przetworzono użytkowników: {processed}/{total_users}")

        # 3. Zwracamy uśrednione wyniki globalne
        return {
            f"HIT@{top_k}": float(np.mean(hits_list)) if hits_list else 0.0,
            f"MRR@{top_k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
            f"NDCG@{top_k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0
        }