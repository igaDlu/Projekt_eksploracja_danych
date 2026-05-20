import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.decomposition import TruncatedSVD
from .base_recommender import BaseRecommender
from ..models import Rating


class SVDRecommender(BaseRecommender):
    def __init__(self, n_components: int = 12, kind: str = "user") -> None:
        """
        n_components: Liczba cech ukrytych (latent features) – dokładnie jak w Twoim notatniku.
        kind: "user" (User-Based SVD) lub "item" (Item-Based SVD).
        """
        super().__init__(kind=kind)
        self.n_components = n_components
        self.pivot_table = None
        self.predictions_df = None

    def fit(self, ratings: List[Rating]) -> None:
        df = pd.DataFrame([{'user_id': r.user_id, 'isbn': r.isbn, 'rating': r.rating} for r in ratings])

        df = df[df['rating'] > 0]

        if self.kind == "user":
            self.pivot_table = df.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
        else:

            self.pivot_table = df.pivot(index='isbn', columns='user_id', values='rating').fillna(0)

        # Inicjalizacja i dopasowanie modelu TruncatedSVD ze scikit-learn
        max_components = min(self.pivot_table.shape) - 1
        actual_components = min(self.n_components, max_components)

        svd = TruncatedSVD(n_components=actual_components, random_state=42)

        matrix_reduced = svd.fit_transform(self.pivot_table.values)
        matrix_reconstructed = svd.inverse_transform(matrix_reduced)

        self.predictions_df = pd.DataFrame(
            matrix_reconstructed,
            index=self.pivot_table.index,
            columns=self.pivot_table.columns
        )

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.predictions_df is None:
            raise ValueError("Model nie został jeszcze wytrenowany. Wywołaj najpierw metodę fit().")

        # Obsługa przypadków "Zimnego startu"
        if self.kind == "user":
            if user_idx not in self.predictions_df.index or item_idx not in self.predictions_df.columns:
                return 0.0
            return float(self.predictions_df.loc[user_idx, item_idx])
        else:
            if item_idx not in self.predictions_df.index or user_idx not in self.predictions_df.columns:
                return 0.0
            return float(self.predictions_df.loc[item_idx, user_idx])

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        raise NotImplementedError("SVD wymaga ponownego przeliczenia macierzy (metoda fit) po dodaniu nowych ocen.")

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.pivot_table is None:
            raise ValueError("Model nie został jeszcze wytrenowany.")

        if self.kind == "user":
            if user_idx not in self.pivot_table.index:
                return []
            user_ratings = self.pivot_table.loc[user_idx]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
        else:
            if user_idx not in self.pivot_table.columns:
                return []
            user_ratings = self.pivot_table.loc[:, user_idx]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()

        predictions = []
        for item_idx in unrated_items:
            pred_score = self.predict(user_idx, item_idx)
            pred_score = max(0.0, pred_score)
            predictions.append((item_idx, pred_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]