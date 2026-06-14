import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from .base_recommender import BaseRecommender
from ..models import Rating


class KNNRecommender(BaseRecommender):
    def __init__(self, n_neighbors: int = 5, weights: str = 'distance', kind: str = "user") -> None:
        """
        n_neighbors: Liczba sąsiadów braku pod uwagę (k z kNN).
        weights: Sposób ważenia sąsiadów ('uniform' lub 'distance').
        kind: "user" (User-Based Collaborative Filtering) lub "item" (Item-Based).
        """
        super().__init__(kind=kind)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = NearestNeighbors(metric='cosine', algorithm='auto', n_jobs=-1)
        self.pivot_table = None

    def fit(self, ratings: List[Rating]) -> None:
        df = pd.DataFrame([{'user_id': r.user_id, 'isbn': r.isbn, 'rating': r.rating} for r in ratings])

        if self.kind == "user":
            self.pivot_table = df.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
        else:
            self.pivot_table = df.pivot(index='isbn', columns='user_id', values='rating').fillna(0)

        self.model.fit(self.pivot_table.values)

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.pivot_table is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj fit().")

        # Obsługa zimnego startu
        if self.kind == "user":
            if user_idx not in self.pivot_table.index or item_idx not in self.pivot_table.columns:
                return 0.0
        else:
            if item_idx not in self.pivot_table.index or user_idx not in self.pivot_table.columns:
                return 0.0

        if self.kind == "user":
            # --- PODEJŚCIE USER-BASED ---
            user_vector = self.pivot_table.loc[user_idx].values.reshape(1, -1)
            # Wyciągamy sąsiadów (użytkowników).
            distances, indices = self.model.kneighbors(user_vector,
                                                       n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))

            neighbor_indices = self.pivot_table.index[indices.flatten()[1:]]
            neighbor_distances = distances.flatten()[1:]

            # Zamieniamy odległość (distance) na podobieństwo (similarity)
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = 1.0 - neighbor_distances

            if np.sum(similarities) == 0:
                return 0.0

            # Sprawdzamy, jak podobni użytkownicy ocenili wybraną książkę (item_idx)
            neighbor_ratings = self.pivot_table.loc[neighbor_indices, item_idx].values

            # Średnia ważona ocen sąsiadów
            predicted_rating = np.sum(neighbor_ratings * similarities) / np.sum(similarities)
            return float(predicted_rating)

        else:
            # --- PODEJŚCIE ITEM-BASED ---
            item_vector = self.pivot_table.loc[item_idx].values.reshape(1, -1)
            # Wyciągamy sąsiednie książki
            distances, indices = self.model.kneighbors(item_vector,
                                                       n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))

            neighbor_indices = self.pivot_table.index[indices.flatten()[1:]]
            neighbor_distances = distances.flatten()[1:]

            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = 1.0 - neighbor_distances

            if np.sum(similarities) == 0:
                return 0.0

            # Sprawdzamy, jak nasz użytkownik (user_idx) ocenił sąsiednie książki
            user_ratings_for_neighbors = self.pivot_table.loc[neighbor_indices, user_idx].values

            predicted_rating = np.sum(user_ratings_for_neighbors * similarities) / np.sum(similarities)
            return float(predicted_rating)

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        raise NotImplementedError("KNN wymaga ponownego przeliczenia macierzy przy nowej ocenie.")

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.pivot_table is None:
            return []

        # Wyciągamy nieocenione pozycje
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
            if pred_score > 0.0:
                predictions.append((item_idx, pred_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]