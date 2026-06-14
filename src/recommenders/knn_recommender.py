import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from .base_recommender import BaseRecommender
from ..models import Rating


class KNNRecommender(BaseRecommender):
    def __init__(self, n_neighbors: int = 5, weights: str = 'distance', kind: str = "user") -> None:
        """
        n_neighbors: Liczba sąsiadów brana pod uwagę (k z kNN).
        weights: Sposób ważenia sąsiadów ('uniform' lub 'distance').
        kind: "user" (User-Based Collaborative Filtering) lub "item" (Item-Based).
        """
        super().__init__(kind=kind)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = NearestNeighbors(metric='cosine', algorithm='auto', n_jobs=-1)
        self.pivot_table = None
        self.pivot_matrix = None

    def fit(self, ratings: List[Rating]) -> None:
        df = pd.DataFrame([{'user_id': r.user_id, 'isbn': r.isbn, 'rating': r.rating} for r in ratings])

        if self.kind == "user":
            self.pivot_table = df.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
        else:
            self.pivot_table = df.pivot(index='isbn', columns='user_id', values='rating').fillna(0)

        self.pivot_matrix = csr_matrix(self.pivot_table.values)
        self.model.fit(self.pivot_matrix)

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.pivot_table is None or self.pivot_matrix is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj fit().")

        if self.kind == "user":
            if user_idx not in self.pivot_table.index or item_idx not in self.pivot_table.columns:
                return 0.0

            row_idx = self.pivot_table.index.get_loc(user_idx)
            user_vector = self.pivot_matrix[row_idx]
            n_neighbors = min(self.n_neighbors + 1, self.pivot_matrix.shape[0])
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors)

            neighbor_positions = indices.flatten()[1:]
            neighbor_distances = distances.flatten()[1:]
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = np.clip(1.0 - neighbor_distances, 0.0, 1.0)

            total_similarity = np.sum(similarities)
            if total_similarity == 0:
                return 0.0

            item_col = self.pivot_table.columns.get_loc(item_idx)
            neighbor_ratings = self.pivot_matrix[neighbor_positions, item_col].toarray().flatten()
            predicted_rating = float(np.dot(similarities, neighbor_ratings) / total_similarity)
            return predicted_rating

        else:
            if item_idx not in self.pivot_table.index or user_idx not in self.pivot_table.columns:
                return 0.0

            row_idx = self.pivot_table.index.get_loc(item_idx)
            item_vector = self.pivot_matrix[row_idx]
            n_neighbors = min(self.n_neighbors + 1, self.pivot_matrix.shape[0])
            distances, indices = self.model.kneighbors(item_vector, n_neighbors=n_neighbors)

            neighbor_positions = indices.flatten()[1:]
            neighbor_distances = distances.flatten()[1:]
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = np.clip(1.0 - neighbor_distances, 0.0, 1.0)

            total_similarity = np.sum(similarities)
            if total_similarity == 0:
                return 0.0

            user_col = self.pivot_table.columns.get_loc(user_idx)
            neighbor_ratings = self.pivot_matrix[neighbor_positions, user_col].toarray().flatten()
            predicted_rating = float(np.dot(similarities, neighbor_ratings) / total_similarity)
            return predicted_rating

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        raise NotImplementedError("KNN wymaga ponownego przeliczenia macierzy przy nowej ocenie.")

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.pivot_table is None or self.pivot_matrix is None:
            return []

        if self.kind == "user":
            if user_idx not in self.pivot_table.index:
                return []

            row_idx = self.pivot_table.index.get_loc(user_idx)
            user_vector = self.pivot_matrix[row_idx]
            n_neighbors = min(self.n_neighbors + 1, self.pivot_matrix.shape[0])
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors)

            neighbor_positions = indices[0, 1:]
            neighbor_distances = distances[0, 1:]
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = np.clip(1.0 - neighbor_distances, 0.0, 1.0)

            total_similarity = np.sum(similarities)
            if total_similarity == 0:
                return []

            neighbor_ratings = self.pivot_matrix[neighbor_positions, :].toarray()
            predicted_scores = (similarities.reshape(1, -1) @ neighbor_ratings).flatten() / total_similarity

            user_ratings = self.pivot_matrix[row_idx, :].toarray().flatten()
            unrated_mask = user_ratings == 0
            candidate_positions = np.where(unrated_mask)[0]
            if candidate_positions.size == 0:
                return []

            candidate_scores = predicted_scores[candidate_positions]
            top_size = min(top_k, candidate_scores.size)
            top_indices = np.argpartition(-candidate_scores, top_size - 1)[:top_size]
            top_order = np.argsort(-candidate_scores[top_indices])
            top_positions = candidate_positions[top_indices[top_order]]

            return [
                (self.pivot_table.columns[pos], float(candidate_scores[top_indices[top_order][i]]))
                for i, pos in enumerate(top_positions)
            ]

        else:
            if user_idx not in self.pivot_table.columns:
                return []

            user_col = self.pivot_table.columns.get_loc(user_idx)
            user_ratings = self.pivot_matrix[:, user_col].toarray().flatten()
            unrated_mask = user_ratings == 0
            if not np.any(user_ratings != 0):
                return []

            n_neighbors = min(self.n_neighbors + 1, self.pivot_matrix.shape[0])
            distances, indices = self.model.kneighbors(self.pivot_matrix, n_neighbors=n_neighbors)

            neighbor_distances = distances[:, 1:]
            neighbor_positions = indices[:, 1:]
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = np.clip(1.0 - neighbor_distances, 0.0, 1.0)

            weighted_ratings = similarities * user_ratings[neighbor_positions]
            similarity_sum = np.sum(similarities, axis=1)
            predicted_scores = np.zeros_like(similarity_sum)
            valid = similarity_sum != 0
            predicted_scores[valid] = np.sum(weighted_ratings[valid, :], axis=1) / similarity_sum[valid]

            candidate_positions = np.where(unrated_mask)[0]
            if candidate_positions.size == 0:
                return []

            candidate_scores = predicted_scores[candidate_positions]
            top_size = min(top_k, candidate_scores.size)
            top_indices = np.argpartition(-candidate_scores, top_size - 1)[:top_size]
            top_order = np.argsort(-candidate_scores[top_indices])
            top_positions = candidate_positions[top_indices[top_order]]

            return [
                (self.pivot_table.index[pos], float(candidate_scores[top_indices[top_order][i]]))
                for i, pos in enumerate(top_positions)
            ]
