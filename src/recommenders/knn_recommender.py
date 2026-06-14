import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from .base_recommender import BaseRecommender
from ..models import Rating


class KNNRecommender(BaseRecommender):
    def __init__(self, n_neighbors: int = 5, weights: str = 'distance', kind: str = "user") -> None:
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

        if self.kind == "user":
            if user_idx not in self.pivot_table.index or item_idx not in self.pivot_table.columns:
                return 0.0
            
            user_vector = self.pivot_table.loc[user_idx].values.reshape(1, -1)
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))
            
            neighbor_indices = self.pivot_table.index[indices.flatten()[1:]]
            neighbor_distances = distances.flatten()[1:]
            
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = 1.0 - neighbor_distances
                
            if np.sum(similarities) == 0: return 0.0
            
            neighbor_ratings = self.pivot_table.loc[neighbor_indices, item_idx].values
            return float(np.sum(neighbor_ratings * similarities) / np.sum(similarities))
        else:
            if item_idx not in self.pivot_table.index or user_idx not in self.pivot_table.columns:
                return 0.0
                
            item_vector = self.pivot_table.loc[item_idx].values.reshape(1, -1)
            distances, indices = self.model.kneighbors(item_vector, n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))
            
            neighbor_indices = self.pivot_table.index[indices.flatten()[1:]]
            neighbor_distances = distances.flatten()[1:]
            
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = 1.0 - neighbor_distances
                
            if np.sum(similarities) == 0: return 0.0
            
            user_ratings_for_neighbors = self.pivot_table.loc[neighbor_indices, user_idx].values
            return float(np.sum(user_ratings_for_neighbors * similarities) / np.sum(similarities))

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        raise NotImplementedError("KNN wymaga ponownego przeliczenia macierzy.")

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.pivot_table is None:
            return []

        # --- ZOPTYMALIZOWANA MACIERZOWO METODA GENEROWANIA RANKINGU ---
        if self.kind == "user":
            if user_idx not in self.pivot_table.index:
                return []
            
            # 1. Pobieramy profil użytkownika i sprawdzamy co czytał
            user_ratings = self.pivot_table.loc[user_idx].values
            unrated_mask = (user_ratings == 0)
            
            # 2. Szukamy sąsiadów TYLKO RAZ dla całego rankingu
            user_vector = user_ratings.reshape(1, -1)
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))
            
            # Pomijamy samego siebie (pierwszy element)
            neighbor_locs = indices.flatten()[1:]
            neighbor_distances = distances.flatten()[1:]
            
            if self.weights == 'uniform':
                similarities = np.ones_like(neighbor_distances)
            else:
                similarities = 1.0 - neighbor_distances
                
            sum_sim = np.sum(similarities)
            if sum_sim == 0:
                return []
                
            # 3. Wyciągamy oceny wszystkich sąsiadów dla WSZYSTKICH książek naraz (wycinek macierzy)
            # matrix.values[neighbor_locs] ma kształt (n_neighbors, n_books)
            neighbor_matrix = self.pivot_table.values[neighbor_locs, :]
            
            # 4. Szybki iloczyn macierzowy (średnia ważona dla wszystkich pozycji w ułamku sekundy)
            all_predictions = (similarities @ neighbor_matrix) / sum_sim
            
            # Wyciągamy tylko nieocenione książki
            book_ids = self.pivot_table.columns.values
            
        else:
            # Item-based optimization
            if user_idx not in self.pivot_table.columns:
                return []
                
            user_col = self.pivot_table.loc[:, user_idx].values
            unrated_mask = (user_col == 0)
            book_ids = self.pivot_table.index.values
            
            # W item-based dla każdego nieocenionego przedmiotu i tak musielibyśmy pytać o sąsiadów,
            # ale możemy to zrobić sprytnie: pobieramy odległości dla WSZYSTKICH książek naraz
            distances, indices = self.model.kneighbors(self.pivot_table.values, n_neighbors=min(self.n_neighbors + 1, len(self.pivot_table)))
            
            # Liczymy podobieństwa
            if self.weights == 'uniform':
                similarities = np.ones_like(distances[:, 1:])
            else:
                similarities = 1.0 - distances[:, 1:]
                
            sum_sim = np.sum(similarities, axis=1, keepdims=True)
            sum_sim = np.where(sum_sim == 0, 1.0, sum_sim)
            
            # Pobieramy oceny użytkownika dla sąsiadów każdej książki
            neighbor_indices = indices[:, 1:]
            user_ratings_for_neighbors = user_col[neighbor_indices]
            
            all_predictions = np.sum(user_ratings_for_neighbors * similarities, axis=1) / sum_sim.flatten()

        # Filtrowanie maseczką nieprzeczytanych książek
        final_scores = all_predictions[unrated_mask]
        final_books = book_ids[unrated_mask]
        
        # Pobieranie tylko tych z oceną większą od zera
        valid_mask = final_scores > 0.0
        final_scores = final_scores[valid_mask]
        final_books = final_books[valid_mask]
        
        if len(final_scores) == 0:
            return []
            
        # Sortowanie za pomocą argpartition i argsort (najszybsza metoda w NumPy)
        top_size = min(top_k, len(final_scores))
        top_indices = np.argpartition(-final_scores, top_size - 1)[:top_size]
        top_order = np.argsort(-final_scores[top_indices])
        final_indices = top_indices[top_order]
        
        return list(zip(final_books[final_indices].astype(int), final_scores[final_indices].astype(float)))