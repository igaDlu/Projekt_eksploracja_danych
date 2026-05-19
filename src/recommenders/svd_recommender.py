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
        self.predictions_df = None  # Tutaj zapiszemy w pełni zrekonstruowaną macierz ocen

    def fit(self, ratings: List[Rating]) -> None:
        # 1. Konwersja na DataFrame
        df = pd.DataFrame([{'user_id': r.user_id, 'isbn': r.isbn, 'rating': r.rating} for r in ratings])

        # NOWOŚĆ (Tylko dla SVD): Odfiltrowujemy zera.
        # SVD będzie się uczyć tylko twardych gustów (1-10).
        df = df[df['rating'] > 0]

        # 2. Budowa Pivot Table (reszta bez zmian)
        if self.kind == "user":
            self.pivot_table = df.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
        # ... (reszta kodu zostaje dokładnie taka sama)
        else:
            # Wiersze: Książki (ISBN), Kolumny: Użytkownicy
            self.pivot_table = df.pivot(index='isbn', columns='user_id', values='rating').fillna(0)

        # 3. Inicjalizacja i dopasowanie modelu TruncatedSVD ze scikit-learn
        # Zabezpieczamy się przed sytuacją, gdy n_components jest większe niż liczba kolumn/wierszy
        max_components = min(self.pivot_table.shape) - 1
        actual_components = min(self.n_components, max_components)

        svd = TruncatedSVD(n_components=actual_components, random_state=42)

        # 4. MATEMATYCZNA REKONSTRUKCJA MACIERZY:
        # fit_transform kompresuje macierz, a inverse_transform odtwarza ją do pełnych wymiarów.
        # Wynikowy obiekt matrix_reconstructed nie ma już zer – zawiera przewidywane oceny.
        matrix_reduced = svd.fit_transform(self.pivot_table.values)
        matrix_reconstructed = svd.inverse_transform(matrix_reduced)

        # 5. Zapisujemy wyniki do nowego DataFrame, zachowując te same indeksy i kolumny
        self.predictions_df = pd.DataFrame(
            matrix_reconstructed,
            index=self.pivot_table.index,
            columns=self.pivot_table.columns
        )

    def predict(self, user_idx: int, item_idx: int) -> float:
        # Sprawdzamy, czy tabela predykcji w ogóle istnieje
        if self.predictions_df is None:
            raise ValueError("Model nie został jeszcze wytrenowany. Wywołaj najpierw metodę fit().")

        # Obsługa przypadków "Zimnego startu" (jeśli ID nie było w zbiorze treningowym)
        if self.kind == "user":
            if user_idx not in self.predictions_df.index or item_idx not in self.predictions_df.columns:
                return 0.0
            # Odczytujemy gotową wartość z zrekonstruowanej macierzy (Wiersz: User, Kolumna: Book)
            return float(self.predictions_df.loc[user_idx, item_idx])
        else:
            if item_idx not in self.predictions_df.index or user_idx not in self.predictions_df.columns:
                return 0.0
            # Odczytujemy gotową wartość (Wiersz: Book, Kolumna: User)
            return float(self.predictions_df.loc[item_idx, user_idx])

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        # Podobnie jak KNN, SVD to algorytm czysto wsadowy (Batch/Offline).
        # Faktoryzacja macierzy wymaga znajomości całego rozkładu danych.
        raise NotImplementedError("SVD wymaga ponownego przeliczenia macierzy (metoda fit) po dodaniu nowych ocen.")

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.pivot_table is None:
            raise ValueError("Model nie został jeszcze wytrenowany.")

        # 1. Szukamy książek, których użytkownik jeszcze NIE ocenił (wartość oryginalna = 0)
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

        # 2. Dla każdej nieprzeczytanej książki pobieramy predykcję za pomocą metody predict()
        predictions = []
        for item_idx in unrated_items:
            pred_score = self.predict(user_idx, item_idx)
            # TruncatedSVD może czasem wygenerować minimalne wartości ujemne przy dekompresji,
            # zaokrąglamy je logicznie do zera.
            pred_score = max(0.0, pred_score)
            predictions.append((item_idx, pred_score))

        # 3. Sortujemy ranking malejąco według przewidywanej oceny i zwracamy Top-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]