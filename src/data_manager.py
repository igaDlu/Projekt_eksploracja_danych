import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from .models import User, Book, Rating


class DataManager:
    def __init__(self):
        self.users: List[User] = []
        self.books: List[Book] = []
        self.ratings: List[Rating] = []

        self._user_to_idx: Dict[int, int] = {}
        self._idx_to_user: Dict[int, int] = {}
        self._isbn_to_idx: Dict[str, int] = {}
        self._idx_to_isbn: Dict[int, str] = {}

        # Wewnętrzne DataFrames do przetwarzania
        self.df_users = None
        self.df_books = None
        self.df_ratings = None

    def load_kaggle_dataset(self, users_path: str, books_path: str, ratings_path: str) -> None:
        """Wczytuje surowe pliki CSV używając pandas."""
        print("Wczytywanie plików CSV...")
        # Zbiór Book-Crossings używa średników i kodowania latin-1, escapechar chroni przed błędami parsowania
        self.df_users = pd.read_csv(users_path, sep=',', encoding='latin-1', escapechar='\\', on_bad_lines='skip')
        self.df_books = pd.read_csv(books_path, sep=',', encoding='latin-1', escapechar='\\', low_memory=False, on_bad_lines='skip')
        self.df_ratings = pd.read_csv(ratings_path, sep=',', encoding='latin-1', escapechar='\\', on_bad_lines='skip')

        # Ujednolicenie nazw kolumn dla wygody
        self.df_users.columns = ['user_id', 'location', 'age']
        self.df_books.columns = ['isbn', 'title', 'author', 'year', 'publisher', 'img_s', 'img_m', 'img_l']
        self.df_ratings.columns = ['user_id', 'isbn', 'rating']

    def clean_metadata(self) -> None:
        """Czyści błędy zbioru zgodnie z MVP."""
        print("Czyszczenie metadanych...")
        self.df_users = self.df_users.drop(columns=['age'], errors='ignore')
        self.df_books = self.df_books.drop(columns=['img_s', 'img_m', 'img_l'], errors='ignore')
        self.df_books['year'] = pd.to_numeric(self.df_books['year'], errors='coerce').fillna(0).astype(int)
        # NIE RUSZAMY df_ratings! Zera zostają jako zera.

    def extract_locations(self) -> None:
        """Rozbija Location na City, Region, Country."""
        print("Rozbijanie lokalizacji...")
        # Podział stringa "miasto, stan, kraj" na 3 kolumny
        location_split = self.df_users['location'].str.split(',', n=2, expand=True)
        self.df_users['city'] = location_split[0].str.strip()
        self.df_users['region'] = location_split[1].str.strip() if 1 in location_split else 'unknown'
        self.df_users['country'] = location_split[2].str.strip() if 2 in location_split else 'unknown'

        self.df_users.drop(columns=['location'], inplace=True)

    def filter_sparse_data(self, min_user_ratings: int = 20, min_book_ratings: int = 20) -> None:
        """Usuwa ogony (użytkowników i książki z małą liczbą ocen)."""
        print(f"Filtrowanie danych (min_user={min_user_ratings}, min_book={min_book_ratings})...")

        # Filtrujemy książki
        book_counts = self.df_ratings['isbn'].value_counts()
        popular_books = book_counts[book_counts >= min_book_ratings].index
        self.df_ratings = self.df_ratings[self.df_ratings['isbn'].isin(popular_books)]

        # Filtrujemy użytkowników
        user_counts = self.df_ratings['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        self.df_ratings = self.df_ratings[self.df_ratings['user_id'].isin(active_users)]

        # Synchronizujemy tabele users i books z odfiltrowanymi ocenami
        self.df_users = self.df_users[self.df_users['user_id'].isin(self.df_ratings['user_id'])]
        self.df_books = self.df_books[self.df_books['isbn'].isin(self.df_ratings['isbn'])]

    def encode_ids(self) -> None:
        """Mapuje oryginalne ID na ciągłe indeksy 0-N dla modeli macierzowych/grafowych."""
        print("Mapowanie ID...")
        # Mapowanie User ID
        unique_users = self.df_ratings['user_id'].unique()
        self._user_to_idx = {orig: idx for idx, orig in enumerate(unique_users)}
        self._idx_to_user = {idx: orig for orig, idx in self._user_to_idx.items()}

        # Mapowanie ISBN
        unique_books = self.df_ratings['isbn'].unique()
        self._isbn_to_idx = {orig: idx for idx, orig in enumerate(unique_books)}
        self._idx_to_isbn = {idx: orig for orig, idx in self._isbn_to_idx.items()}

        # Podmiana w DataFrame (bardzo szybka dzięki map)
        self.df_ratings['user_idx'] = self.df_ratings['user_id'].map(self._user_to_idx)
        self.df_ratings['item_idx'] = self.df_ratings['isbn'].map(self._isbn_to_idx)

    def populate_entities(self) -> None:
        """Przerzuca dane z Pandas do naszych obiektów systemowych (User, Book, Rating)."""
        print("Populowanie encji systemowych...")

        # Dla Rating używamy zmapowanych indeksów wewnętrznych (user_idx, item_idx)
        self.ratings = [
            Rating(user_id=row.user_idx, isbn=row.item_idx, rating=row.rating)
            for row in self.df_ratings.itertuples(index=False)
        ]

        # Zapis Userów (zachowujemy oryginalne ID dla frontendu)
        self.users = [
            User(user_id=row.user_id, city=row.city, region=row.region, country=row.country)
            for row in self.df_users.itertuples(index=False)
        ]

        # Zapis Książek (zachowujemy oryginalne ISBN)
        self.books = [
            Book(isbn=row.isbn, title=row.title, author=row.author, publisher=row.publisher, year_of_publication=row.year)
            for row in self.df_books.itertuples(index=False)
        ]

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Rating], List[Rating]]:
        """Dzieli listę obiektów Rating na zbiór treningowy i testowy."""
        print("Dzielenie zbioru na Train/Test...")
        train_data, test_data = train_test_split(self.ratings, test_size=test_size, random_state=random_state)
        return train_data, test_data