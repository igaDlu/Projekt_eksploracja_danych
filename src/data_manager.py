from typing import List, Dict, Tuple
from models import User, Book, Rating

class DataManager:
    def __init__(self):
        self.users: List[User] = []
        self.books: List[Book] = []
        self.ratings: List[Rating] = []

        self._user_to_idx: Dict[int, int] = {}
        self._idx_to_user: Dict[int, int] = {}
        self._isbn_to_idx: Dict[str, int] = {}
        self._idx_to_isbn: Dict[int, str] = {}

    def load_kaggle_dataset(self, users_path: str, books_path: str, ratings_path: str) -> None:
        """Wczytuje pliki CSV"""
        pass

    def clean_metadata(self) -> None:
        """Czyści błędy zbioru"""
        pass

    def filter_sparse_data(self, min_user_ratings: int = 20, min_book_ratings: int = 20) -> None:
        """Usuwa użytkowników i książki z liczbą ocen poniżej progu."""
        pass

    def encode_ids(self) -> None:
        """Wypełnia słowniki mapujące i podmienia ID w interakcjach na indeksy wewnętrzne."""
        pass

    def get_train_test_split(self, test_size: float = 0.2) -> Tuple[List[Rating], List[Rating]]:
        """Dzieli zbiór na treningowy i testowy."""
        pass