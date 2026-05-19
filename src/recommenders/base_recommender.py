from ..models import *
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseRecommender(ABC):
    """Abstrakcyjna klasa bazowa (Interfejs) dla wszystkich modeli rekomendacyjnych."""

    def __init__(self, kind: str = "user") -> None:
        """
        kind: "user" dla podejścia User-Based, "item" dla Item-Based.
        """
        if kind not in ["user", "item"]:
            raise ValueError("Parametr kind musi wynosić 'user' lub 'item'")
        self.kind = kind

    @abstractmethod
    def fit(self, ratings: List['Rating']) -> None:
        """Trenuje model na dostarczonej liście interakcji."""
        pass

    @abstractmethod
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Przewiduje konkretną ocenę (float) dla danej pary użytkownik-książka."""
        pass

    @abstractmethod
    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Zwraca listę krotek (item_idx, przewidywana_ocena) posortowaną malejąco."""
        pass

    @abstractmethod
    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        """Aktualizuje wiedzę modelu o nową ocenę."""
        pass