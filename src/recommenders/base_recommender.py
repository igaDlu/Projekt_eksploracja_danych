from ..models import *
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseRecommender(ABC):
    """Abstrakcyjna klasa bazowa (Interfejs) dla wszystkich modeli rekomendacyjnych."""

    @abstractmethod
    def fit(self, ratings: List[Rating]) -> None:
        """Trenuje model na dostarczonej liście interakcji."""
        pass

    @abstractmethod
    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        """Aktualizuje wiedzę modelu o nową ocenę."""
        pass

    @abstractmethod
    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Zwraca listę krotek (item_idx, przewidywana_ocena)."""
        pass