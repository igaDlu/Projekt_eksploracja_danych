import networkx as nx
import random
import numpy as np
from typing import List, Tuple
from gensim.models import Word2Vec
from .base_recommender import BaseRecommender
from ..models import Rating


class Node2VecRecommender(BaseRecommender):
    def __init__(self, dimensions: int = 32, walk_length: int = 10, num_walks: int = 10, window_size: int = 5,
                 kind: str = "user") -> None:
        """
        dimensions: Rozmiar wektora embeddingu (np. 32 cechy ukryte).
        walk_length: Długość jednego błądzenia losowego (ile kroków robi wędrowiec).
        num_walks: Ile błądzeń losowych zaczynamy z KAŻDEGO węzła w grafie.
        window_size: Rozmiar okna kontekstu dla Word2Vec.
        """
        super().__init__(kind=kind)
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size

        self.graph = None
        self.word2vec_model = None

    def _generate_random_walks(self) -> List[List[str]]:
        """Wewnętrzna metoda generująca ścieżki błądzenia losowego po grafie."""
        walks = []
        nodes = list(self.graph.nodes())

        # Reprezentujemy węzły jako stringi, bo Word2Vec z biblioteki gensim oczekuje tokenów tekstowych
        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)  # Tasowanie dla zachowania losowości
            for node in nodes:
                walk = [str(node)]
                curr_node = node

                # Robimy wędrowanie po krawędziach uwzględniając ich wagi (oceny)
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.graph.neighbors(curr_node))
                    if len(neighbors) == 0:
                        break

                    # Pobieramy wagi krawędzi do sąsiadów, by wędrowiec chętniej szedł w stronę wyższych ocen
                    weights = [self.graph[curr_node][nbr].get('weight', 1.0) for nbr in neighbors]

                    # Normalizacja wag, by tworzyły rozkład prawdopodobieństwa
                    sum_weights = sum(weights)
                    if sum_weights > 0:
                        probabilities = [w / sum_weights for w in weights]
                    else:
                        probabilities = None

                    # Losowy krok z uwzględnieniem prawdopodobieństwa wagowego
                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    walk.append(str(next_node))
                    curr_node = next_node

                walks.append(walk)
        return walks

    def fit(self, ratings: List[Rating]) -> None:
        print("Budowanie grafu dwudzielnego (Użytkownicy <-> Książki)...")
        self.graph = nx.Graph()

        for r in ratings:
            u_node = f"u_{r.user_id}"
            b_node = f"b_{r.isbn}"

            edge_weight = float(r.rating) if r.rating > 0 else 1.0

            self.graph.add_edge(u_node, b_node, weight=edge_weight)

        print("Generowanie ścieżek błądzenia losowego (Random Walks)...")
        walks = self._generate_random_walks()

        print("Trenowanie modelu Word2Vec na wygenerowanych ścieżkach...")
        # sg=1 oznacza użycie algorytmu Skip-Gram (serce Node2Vec)
        self.word2vec_model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=4,
            epochs=5
        )

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.word2vec_model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj najpierw fit().")

        u_node = f"u_{user_idx}"
        b_node = f"b_{item_idx}"

        # Jeśli któregoś węzła nie ma w słowniku embeddingów (zimny start), zwracamy 0.0
        if u_node not in self.word2vec_model.wv or b_node not in self.word2vec_model.wv:
            return 0.0

        # Wyliczamy podobieństwo cosinusowe między wektorem użytkownika a książki.
        # Wynik jest w przedziale [-1, 1], skalujemy go liniowo do rozdzielczości ocen [0, 1] dla kompatybilności z SVD
        similarity = self.word2vec_model.wv.similarity(u_node, b_node)
        scaled_score = (similarity + 1.0) / 2.0
        return float(scaled_score)

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        if self.graph is not None:
            self.graph.add_edge(f"u_{user_idx}", f"b_{item_idx}", weight=float(score))

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.word2vec_model is None:
            return []

        u_node = f"u_{user_idx}"
        if u_node not in self.word2vec_model.wv:
            return []

        # Wyciągamy z grafu listę książek, które użytkownik już ocenił
        rated_items = set()
        if self.graph.has_node(u_node):
            for nbr in self.graph.neighbors(u_node):
                # Odszyfrowujemy ID książki usuwając prefiks 'b_'
                rated_items.add(int(nbr.replace('b_', '')))

        # Szukamy rekomendacji wśród wszystkich unikalnych książek w modelu Word2Vec
        predictions = []
        for word in self.word2vec_model.wv.index_to_key:
            if word.startswith('b_'):
                item_id = int(word.replace('b_', ''))
                # Pomijamy pozycje już przeczytane
                if item_id in rated_items:
                    continue

                pred_score = self.predict(user_idx, item_id)
                predictions.append((item_id, pred_score))

        # Sortujemy ranking malejąco i zwracamy Top-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]