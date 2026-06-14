import networkx as nx
import random
import numpy as np
import multiprocessing
from typing import List, Tuple
from gensim.models import Word2Vec
from .base_recommender import BaseRecommender
from ..models import Rating


class Node2VecRecommender(BaseRecommender):
    def __init__(self, dimensions: int = 32, walk_length: int = 10, num_walks: int = 10, window_size: int = 5,
                 kind: str = "user") -> None:
        super().__init__(kind=kind)
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size

        self.graph = None
        self.word2vec_model = None

    def _generate_random_walks(self) -> List[List[str]]:
        walks = []
        nodes = list(self.graph.nodes())
        
        print(" Przygotowywanie pamięci podręcznej grafu (cache)...")
        graph_cache = {}
        for node in nodes:
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                weights = [float(self.graph[node][nbr].get('weight', 1.0)) for nbr in neighbors]
                sum_weights = sum(weights)
                # Bezpieczne mapowanie wag
                probabilities = [w / sum_weights for w in weights] if sum_weights > 0 else None
                graph_cache[node] = (neighbors, probabilities)

        print(f" Generowanie {self.num_walks * len(nodes)} kroków spaceru...")
        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                curr_node = node

                for _ in range(self.walk_length - 1):
                    cache = graph_cache.get(curr_node)
                    if not cache or not cache[0]:
                        break
                    neighbors, probabilities = cache
                    
                    if probabilities is not None:
                        curr_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    else:
                        curr_node = random.choice(neighbors)
                    walk.append(str(curr_node))

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
        cpus = multiprocessing.cpu_count()
        print(f"  [Użycie procesora: Wykryto {cpus} wątków logicznych dla Word2Vec]")

        self.word2vec_model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=cpus,
            epochs=5
        )

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.word2vec_model is None:
            raise ValueError("Model nie został wytrenowany.")

        u_node = f"u_{user_idx}"
        b_node = f"b_{item_idx}"

        if u_node not in self.word2vec_model.wv or b_node not in self.word2vec_model.wv:
            return 0.0

        similarity = self.word2vec_model.wv.similarity(u_node, b_node)
        return float((similarity + 1.0) / 2.0)

    def rate(self, user_idx: int, item_idx: int, score: float) -> None:
        if self.graph is not None:
            self.graph.add_edge(f"u_{user_idx}", f"b_{item_idx}", weight=float(score))

    def create_ranking(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.word2vec_model is None:
            return []

        u_node = f"u_{user_idx}"
        if u_node not in self.word2vec_model.wv:
            return []

        rated_items = set()
        if self.graph.has_node(u_node):
            for nbr in self.graph.neighbors(u_node):
                rated_items.add(nbr)

        # KLUCZOWA POPRAWKA: Wymuszamy pobranie znormalizowanego wektora użytkownika
        user_vector = self.word2vec_model.wv.get_vector(u_node, norm=True)
        
        all_keys = self.word2vec_model.wv.index_to_key
        book_keys = [k for k in all_keys if k.startswith('b_') and k not in rated_items]
        
        if not book_keys:
            return []

        # KLUCZOWA POPRAWKA: Pobieramy ZNORMALIZOWANE wektory macierzy książek
        # Dzięki temu @ zadziała dokładnie tak jak wv.similarity() w starym kodzie
        book_indices = [self.word2vec_model.wv.key_to_index[k] for k in book_keys]
        book_vectors = self.word2vec_model.wv.get_normed_vectors()[book_indices]

        # Teraz to jest prawdziwe, czyste podobieństwo cosinusowe w NumPy
        similarities = book_vectors @ user_vector
        scaled_scores = (similarities + 1.0) / 2.0

        top_size = min(top_k, len(scaled_scores))
        top_indices = np.argpartition(-scaled_scores, top_size - 1)[:top_size]
        top_order = np.argsort(-scaled_scores[top_indices])
        final_indices = top_indices[top_order]

        ranking = []
        for idx in final_indices:
            item_id = int(book_keys[idx].replace('b_', ''))
            ranking.append((item_id, float(scaled_scores[idx])))

        return ranking