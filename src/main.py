# main.py
from src.data_manager import DataManager
from src.recommenders.svd_recommender import SVDRecommender
from src.recommenders.knn_recommender import KNNRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.node2vec_recommender import Node2VecRecommender
from src.evaluator import Evaluator


def main():
    # 1. Inicjalizacja i wczytanie danych
    dm = DataManager()
    # Ścieżki dopasowane do struktury projektu (zakładamy uruchamianie z głównego folderu)
    dm.load_kaggle_dataset('../data/Users.csv', '../data/Books.csv', '../data/Ratings.csv')

    # 2. Przetwarzanie i przygotowanie potoku danych (Zgodnie z MVP)
    dm.clean_metadata()
    dm.extract_locations()

    # Wyższe wartości progowe gwarantują brak szumu i szybkie testowanie modeli
    dm.filter_sparse_data(min_user_ratings=50, min_book_ratings=50)
    dm.encode_ids()
    dm.populate_entities()

    # 3. Podział danych na zbiór treningowy i testowy
    train_ratings, test_ratings = dm.get_train_test_split(test_size=0.2, random_state=42)
    print(f"Dane treningowe: {len(train_ratings)}, testowe: {len(test_ratings)}")

    # 4. Inicjalizacja Ewaluatora metryk (HIT@10, MRR@10, NDCG@10)
    evaluator = Evaluator()

    # 1. Definiujemy silne instancje naszych składowych
    svd_model = SVDRecommender(n_components=12, kind="item")

    # Używamy parametrów wymiary 8, spacery 50, item-based
    n2v_model = Node2VecRecommender(dimensions=16, walk_length=30, num_walks=50, window_size=5, kind="item")

    # 2. Tworzymy Hybrydę (np. 60% głosu ma SVD, 40% Node2Vec)
    hybrid_model = HybridRecommender(model_a=svd_model, model_b=n2v_model, weight_a=0.6)

    # Lista do finałowego testu
    models_to_test = [
        svd_model,
        n2v_model,
        hybrid_model
    ]

    # Słownik do przechowywania końcowych wyników dla podsumowania
    all_results = {}

    # 5. Pętla uruchamiająca benchmark dla każdego modelu
    for model in models_to_test:
        model_name = f"{model.__class__.__name__} ({model.kind}-based)"
        print(f"\n" + "=" * 50)
        print(f"Rozpoczynam proces dla: {model_name}")
        print("=" * 50)

        # Trening modelu
        print(f"Trenowanie modelu {model_name}...")
        model.fit(train_ratings)

        # Ewaluacja na zbiorze testowym
        metrics = evaluator.evaluate(model, test_ratings, top_k=10)
        all_results[model_name] = metrics

    # 6. Czyste podsumowanie wyników w konsoli
    print("\n" + "=====" * 3 + " OSTATECZNE PORÓWNANIE MODELI BASELINE " + "=====" * 3)
    for model_name, metrics in all_results.items():
        print(f"\nModel: {model_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print("=" * 69)


if __name__ == "__main__":
    main()