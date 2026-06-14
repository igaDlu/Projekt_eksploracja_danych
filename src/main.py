# main.py
import argparse
import csv
import os
from src.data_manager import DataManager
from src.recommenders.svd_recommender import SVDRecommender
from src.recommenders.knn_recommender import KNNRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.node2vec_recommender import Node2VecRecommender
from src.evaluator import Evaluator


def parse_arguments():
    def float_0_1(value):
        float_value = float(value)
        if float_value < 0.0 or float_value > 1.0:
            raise argparse.ArgumentTypeError("Wartość musi być w zakresie 0.0-1.0")
        return float_value

    parser = argparse.ArgumentParser(
        description="Porównanie modeli systemu rekomendacji książek przy użyciu SVD, KNN, Node2Vec i hybrydy."
    )
    parser.add_argument(
        "--cutoff", type=int, choices=[10, 50], default=50,
        help="Wartość odcięcia Top-K dla metryk (HIT@K, MRR@K, NDCG@K)."
    )
    parser.add_argument(
        "--dimension", type=int, choices=[16, 100, 300], default=16,
        help="Wymiarowość wektorów dla SVD i Node2Vec (Najlepsze wyniki dla 16!)."
    )
    parser.add_argument(
        "--walk_length", type=int, default=20,
        help="Długość spaceru losowego dla Node2Vec."
    )
    parser.add_argument(
        "--knn_mode", choices=["item", "user"], default="user",
        help="Tryb działania KNN: item-based lub user-based (Najlepsze wyniki dla user!)."
    )
    parser.add_argument(
        "--knn_neighbors", type=int, choices=[5, 20, 50], default=50,
        help="Liczba sąsiadów k dla KNN (Najlepsze wyniki dla 50!)."
    )
    parser.add_argument(
        "--knn_weights", choices=["uniform", "distance"], default="distance",
        help="Sposób ważenia głosów sąsiadów w KNN."
    )
    parser.add_argument(
        "--svd_mode", choices=["item", "user"], default="item",
        help="Tryb działania SVD: item-based lub user-based."
    )
    parser.add_argument(
        "--node2vec_mode", choices=["item", "user"], default="user",
        help="Tryb działania Node2Vec: item-based lub user-based."
    )
    parser.add_argument(
        "--model_to_run", choices=["all", "svd", "knn", "node2vec", "hybrid"], default="hybrid",
        help="Wybór modelu do uruchomienia."
    )
    parser.add_argument(
        "--hybrid_weight_a", type=float_0_1, default=0.5,
        help="Waga pierwszego wybranego modelu w hybrydzie."
    )
    parser.add_argument(
        "--hybrid_models", nargs=2, choices=["svd", "knn", "node2vec"], default=["svd", "knn"],
        help="Lista dwóch modeli dla HybridRecommender, np. svd knn."
    )

    args = parser.parse_args()

    if len(args.hybrid_models) != 2:
        parser.error("--hybrid_models musi zawierać dokładnie dwa modele.")
    if args.hybrid_models[0] == args.hybrid_models[1]:
        parser.error("--hybrid_models musi zawierać dwa różne modele.")

    return args


def main():
    args = parse_arguments()

    dm = DataManager()
    dm.load_kaggle_dataset('./data/Users.csv', './data/Books.csv', './data/Ratings.csv')
    dm.clean_metadata()
    dm.extract_locations()
    dm.filter_sparse_data(min_user_ratings=50, min_book_ratings=50)
    dm.encode_ids()
    dm.populate_entities()

    train_ratings, test_ratings = dm.get_train_test_split(test_size=0.2, random_state=42)
    print(f"Dane treningowe: {len(train_ratings)}, testowe: {len(test_ratings)}")

    evaluator = Evaluator()

    # Inicjalizacja modeli bazowych z poprawnymi parametrami przekazanymi z konsoli
    svd_model = SVDRecommender(n_components=args.dimension, kind=args.svd_mode)
    knn_model = KNNRecommender(n_neighbors=args.knn_neighbors, weights=args.knn_weights, kind=args.knn_mode)
    node2vec_model = Node2VecRecommender(
        dimensions=args.dimension, walk_length=args.walk_length, num_walks=50, window_size=5, kind=args.node2vec_mode
    )

    model_map = {
        "svd": svd_model,
        "knn": knn_model,
        "node2vec": node2vec_model,
    }

    # Budujemy hybrydę na podstawie modeli ze słownika
    hybrid_models = [name.lower() for name in args.hybrid_models]
    hybrid_a = model_map[hybrid_models[0]]
    hybrid_b = model_map[hybrid_models[1]]
    
    hybrid_model = HybridRecommender(model_a=hybrid_a, model_b=hybrid_b, weight_a=args.hybrid_weight_a)

    # Decyzja, co uruchamiamy
    if args.model_to_run == "all":
        models_to_test = [svd_model, knn_model, node2vec_model, hybrid_model]
    elif args.model_to_run == "svd":
        models_to_test = [svd_model]
    elif args.model_to_run == "knn":
        models_to_test = [knn_model]
    elif args.model_to_run == "node2vec":
        models_to_test = [node2vec_model]
    else:  # hybrid
        models_to_test = [hybrid_model]

    all_results = {}

    for model in models_to_test:
        # Dynamiczne budowanie nazwy dla prezentacji w pliku CSV i konsoli
        if isinstance(model, HybridRecommender):
            model_name = f"Hybrid ({model.model_a.__class__.__name__} w={model.weight_a} + {model.model_b.__class__.__name__} w={round(model.weight_b, 2)})"
        else:
            model_name = f"{model.__class__.__name__} ({model.kind}-based)"
            
        print(f"\n" + "=" * 50)
        print(f"Rozpoczynam proces dla: {model_name}")
        print("=" * 50)

        print(f"Trenowanie modelu {model_name}...")
        model.fit(train_ratings)

        print(f"Ewaluacja modelu {model_name}...")
        metrics = evaluator.evaluate(model, test_ratings, top_k=args.cutoff)
        all_results[model_name] = metrics

    # Podsumowanie w konsoli
    print("\n" + "=====" * 3 + " OSTATECZNE PORÓWNANIE MODELI " + "=====" * 3)
    for model_name, metrics in all_results.items():
        print(f"\nModel: {model_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print("=" * 69)

    # Zapis do CSV
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "tuning_wyniki.csv")
    file_exists = os.path.isfile(results_path)

    with open(results_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([
                "Model", "Cutoff", "Dimension", "Walk_Length", "KNN_Mode",
                "KNN_Neighbors", "KNN_Weights", "SVD_Mode", "Node2Vec_Mode", "HIT", "MRR", "NDCG"
            ])

        for model_name, metrics in all_results.items():
            hit_value = next((float(metrics[key]) for key in metrics if "HIT" in key.upper()), 0.0)
            mrr_value = next((float(metrics[key]) for key in metrics if "MRR" in key.upper()), 0.0)
            ndcg_value = next((float(metrics[key]) for key in metrics if "NDCG" in key.upper()), 0.0)

            writer.writerow([
                model_name, args.cutoff, args.dimension, args.walk_length, args.knn_mode,
                args.knn_neighbors, args.knn_weights, args.svd_mode, args.node2vec_mode,
                round(hit_value, 4), round(mrr_value, 4), round(ndcg_value, 4)
            ])


if __name__ == "__main__":
    main()