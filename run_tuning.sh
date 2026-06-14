#!/bin/bash

# Przejdź do katalogu, w którym znajduje się skrypt
cd "$(dirname "$0")"

# echo "========================================"
# echo "TUNING SVD - item mode"
# echo "========================================"
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode item --dimension 16
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode item --dimension 100
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode item --dimension 300

# echo "========================================"
# echo "TUNING SVD - user mode"
# echo "========================================"
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode user --dimension 16
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode user --dimension 100
# python -m src.main --model_to_run svd --cutoff 50 --svd_mode user --dimension 300

# echo "========================================"
# echo "TUNING KNN - item mode"
# echo "========================================"
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode item --knn_neighbors 5 --knn_weights uniform
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode item --knn_neighbors 20 --knn_weights uniform
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode item --knn_neighbors 20 --knn_weights distance
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode item --knn_neighbors 50 --knn_weights distance

# echo "========================================"
# echo "TUNING KNN - user mode"
# echo "========================================"
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode user --knn_neighbors 5 --knn_weights uniform
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode user --knn_neighbors 20 --knn_weights uniform
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode user --knn_neighbors 20 --knn_weights distance
# python -m src.main --model_to_run knn --cutoff 50 --knn_mode user --knn_neighbors 50 --knn_weights distance

# echo "========================================"
# echo "TUNING NODE2VEC - item mode"
# echo "========================================"
# python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode item --dimension 16 --walk_length 20
# python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode item --dimension 100 --walk_length 50
# python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode item --dimension 300 --walk_length 100

echo "========================================"
echo "TUNING NODE2VEC - user mode"
echo "========================================"
python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode user --dimension 16 --walk_length 20
python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode user --dimension 100 --walk_length 50
python -m src.main --model_to_run node2vec --cutoff 50 --node2vec_mode user --dimension 300 --walk_length 100

echo "========================================"
echo "Tuning finished."
echo "Results appended to results/tuning_wyniki.csv"
echo "========================================"