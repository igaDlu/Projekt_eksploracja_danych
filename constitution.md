# Constitution of the Book Recommender Comparison Project

## 1. Cel projektu
Projekt porównuje metody rekomendacji książek wykorzystując dane użytkowników, książek i ocen. Jego celem jest sprawdzenie, które podejście daje najlepsze wyniki w ocenie jakości rekomendacji na jednym zbiorze danych.

## 2. Dane
Dane znajdują się w katalogu `data/` i są reprezentowane przez trzy pliki CSV:
- `Users.csv` — informacje o użytkownikach, w tym `user_id`, `location` i `age`.
- `Books.csv` — metadane książek, takie jak `isbn`, `title`, `author`, `year`, `publisher` oraz linki do obrazków.
- `Ratings.csv` — oceny książek przez użytkowników w formie `user_id`, `isbn`, `rating`.

Wczytywanie i czyszczenie danych odbywa się w `src/data_manager.py` metodą `DataManager.load_kaggle_dataset()`.

## 3. Przetwarzanie danych
Przetwarzanie danych w `src/data_manager.py` obejmuje:
- wczytanie CSV do obiektów pandas,
- normalizację nazw kolumn i usuwanie zbędnych kolumn z obrazkami,
- konwersję roku publikacji do typu całkowitego,
- rozdzielenie pola `location` na `city`, `region`, `country`,
- filtrowanie rzadkich użytkowników i książek (domyślnie min. 50 ocen),
- mapowanie oryginalnych identyfikatorów `user_id` i `isbn` na indeksy wewnętrzne,
- stworzenie list obiektów `Rating`, `User`, `Book`.

## 4. Reprezentacja encji
W `src/models.py` są zdefiniowane trzy dataclassy:
- `User(user_id, city, region, country)`
- `Book(isbn, title, author, year_of_publication, publisher)`
- `Rating(user_id, isbn, rating)`

## 5. Recommender Base
W `src/recommenders/base_recommender.py` znajduje się abstrakcyjna klasa `BaseRecommender`, definiująca interfejs:
- `fit(ratings)`
- `predict(user_idx, item_idx)`
- `create_ranking(user_idx, top_k)`
- `rate(user_idx, item_idx, score)`

Każdy model może działać w trybie `user` lub `item` zgodnie z parametrem `kind`.

## 6. Modele rekomendacyjne
### 6.1. `SVDRecommender`
- używa `TruncatedSVD` ze scikit-learn,
- w trybie `user` buduje macierz `user x item`,
- w trybie `item` buduje macierz `item x user`,
- rekonstruuje pełną macierz ocen i tworzy ranking dla nieocenionych pozycji.

### 6.2. `KNNRecommender`
- wykorzystuje `NearestNeighbors` z metryką kosinusową,
- w trybie `user` porównuje podobnych użytkowników,
- w trybie `item` porównuje podobne książki,
- predykcja to ważona średnia ocen od sąsiadów.

### 6.3. `Node2VecRecommender`
- buduje dwudzielny graf użytkownik-książka z `networkx`,
- tworzy losowe spacerowe ścieżki (`random walks`),
- trenuje `Word2Vec` na wygenerowanych ścieżkach,
- rekomendacje bazują na podobieństwie wektorów użytkownik-książka.

### 6.4. `HybridRecommender`
- łączy dwie składowe modelowe w jednej hybrydzie,
- agreguje ich wyniki ważoną sumą,
- zachowuje dodatkową normalizację, gdy jedną z metod jest `Node2VecRecommender`.

## 7. Ewaluacja
W `src/evaluator.py` obliczane są metryki:
- `HIT@K` — czy w Top-K rekomendacji znalazło się przynajmniej jedno trafienie,
- `MRR@K` — średnia odwrotności rangi pierwszego trafienia,
- `NDCG@K` — znormalizowane dyskontowane zyski uwzględniające pozycję trafień.

Ewaluacja grupuje oceny testowe po użytkownikach i dla każdego generuje ranking z modelu.

## 8. Logika główna
W `src/main.py` proces wygląda następująco:
1. parsowanie argumentów z linii poleceń,
2. wczytanie i przygotowanie danych,
3. podział listy ocen na `train` / `test` (domyślnie 80/20),
4. trenowanie wybranych modeli,
5. ewaluacja modeli przy `top_k` określonym flagą `--cutoff`,
6. zapis wyników do `results/tuning_wyniki.csv`.

Domyślne parametry w `main.py` to:
- `--cutoff 50`,
- `--dimension 16`,
- `--walk_length 20`,
- `--knn_mode user`,
- `--knn_neighbors 50`,
- `--knn_weights distance`,
- `--svd_mode item`,
- `--node2vec_mode user`,
- `--model_to_run hybrid`.

## 9. Wyniki eksperymentów
Wyniki są zapisywane w `results/tuning_wyniki.csv` i zawierają pola:
- `Model`, `Cutoff`, `Dimension`, `Walk_Length`,
- `KNN_Mode`, `KNN_Neighbors`, `KNN_Weights`,
- `SVD_Mode`, `Node2Vec_Mode`, `HIT`, `MRR`, `NDCG`.

## 10. Uwagi do przeglądu kodu
Podczas przeglądu kodu znaleziono:
- w `src/recommenders/hybrid_recommender.py` w metodzie `predict()` jest pozostawiona linia debugująca `print(12345)` — można ją usunąć, ponieważ nie jest potrzebna.

## 11. Wskazówki do rozwoju
Można rozszerzyć projekt o:
- uruchamianie `KNNRecommender` w `main.py`,
- eksperymenty z różnymi `min_user_ratings` / `min_book_ratings`,
- dodatkowe metryki lub hybrydowe konfiguracje.
