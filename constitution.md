# Constitution of the Book Recommender Comparison Project

## 1. Cel projektu
Projekt ma porównać różne metody rekomendacji książek na podstawie danych użytkowników, książek i ocen. Jego głównym celem jest ocenienie jakości różnych algorytmów rekomendacyjnych i porównanie ich wyników na tych samych danych.

## 2. Dane
Dane są w katalogu `data/` i obejmują trzy pliki CSV:
- `Users.csv` — informacje o użytkownikach, w tym oryginalne ID, lokalizację i wiek.
- `Books.csv` — metadane książek, takie jak ISBN, tytuł, autor, wydawca i rok publikacji.
- `Ratings.csv` — oceny użytkowników dla książek.

Dane są ładowane w `src/data_manager.py` metodą `DataManager.load_kaggle_dataset()` i przetwarzane przed trenowaniem modeli.

## 3. Przetwarzanie danych
Pipeline przetwarzania w `src/data_manager.py` obejmuje:
- wczytanie CSV do DataFrame z Pandas,
- czyszczenie metadanych książek (usuwanie kolumn z obrazkami, konwersja roku publikacji do liczby całkowitej),
- ekstrakcję lokalizacji użytkownika do pól `city`, `region`, `country`,
- filtrowanie rzadkich użytkowników i książek, by usunąć ogon danych,
- mapowanie oryginalnych identyfikatorów `user_id` i `isbn` na indeksy wewnętrzne 0..N,
- tworzenie obiektów `Rating`, `User`, `Book` do dalszego użycia w modelach.

## 4. Modele danych
W `src/models.py` znajdują się proste klasy danych:
- `User` — `user_id`, `city`, `region`, `country`
- `Book` — `isbn`, `title`, `author`, `year_of_publication`, `publisher`
- `Rating` — `user_id`, `isbn`, `rating`

## 5. Ewaluacja
W `src/evaluator.py` metodyka oceny obejmuje:
- `HIT@K` — czy przynajmniej jedno trafienie było w Top-K rekomendacji,
- `MRR@K` — średnia odwrotności pozycji pierwszego trafienia,
- `NDCG@K` — uwzględnia jakość i pozycję trafień w Top-K.

Ewaluacja odbywa się dla każdego użytkownika ze zbioru testowego, a wyniki są uśredniane globalnie.

## 6. Implementowane metody rekomendacji
W katalogu `src/recommenders/` znajdują się następujące klasy:

### 6.1. `SVDRecommender`
- implementacja oparta o `TruncatedSVD` ze scikit-learn,
- może działać w trybie `user` lub `item`,
- buduje macierz użytkownik-książka lub książka-użytkownik,
- dekomponuje ją do niższej wymiarowości i rekonstruuje oceny,
- generuje ranking książek na podstawie przewidywanych ocen dla nieocenionych pozycji.

### 6.2. `KNNRecommender`
- oparcie na algorytmie k-NN (`NearestNeighbors` z metryką kosinusową),
- tryb `user` — porównanie podobnych użytkowników,
- tryb `item` — porównanie podobnych książek,
- predykcja wykorzystuje ważoną średnią ocen sąsiadów.

### 6.3. `Node2VecRecommender`
- buduje dwudzielny graf użytkownicy-książki przy użyciu `networkx`,
- generuje losowe przechadzki (`random walks`) po grafie,
- trenuje model `Word2Vec` na tych ścieżkach,
- każdemu użytkownikowi i książce przypisuje embedding,
- rekomendacje powstają na bazie kosinusowego podobieństwa wektorów użytkownik-książka.

### 6.4. `HybridRecommender`
- łączy dwa modele podstawowe w jednej hybrydzie,
- wykorzystuje ważone średnie predykcji z dwóch komponentów,
- w `src/main.py` używa `SVDRecommender` oraz `Node2VecRecommender` z wagą `0.6` dla SVD i `0.4` dla Node2Vec,
- pozwala łatwo dodać inne modele do mieszanki.

## 7. Główna logika wykonywania projektu
W `src/main.py` wykonuje się następujący scenariusz:
1. Wczytanie danych z katalogu `data/`,
2. oczyszczenie i przygotowanie zbiorów,
3. filtrowanie rzadkich użytkowników/książek,
4. zamiana oryginalnych identyfikatorów na indeksy,
5. podział na zbiór treningowy i testowy (20% testowych),
6. trenowanie modeli i generowanie rekomendacji,
7. ewaluacja każdego modelu wg `HIT@10`, `MRR@10`, `NDCG@10`,
8. wypisanie porównania wyników dla każdego modelu.

## 8. Aktualny zestaw eksperymentów
W `src/main.py` są obecnie porównywane:
- `SVDRecommender` w trybie `item`,
- `Node2VecRecommender` w trybie `item`,
- `HybridRecommender` łączący `SVDRecommender` i `Node2VecRecommender`.

Dostępne, ale nieużywane w głównym pliku, jest również `KNNRecommender`.

## 9. Jak używać tego pliku
Ten dokument stanowi prosty kontekst dla dalszych zadań. Jeśli chcesz dodać nowe eksperymenty lub rozwijać projekt, możesz:
- zaproponować nowy model rekomendacyjny,
- poprosić o porównanie wyników dla różnych konfiguracji,
- dodać metryki ewaluacyjne,
- zmienić sposób przygotowania danych,
- włączyć `KNNRecommender` do benchmarku.
