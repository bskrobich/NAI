Napisz implementację algorytmu K najbliższych sąsiadów (kNN).
Do obliczenia odległości między obserwacjami należy wykorzystać kwadrat odległości euklidesowej.

Po uruchomieniu programu użytkownik podaje ścieżkę do pliku treningowego.
Następnie,  użytkownik (w konsoli) podaje liczbę K, która oznacza liczbę najbliższych sąsiadów.

Program działając w pętli umożliwia wybór jednej z 4 opcji:
a) klasyfikacja wszystkich obserwacji ze zbioru testowego podanego w oddzielnym pliku - program umożliwia podanie ścieżki do pliku ze zbiorem testowym, wyznacza etykietę i wypisuje odpowiedź dla każdej obserwacji a następnie podaje dokładność dla całego zbioru
b) klasyfikacja obserwacji podanej przez użytkownika w konsoli - program wczytuje obserwację podaną w konsoli i wyznacza etykietę dla podanej obserwacji. Zakładamy że liczba cech jest taka sama jak w danych treningowych, nie ma potrzeby podawania poprawnej etykiety
c) zmień K
d) wyjście z programu

Program powinien działać dla dowolnych danych wejściowych podanych w postaci pliku tekstowego gdzie:
- każdy wiersz jest pojedynczą obserwacją
- każdy wiersz zawiera N kolumn oddzielonych znakiem ","
- pierwsze N-1 kolumn to cechy obserwacji (features) a ostatnia kolumna to poprawna etykieta (label)
- cechy obserwacji są numeryczne

Program nie zna z góry liczby kolumn N, określa ją na podstawie wczytanego pliku treningowego.