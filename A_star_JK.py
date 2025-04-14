from typing import Dict, List, Tuple, Any, Set, Optional
from heapq import heappush, heappop
import networkx as nx
import matplotlib.pyplot as plt
import folium
import webbrowser
from math import radians, sin, cos, sqrt, atan2

def haversine(coord1, coord2):
    """
    Oblicza odległość między dwoma punktami na powierzchni Ziemi używając wzoru haversine.
    
    Wzór haversine pozwala na obliczenie odległości po łuku wielkiego koła między dwoma punktami
    na powierzchni sfery (Ziemi) na podstawie ich współrzędnych geograficznych.
    Jest to dokładniejsza metoda niż proste obliczenia euklidesowe, ponieważ uwzględnia kulistość Ziemi.
    
    Parametry:
    ----------
    coord1 : tuple lub list
        Para współrzędnych (szerokość, długość) pierwszego punktu w stopniach.
    coord2 : tuple lub list
        Para współrzędnych (szerokość, długość) drugiego punktu w stopniach.
    
    Zwraca:
    -------
    float
        Odległość między punktami w kilometrach.
    """
    # Promień Ziemi w kilometrach - średni promień używany w obliczeniach geograficznych
    R = 6371
    
    # Konwersja współrzędnych geograficznych ze stopni na radiany
    # Radiany są wymagane przez funkcje trygonometryczne jak sin, cos
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Obliczenie różnic między współrzędnymi
    # To są delty (zmiany) szerokości i długości geograficznej
    dlat = lat2 - lat1  # Różnica szerokości geograficznych
    dlon = lon2 - lon1  # Różnica długości geograficznych

    # Obliczenie wzoru haversine:
    # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
    # Jest to kwadrat połowy długości cięciwy łuku między punktami
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    # Obliczenie kąta centralnego między punktami w radianach
    # c = 2 * atan2(√a, √(1-a))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Ostateczne obliczenie odległości: długość łuku = promień * kąt centralny
    # Zwracamy odległość w kilometrach
    return R * c 

def format_time(t: float) -> str:
    """
    Formatuje czas podany w minutach do formatu "X godz. Y min."
    
    Funkcja przyjmuje czas w minutach i konwertuje go na bardziej czytelny format
    zawierający godziny i minuty. Jest używana do prezentacji czasów podróży.
    
    Parametry:
    ----------
    t : float
        Czas w minutach do sformatowania.
    
    Zwraca:
    -------
    str
        Sformatowany ciąg znaków w formacie "X godz. Y min."
    """
    # Obliczanie liczby pełnych godzin przez dzielenie całkowite
    godziny = int(t // 60)
    
    # Obliczanie pozostałych minut przy pomocy reszty z dzielenia
    minuty = int(t % 60)
    
    # Zwracanie sformatowanego napisu
    return f"{godziny} godz. {minuty} min."

def wypisz_szczegoly_trasy(graph: Dict[str, Dict[str, dict]], path: List[str]):
    """
    Wypisuje szczegółowe informacje o trasie złożonej z listy odcinków.
    
    Dla każdego odcinka trasy (między kolejnymi punktami) wyświetla:
    - punkt początkowy i końcowy
    - odległość w kilometrach
    - średnią prędkość na odcinku
    - typ drogi (jeśli dostępny)
    - czas przejazdu
    
    Parametry:
    ----------
    graph : Dict[str, Dict[str, dict]]
        Słownik reprezentujący graf, gdzie kluczami są nazwy miejsc,
        a wartościami słowniki sąsiedztwa z atrybutami połączeń.
    path : List[str]
        Lista nazw miejsc tworzących trasę (w kolejności przejazdu).
    
    Zwraca:
    -------
    None
        Funkcja nic nie zwraca, tylko wypisuje informacje na standardowe wyjście.
    """
    # Wyświetlenie nagłówka sekcji ze szczegółami
    print("\nSzczegóły trasy:")
    
    # Iteracja po kolejnych parach miejsc w ścieżce
    for i in range(len(path) - 1):
        # Pobranie nazwy punktu początkowego i końcowego dla bieżącego odcinka
        start = path[i]      # Punkt początkowy odcinka
        end = path[i + 1]    # Punkt końcowy odcinka
        
        # Pobranie danych o połączeniu z grafu
        dane = graph[start][end]  # Słownik z danymi o połączeniu
        
        # Wyodrębnienie poszczególnych informacji o połączeniu
        distance = dane['distance']  # Odległość w km
        speed = dane['speed']        # Prędkość w km/h
        typ = dane.get('type', 'brak')  # Typ drogi, domyślnie 'brak' jeśli nie określono
        
        # Obliczenie czasu przejazdu w minutach: t = s/v * 60
        time = (distance / speed) * 60  # czas w minutach
        
        # Rozdzielenie czasu na godziny i minuty
        godziny = int(time // 60)  # Liczba pełnych godzin
        minuty = int(time % 60)    # Pozostałe minuty
        
        # Wypisanie informacji o odcinku trasy
        print(f"{start} → {end} | {distance} km | {speed} km/h | {typ} | {godziny} godz. {minuty} min.")

def a_star(graph, heuristics, s, k):
    """
    Implementacja algorytmu A* (A gwiazda) do znajdowania najkrótszej ścieżki w grafie.
    
    A* jest algorytmem przeszukiwania grafów, który znajduje najkrótszą ścieżkę z wierzchołka
    startowego do docelowego. Wykorzystuje funkcję heurystyczną do optymalizacji przeszukiwania.
    
    W tej implementacji funkcja minimalizująca to f(n) = g(n) + h(n), gdzie:
    - g(n) to rzeczywisty koszt dojścia od wierzchołka startowego do n
    - h(n) to wartość heurystyczna - szacowany koszt dojścia od n do celu
    
    Parametry:
    ----------
    graph : dict
        Słownik reprezentujący graf w postaci listy sąsiedztwa. Może zawierać:
        - proste krawędzie: {wierzchołek: {sąsiad: koszt, ...}, ...}
        - złożone krawędzie: {wierzchołek: {sąsiad: {'distance': x, 'speed': y}, ...}, ...}
    heuristics : dict
        Słownik heurystyk {wierzchołek: wartość heurystyki, ...}.
        Wartość heurystyki dla wierzchołka to szacowany koszt dotarcia do celu.
    s : str lub int
        Wierzchołek startowy.
    k : str lub int
        Wierzchołek docelowy (końcowy).
    
    Zwraca:
    -------
    tuple lub str
        Jeśli ścieżka istnieje: krotka (lista wierzchołków ścieżki, całkowity czas w minutach).
        Jeśli ścieżka nie istnieje: ciąg znaków "failure".
    """
    # Zbiór wierzchołków przejrzanych (zamkniętych) - zawiera wierzchołki już przetworzone
    Tz = set()
    
    # Zbiór wierzchołków otwartych - zawiera wierzchołki oczekujące na przetworzenie
    # oraz kolejka priorytetowa do wybierania wierzchołka o najniższej wartości f
    To = set()                  # Zbiór wierzchołków otwartych
    open_queue = []             # Kolejka priorytetowa dla wierzchołków otwartych
    
    # Inicjalizacja słownika g, który przechowuje rzeczywisty koszt dojścia do wierzchołka
    # Początkowo ustawiamy nieskończoność dla wszystkich wierzchołków
    g = {wierzcholek: float('inf') for wierzcholek in graph}
    g[s] = 0  # Koszt dojścia do wierzchołka startowego wynosi 0
    
    # Inicjalizacja słownika f, który przechowuje wartość funkcji oceniającej f(n) = g(n) + h(n)
    # Początkowo również ustawiamy nieskończoność
    f = {wierzcholek: float('inf') for wierzcholek in graph}
    f[s] = heuristics.get(s, 0)  # Dla wierzchołka startowego f[s] = g[s] + h[s] = 0 + h[s]
    
    # Słownik przechowujący poprzedników na optymalnej ścieżce
    p = {}
    
    # Dodanie wierzchołka startowego do zbioru otwartych i kolejki priorytetowej
    To.add(s)                         # Dodajemy startowy do zbioru otwartych
    heappush(open_queue, (f[s], s))   # Dodajemy do kolejki priorytetowej z wartością f[s]
    
    # Główna pętla algorytmu - kontynuujemy dopóki są wierzchołki w kolejce
    while open_queue:
        # Wybieramy wierzchołek x o najmniejszej wartości f[x] z kolejki priorytetowej
        # Pierwszy element w krotce to wartość f (priorytet), drugi to identyfikator wierzchołka
        _, x = heappop(open_queue)
        
        # Jeśli wierzchołek był już przetworzony (jest w Tz), pomijamy go
        # To może się zdarzyć, gdy dodamy wierzchołek do kolejki wielokrotnie z różnymi priorytetami
        if x in Tz:
            continue
        
        # Jeśli dotarliśmy do celu, zwróć odtworzoną ścieżkę oraz całkowity czas podróży
        if x == k:
            return path(p, k), g[k]
        
        # Przenosimy wierzchołek x ze zbioru otwartych do przejrzanych
        To.remove(x)   # Usuwamy x ze zbioru otwartych
        Tz.add(x)      # Dodajemy x do zbioru przejrzanych
        
        # Przetwarzamy wszystkich sąsiadów wierzchołka x
        neighbors = graph.get(x, {})  # Pobieramy sąsiadów wierzchołka x
        for y, edge_data in neighbors.items():
            # Pomijamy sąsiadów, którzy są już w zbiorze przejrzanych
            if y in Tz:
                continue
            
            # Obliczamy koszt przejścia krawędzi w zależności od typu grafu
            if isinstance(edge_data, dict):  # Sprawdzamy czy dane krawędzi są słownikiem
                # Dla złożonego grafu z atrybutami obliczamy czas podróży
                distance = edge_data['distance']  # Odległość w km
                speed = edge_data['speed']        # Prędkość w km/h
                travel_time = (distance / speed) * 60  # Czas w minutach: t = s/v * 60
            else:  # Dla prostego grafu zakładamy, że wartość to bezpośrednio koszt/czas
                travel_time = edge_data
            
            # Obliczamy nowy koszt dojścia do y przez x: g*(y) = g(x) + koszt(x,y)
            g_star = g[x] + travel_time
            
            # Flaga określająca, czy znaleźliśmy lepszą drogę
            poprawa = False
            
            # Sprawdzamy, czy y jest w zbiorze otwartych
            if y not in To:
                # Jeśli nie, dodajemy go do zbioru otwartych
                To.add(y)
                poprawa = True
            # Jeśli y jest w To i znaleźliśmy krótszą drogę
            elif g_star < g[y]:
                poprawa = True
                
            # Jeśli znaleźliśmy lepszą drogę do y, aktualizujemy informacje
            if poprawa:
                p[y] = x                         # Aktualizujemy poprzednika y
                g[y] = g_star                    # Aktualizujemy koszt dojścia do y
                f[y] = g[y] + heuristics.get(y, 0)  #! liczymy koszt dotarcia do y(jak blisko jest do celu) Aktualizujemy wartość funkcji f
                
                # Dodajemy y do kolejki priorytetowej z nowym priorytetem f[y]
                heappush(open_queue, (f[y], y))
    
    # Jeśli przeszukaliśmy cały graf i nie znaleźliśmy ścieżki do celu
    # zwracamy informację o niepowodzeniu
    return "failure"

def path(p, u):
    """
    Funkcja odtwarzająca ścieżkę od wierzchołka startowego do docelowego u.
    
    Na podstawie słownika poprzedników funkcja odtwarza ścieżkę, cofając się
    od wierzchołka docelowego do startowego, a następnie odwraca kolejność,
    aby uzyskać ścieżkę w prawidłowej kolejności.
    
    Parametry:
    ----------
    p : dict
        Słownik poprzedników, gdzie p[v] to poprzednik wierzchołka v na optymalnej ścieżce.
        Na przykład, jeśli p[5] = 3, oznacza to, że wierzchołek 3 poprzedza wierzchołek 5 na ścieżce.
    u : str lub int
        Wierzchołek końcowy, dla którego odtwarzamy ścieżkę.
    
    Zwraca:
    -------
    list
        Lista wierzchołków tworzących ścieżkę od wierzchołka startowego do u.
    """
    # Inicjalizacja listy ścieżki z wierzchołkiem końcowym
    sciezka = [u]
    
    # Cofamy się wzdłuż ścieżki, dopóki nie dojdziemy do wierzchołka startowego
    # Warunek u in p sprawdza, czy wierzchołek u ma poprzednika w słowniku p
    while u in p:
        # Przechodzimy do poprzednika wierzchołka u
        u = p[u]
        # Dodajemy poprzednika do ścieżki
        sciezka.append(u)
    
    # Odwracamy ścieżkę, aby uzyskać kolejność od startu do celu
    # Ponieważ tworzyliśmy ścieżkę od końca do początku, musimy ją odwrócić
    sciezka.reverse()
    
    # Zwracamy gotową ścieżkę w prawidłowej kolejności
    return sciezka

def draw_graph(graph: Dict[str, Dict[str, dict]], path: List[str]):
    """
    Rysuje graf i zaznacza najkrótszą ścieżkę.
    
    Parametry:
    - graph: słownik sąsiedztwa reprezentujący graf
    - path: lista wierzchołków na ścieżce do zaznaczenia
    """
    G = nx.DiGraph()

    # Dodaj wierzchołki i krawędzie z wagą czasową (w minutach)
    for u in graph:
        for v, data in graph[u].items():
            if isinstance(data, dict):
                time = (data['distance'] / data['speed']) * 60  # czas w minutach
            else:
                time = data  # Zakładamy, że wartość to bezpośrednio koszt/czas
            G.add_edge(u, v, weight=round(time, 1))

    # Używamy zdefiniowanych pozycji zamiast automatycznego layoutu
    pos = {
        'Opole': (2, 2),
        'Wrocław': (1, 3),
        'Częstochowa': (3, 3),
        'Kraków': (4, 2),
        'Katowice': (3.5, 1.5),
        'Łódź': (3, 4),
        'Warszawa': (5, 5),
        'Kielce': (5, 3),
        'Lublin': (6, 4),
        'Rzeszów': (6, 2)
    }

    # Rysowanie wszystkich krawędzi
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx(G, pos, with_labels=True, node_size=2000, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Podświetlenie ścieżki
    if path:
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=3)

    plt.title("Mapa połączeń między miastami i najkrótsza ścieżka (czasowo)")
    plt.axis('off')
    plt.show()

# --- Dwukierunkowy graf z większą liczbą połączeń ---
graph = {
    'Opole': {
        'Wrocław': {'distance': 90, 'type': 'autostrada', 'speed': 130},
        'Częstochowa': {'distance': 110, 'type': 'lokalna', 'speed': 70}
    },
    'Wrocław': {
        'Opole': {'distance': 90, 'type': 'autostrada', 'speed': 130},
        'Łódź': {'distance': 200, 'type': 'ekspresowa', 'speed': 110},
        'Częstochowa': {'distance': 100, 'type': 'ekspresowa', 'speed': 100}
    },
    'Częstochowa': {
        'Opole': {'distance': 110, 'type': 'lokalna', 'speed': 70},
        'Wrocław': {'distance': 100, 'type': 'ekspresowa', 'speed': 100},
        'Katowice': {'distance': 65, 'type': 'ekspresowa', 'speed': 100},
        'Kielce': {'distance': 130, 'type': 'lokalna', 'speed': 70}
    },
    'Katowice': {
        'Częstochowa': {'distance': 65, 'type': 'ekspresowa', 'speed': 100},
        'Kraków': {'distance': 80, 'type': 'autostrada', 'speed': 130},
        'Łódź': {'distance': 170, 'type': 'lokalna', 'speed': 80}
    },
    'Kraków': {
        'Katowice': {'distance': 80, 'type': 'autostrada', 'speed': 130},
        'Rzeszów': {'distance': 160, 'type': 'lokalna', 'speed': 80},
        'Kielce': {'distance': 110, 'type': 'lokalna', 'speed': 70}
    },
    'Łódź': {
        'Katowice': {'distance': 170, 'type': 'lokalna', 'speed': 80},
        'Wrocław': {'distance': 200, 'type': 'ekspresowa', 'speed': 110},
        'Warszawa': {'distance': 130, 'type': 'ekspresowa', 'speed': 120}
    },
    'Warszawa': {
        'Łódź': {'distance': 130, 'type': 'ekspresowa', 'speed': 120},
        'Lublin': {'distance': 170, 'type': 'ekspresowa', 'speed': 110},
        'Kielce': {'distance': 180, 'type': 'lokalna', 'speed': 90}
    },
    'Kielce': {
        'Częstochowa': {'distance': 130, 'type': 'lokalna', 'speed': 70},
        'Kraków': {'distance': 110, 'type': 'lokalna', 'speed': 70},
        'Warszawa': {'distance': 180, 'type': 'lokalna', 'speed': 90},
        'Rzeszów': {'distance': 160, 'type': 'lokalna', 'speed': 80}
    },
    'Lublin': {
        'Warszawa': {'distance': 170, 'type': 'ekspresowa', 'speed': 110},
        'Rzeszów': {'distance': 170, 'type': 'ekspresowa', 'speed': 100}
    },
    'Rzeszów': {
        'Kraków': {'distance': 160, 'type': 'lokalna', 'speed': 80},
        'Kielce': {'distance': 160, 'type': 'lokalna', 'speed': 80},
        'Lublin': {'distance': 170, 'type': 'ekspresowa', 'speed': 100}
    }
}

# Współrzędne miast
coords = {
    'Opole': (50.6751, 17.9213),
    'Wrocław': (51.1079, 17.0385),
    'Częstochowa': (50.8118, 19.1203),
    'Kraków': (50.0647, 19.9450),
    'Katowice': (50.2649, 19.0238),
    'Łódź': (51.7592, 19.4560),
    'Warszawa': (52.2297, 21.0122),
    'Kielce': (50.8661, 20.6286),
    'Lublin': (51.2465, 22.5684),
    'Rzeszów': (50.0412, 21.9991)
}

# Interaktywny wybór miast
miasta = {
    1: 'Kraków',
    2: 'Opole',
    3: 'Wrocław',
    4: 'Częstochowa',
    5: 'Katowice',
    6: 'Łódź',
    7: 'Warszawa',
    8: 'Kielce',
    9: 'Lublin',
    10: 'Rzeszów'
}

print("Wybierz miasto startowe:")
for k, v in miasta.items():
    print(f"{k}: {v}")
s_index = int(input("Podaj numer miasta startowego: "))
k_index = int(input("Podaj numer miasta docelowego: "))

start = miasta[s_index]
cel = miasta[k_index]

# Obliczenie heurystyki używając odległości haversine
heuristics = {
    city: (haversine(coords[city], coords[cel]) / 130) * 60  # czas w minutach
    for city in coords
}

trasa, czas = a_star(graph, heuristics, start, cel)
print("Ścieżka:", trasa)
print("Szacowany czas podróży:", format_time(czas))
wypisz_szczegoly_trasy(graph, trasa)

# Narysuj graf z zaznaczoną trasą
draw_graph(graph, trasa)

# Tworzenie mapy z widokiem na miasto startowe
m = folium.Map(location=coords[trasa[0]], zoom_start=7)

# Dodanie znaczników dla wszystkich miast
for city, position in coords.items():
    folium.Marker(position, tooltip=city).add_to(m)

# Dodanie ścieżki (trasa wyznaczona przez A*)
path_coords = [coords[city] for city in trasa]
folium.PolyLine(path_coords, color='red', weight=5, tooltip="Trasa A*").add_to(m)

# Zapisz mapę do pliku HTML
m.save('trasa.html')
import os

filepath = os.path.abspath('trasa.html')
print(f"\nMapa została zapisana tutaj: {filepath}")

import webbrowser
otworz = input("\nCzy chcesz otworzyć mapę w przeglądarce? (t/n): ")
if otworz.lower() == 't':
    webbrowser.open(f'file://{filepath}')
else:
    print("Możesz otworzyć plik ręcznie z podanej lokalizacji.")