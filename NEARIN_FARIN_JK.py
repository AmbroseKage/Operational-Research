import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Union, Any, Literal

class TSPSolver:
    """
    Klasa implementująca algorytmy NEARIN i FARIN do rozwiązywania problemu komiwojażera (TSP).
    
    NEARIN (Nearest Insertion) - wstawia najbliższy nieodwiedzony wierzchołek
    FARIN (Farthest Insertion) - wstawia najdalszy nieodwiedzony wierzchołek
    
    Oba algorytmy działają zachłannie, ale różnią się strategią wyboru kolejnych wierzchołków.
    """

    def __init__(self, 
                 points: List[Tuple[float, float]] = None, 
                 distance_matrix: np.ndarray = None, 
                 random_seed: Optional[int] = None):
        """
        Inicjalizuje solver TSP.
        
        Args:
            points (List[Tuple[float, float]]): Lista współrzędnych (x, y). 
                Wymagane, jeśli nie podano macierzy odległości.
            distance_matrix (np.ndarray): Macierz odległości. 
                Obliczana na podstawie punktów, jeśli nie podano.
            random_seed (int): Ziarno losowe, pozwala powtórzyć wyniki.
        """
        # Ustawiamy ziarno losowe, jeśli podano (zapewnia powtarzalność wyników)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        # Ziarno losowe ustawione – wyniki będą powtarzalne przy tym samym seedzie

        self.points = points
        self.distance_matrix = distance_matrix
        self.route = []          # Tutaj będzie finalna trasa
        self.route_length = 0    # Łączna długość trasy

        # Jeśli mamy podane punkty, ale nie mamy macierzy odległości, to ją wyliczamy
        if self.points and self.distance_matrix is None:
            self._precalculate_distances()

    def _precalculate_distances(self) -> None:
        """
        Oblicza i zapisuje macierz odległości między punktami.
        
        Dla każdego punktu i, j liczymy odległość euklidesową i wpisujemy w macierz.
        """
        n = len(self.points)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(self.points[i], self.points[j])
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist  # Symetria w TSP

    @staticmethod
    def _euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """
        Liczy odległość euklidesową między dwoma punktami (x1, y1) i (x2, y2).
        Przydatna do wyznaczenia kosztu przejścia w TSP.
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def solve(self, 
              start_node: Optional[int] = None, 
              use_2opt: bool = True,
              algorithm: Literal["NEARIN", "FARIN"] = "NEARIN") -> List[int]:
        """
        Rozwiązuje problem komiwojażera (TSP) algorytmem NEARIN lub FARIN.
        
        Args:
            start_node (int): Indeks węzła startowego. Jeśli None, wybierany losowo.
            use_2opt (bool): Czy stosować później optymalizację 2-opt (domyślnie False).
            algorithm (str): Wybór algorytmu - "NEARIN" (najbliższy) lub "FARIN" (najdalszy).
        
        Returns:
            List[int]: Kolejność odwiedzania węzłów w trasie.
        """
        # Sprawdzamy, czy mamy jakieś dane wejściowe
        if self.points is None and self.distance_matrix is None:
            raise ValueError("Należy podać punkty albo macierz odległości.")
        
        n = len(self.distance_matrix)

        # Jeśli start_node nie jest podany, wybieramy go losowo
        if start_node is None:
            start_node = random.randint(0, n - 1)
        
        # Zbiór nieodwiedzonych węzłów
        unvisited = set(range(n))
        # Dodajemy startowy węzeł do trasy
        self.route = [start_node]
        # Usuwamy go z nieodwiedzonych
        unvisited.remove(start_node)

        # Budujemy trasę, dodając kolejne wierzchołki w optymalnym miejscu
        while unvisited:
            # Wybieramy odpowiedni węzeł w zależności od wybranego algorytmu
            if algorithm == "NEARIN":
                # Dla NEARIN wybieramy najbliższy węzeł
                next_node, _ = self._find_nearest_node(unvisited)
            else:  # algorithm == "FARIN"
                # Dla FARIN wybieramy najdalszy węzeł
                next_node, _ = self._find_farthest_node(unvisited)
                
            # Szukamy najlepszego miejsca do wstawienia węzła w trasie
            best_pos = self._find_best_insertion_position(next_node)
            # Wstawiamy węzeł w tę pozycję
            self.route.insert(best_pos, next_node)
            # Usuwamy ten węzeł z nieodwiedzonych
            unvisited.remove(next_node)
        
        # Jeśli włączona jest opcjonalna optymalizacja 2-opt, stosujemy ją
        if use_2opt:
            self._apply_2opt()
            
        # Obliczamy finalną długość trasy, sumując odległości
        self.route_length = self._calculate_tour_length()
        return self.route
    
    def _find_nearest_node(self, unvisited: set) -> Tuple[int, float]:
        """
        Szuka węzła z unvisited, który jest najbliższy do któregokolwiek z aktualnej trasy.
        Używany w algorytmie NEARIN.
        
        Args:
            unvisited (set): Zbiór indeksów nieodwiedzonych węzłów.
        
        Returns:
            Tuple[int, float]: (najbliższy_węzeł, odległość).
        """
        min_dist = float('inf')
        nearest_node = None
        
        # Iterujemy po każdym węźle w trasie i sprawdzamy wszystkie nieodwiedzone
        for node in self.route:
            for unvisited_node in unvisited:
                dist = self.distance_matrix[node][unvisited_node]
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = unvisited_node
        
        # Zwracamy węzeł o minimalnej odległości
        return nearest_node, min_dist
    
    def _find_farthest_node(self, unvisited: set) -> Tuple[int, float]:
        """
        Szuka węzła z unvisited, który jest najdalszy od któregokolwiek z aktualnej trasy.
        Używany w algorytmie FARIN.
        
        Args:
            unvisited (set): Zbiór indeksów nieodwiedzonych węzłów.
        
        Returns:
            Tuple[int, float]: (najdalszy_węzeł, odległość).
        """
        max_dist = -1  # Szukamy maksymalnej odległości
        farthest_node = None
        
        # Dla każdego węzła z trasy i każdego nieodwiedzonego węzła
        for node in self.route:
            for unvisited_node in unvisited:
                dist = self.distance_matrix[node][unvisited_node]
                # Szukamy maksymalnej odległości zamiast minimalnej
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = unvisited_node
        
        # Zwracamy węzeł o maksymalnej odległości
        return farthest_node, max_dist
    
    def _find_best_insertion_position(self, node: int) -> int:
        """
        Znajduje najlepszą pozycję do wstawienia danego węzła, 
        żeby wzrost długości trasy był najmniejszy.
        
        Args:
            node (int): Indeks węzła, który chcemy wstawić.
        
        Returns:
            int: Najlepszy indeks w trasie, przed którym wstawiamy węzeł.
        """
        best_insertion_cost = float('inf')
        best_insertion_pos = 0
        
        # Sprawdzamy koszt wstawienia węzła pomiędzy każdą parę kolejnych węzłów w trasie
        for i in range(len(self.route)):
            j = (i + 1) % len(self.route)
            curr = self.route[i]
            next_node = self.route[j]
            
            cost_before = self.distance_matrix[curr][next_node]  # Koszt danej krawędzi
            cost_after = (self.distance_matrix[curr][node] + 
                          self.distance_matrix[node][next_node])  # Koszt po wstawieniu
            insertion_cost = cost_after - cost_before
            
            # Wybieramy minimalny przyrost kosztu
            if insertion_cost < best_insertion_cost:
                best_insertion_cost = insertion_cost
                best_insertion_pos = j
                
        return best_insertion_pos
    
    def _apply_2opt(self, max_iterations: int = 100) -> None:
        """
        Ulepsza trasę za pomocą lokalnego wyszukiwania 2-opt.
        
        Args:
            max_iterations (int): Maksymalna liczba iteracji 2-opt.
        
        2-opt sprawdza każdą parę krawędzi i może je zamieniać, 
        jeśli to skróci obecną trasę.
        """
        improved = True
        iteration = 0
        
        # Powtarzamy aż nie znajdziemy już żadnej poprawy lub osiągniemy limit iteracji
        while improved and iteration < max_iterations:
            improved = False
            best_improvement = 0
            best_i, best_j = -1, -1
            
            # Szukamy par krawędzi, których zamiana mogłaby poprawić trasę
            for i in range(len(self.route) - 2):
                for j in range(i + 2, len(self.route) - (1 if i == 0 else 0)):
                    a, b = self.route[i], self.route[i + 1]
                    c, d = self.route[j], self.route[(j + 1) % len(self.route)]
                    
                    current_distance = (self.distance_matrix[a][b] +
                                        self.distance_matrix[c][d])
                    new_distance = (self.distance_matrix[a][c] +
                                    self.distance_matrix[b][d])
                    
                    improvement = current_distance - new_distance
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i, j
            
            # Odwracamy fragment trasy między best_i a best_j, jeśli poprawa jest możliwa
            if best_improvement > 0:
                self.route[best_i + 1:best_j + 1] = self.route[best_j:best_i:-1]
                improved = True
                
            iteration += 1
    
    def _calculate_tour_length(self) -> float:
        """
        Liczy łączną długość powstałej trasy, 
        sumując odległości między kolejnymi punktami w route.
        """
        total_distance = 0
        # Sumujemy odległości kolejnych krawędzi w trasie
        for i in range(len(self.route)):
            total_distance += self.distance_matrix[
                self.route[i]][self.route[(i + 1) % len(self.route)]
            ]
        return total_distance
    
    def plot_tour(self, 
                  show_node_labels: bool = True, 
                  show_iterations: bool = False, 
                  node_size: int = 100,
                  title_prefix: str = "Rozwiązanie TSP",
                  save_path: Optional[str] = None) -> None:
        """
        Rysuje trasę na wykresie, pokazując kolejność odwiedzanych węzłów.
        
        Args:
            show_node_labels (bool): Czy wypisywać etykiety węzłów (ich indeksy).
            show_iterations (bool): Animacja wstawiania węzłów (niewdrożona).
            node_size (int): Wielkość kropek reprezentujących węzły.
            title_prefix (str): Tekst pojawiający się w tytule wykresu.
            save_path (str): Ścieżka do zapisu wykresu (jeśli chcemy zapisać zamiast wyświetlać).
        """
        if not self.points:
            raise ValueError("Nie można narysować trasy: brak danych punktów.")
            
        if not self.route:
            raise ValueError("Brak trasy do narysowania. Uruchom solve() najpierw.")

        # Tworzymy figurę i oś
        plt.figure(figsize=(10, 8))
        
        # Rozdzielamy punkty na listy współrzędnych X i Y
        xs = [self.points[i][0] for i in range(len(self.points))]
        ys = [self.points[i][1] for i in range(len(self.points))]
        
        # Rysujemy punkty symbolizujące miasta (węzły)
        plt.scatter(xs, ys, s=node_size, color='blue', zorder=2)
        
        # Etykiety węzłów (opcjonalne)
        if show_node_labels:
            for i, point in enumerate(self.points):
                plt.text(point[0] + 0.5, point[1] + 0.5, str(i), 
                         fontsize=9, ha='center', va='center')
        
        # Przygotowujemy listy współrzędnych do narysowania trasy (z domknięciem)
        route_x = [self.points[node][0] for node in self.route] + [self.points[self.route[0]][0]]
        route_y = [self.points[node][1] for node in self.route] + [self.points[self.route[0]][1]]
        # Rysujemy linię łączącą kolejne punkty w trasie
        plt.plot(route_x, route_y, 'r-', zorder=1, linewidth=1.5)
        
        # Rysujemy małe strzałki oznaczające kierunek (od węzła i do i+1)
        for i in range(len(self.route)):
            j = (i + 1) % len(self.route)
            xi, yi = self.points[self.route[i]]
            xj, yj = self.points[self.route[j]]
            
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            dx, dy = xj - xi, yj - yi
            plt.arrow(mx - 0.2*dx, my - 0.2*dy, 0.1*dx, 0.1*dy, 
                      head_width=0.15, head_length=0.2, fc='green', ec='green')
        
        # Dodajemy tytuł z długością trasy
        plt.title(f"{title_prefix}\nDługość trasy: {self.route_length:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Legendy do oznaczenia elementów wykresu
        plt.plot([], [], 'bo', label='Miasta')
        plt.plot([], [], 'r-', label='Trasa')
        plt.plot([], [], 'g>', label='Kierunek')
        plt.legend()

        # Jeśli chcemy zapisać obraz do pliku
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()


# ---- Funkcje pomocnicze ----

def find_best_starting_node(points: List[Tuple[float, float]], 
                           use_2opt: bool = True,
                           algorithm: str = "NEARIN",
                           random_seed: int = 42) -> Tuple[int, float, List[int]]:
    """
    Znajduje najlepszy wierzchołek startowy dla algorytmu TSP w podanych punktach.
    
    Args:
        points (List[Tuple[float, float]]): Lista współrzędnych.
        use_2opt (bool): Czy używać 2-opt do poprawy trasy.
        algorithm (str): Wybór algorytmu - "NEARIN" lub "FARIN".
        random_seed (int): Ziarno losowe do powtarzalności.
    
    Returns:
        (int, float, List[int]): (indeks_wierzchołka, długość_trasy, sama_trasa).
    """
    results = []
    # Przetestuj startowanie z każdego wierzchołka i wybierz ten z najkrótszą trasą
    for i in range(len(points)):
        solver = TSPSolver(points=points, random_seed=random_seed)
        route = solver.solve(start_node=i, use_2opt=use_2opt, algorithm=algorithm)
        results.append((i, solver.route_length, route))
    
    # Zwracamy krotkę z najlepszym (minimalnym) wynikiem
    return min(results, key=lambda x: x[1])


def compare_algorithms(points: List[Tuple[float, float]], 
                     start_node: Optional[int] = None, 
                     use_2opt: bool = False,
                     random_seed: int = 42) -> None:
    """
    Porównuje wyniki algorytmów NEARIN i FARIN.
    
    Args:
        points: Lista punktów do rozwiązania TSP
        start_node: Węzeł startowy (jeśli None, wybierany losowo)
        use_2opt: Czy stosować optymalizację 2-opt
        random_seed: Ziarno losowe dla powtarzalności
    """
    # Uruchamiamy NEARIN
    solver_nearin = TSPSolver(points=points, random_seed=random_seed)
    start_time = time.time()
    route_nearin = solver_nearin.solve(start_node=start_node, use_2opt=use_2opt, algorithm="NEARIN")
    time_nearin = time.time() - start_time
    
    # Uruchamiamy FARIN
    solver_farin = TSPSolver(points=points, random_seed=random_seed)
    start_time = time.time()
    route_farin = solver_farin.solve(start_node=start_node, use_2opt=use_2opt, algorithm="FARIN")
    time_farin = time.time() - start_time
    
    # Wyświetlamy wyniki
    print("\n--- Porównanie algorytmów NEARIN i FARIN ---")
    print(f"Węzeł startowy: {start_node if start_node is not None else 'losowy'}")
    print(f"Używa 2-opt: {use_2opt}")
    
    print("\nNEARIN:")
    print(f"Trasa: {route_nearin}")
    print(f"Długość trasy: {solver_nearin.route_length:.2f}")
    print(f"Czas wykonania: {time_nearin:.4f} sekund")
    
    print("\nFARIN:")
    print(f"Trasa: {route_farin}")
    print(f"Długość trasy: {solver_farin.route_length:.2f}")
    print(f"Czas wykonania: {time_farin:.4f} sekund")
    
    print("\nRóżnica długości trasy (FARIN - NEARIN): {:.2f}".format(
        solver_farin.route_length - solver_nearin.route_length))
    
    # Wizualizacja wyników
    solver_nearin.plot_tour(title_prefix="Rozwiązanie NEARIN TSP")
    solver_farin.plot_tour(title_prefix="Rozwiązanie FARIN TSP")


if __name__ == "__main__":
    print("\n--- Przykład oryginalny: Mapa miast ---")
    original_points = [
        (25, 54),  # 0: Suwałki
        (19, 54),  # 1: Gdańsk
        (15, 52),  # 2: Szczecin
        (17, 50),  # 3: Poznań
        (21, 51),  # 4: Warszawa
        (20, 49),  # 5: Łódź
        (18, 47),  # 6: Wrocław
        (23, 50),  # 7: Lublin
        (22, 45),  # 8: Kraków
        (24, 43),  # 9: Rzeszów
    ]
    
    # Porównanie algorytmów NEARIN i FARIN
    compare_algorithms(original_points, start_node=0, use_2opt=True)
    
    # Znajdowanie najlepszego węzła startowego
    print("\n--- Znajdowanie najlepszego wierzchołka startowego dla NEARIN ---")
    best_start_nearin, best_length_nearin, best_route_nearin = find_best_starting_node( original_points, algorithm="NEARIN")

    print(f"Najlepszy wierzchołek startowy: {best_start_nearin}")
    print(f"Najkrótsza długość trasy: {best_length_nearin:.2f}")
    print(f"Trasa: {best_route_nearin}")

    # Wizualizacja dla najlepszego wierzchołka startowego NEARIN
    best_solver_nearin = TSPSolver(points=original_points, random_seed=42)
    best_solver_nearin.solve(start_node=best_start_nearin, use_2opt=True, algorithm="NEARIN")
    best_solver_nearin.plot_tour(title_prefix=f"Rozwiązanie NEARIN TSP z najlepszym wierzchołkiem startowym ({best_start_nearin})")

    print("\n--- Znajdowanie najlepszego wierzchołka startowego dla FARIN ---")
    best_start_farin, best_length_farin, best_route_farin = find_best_starting_node(original_points, algorithm="FARIN")

    print(f"Najlepszy wierzchołek startowy: {best_start_farin}")
    print(f"Najkrótsza długość trasy: {best_length_farin:.2f}")
    print(f"Trasa: {best_route_farin}")

# Wizualizacja dla najlepszego wierzchołka startowego FARIN
    best_solver_farin = TSPSolver(points=original_points, random_seed=42)
    best_solver_farin.solve(start_node=best_start_farin, use_2opt=True, algorithm="FARIN")
    best_solver_farin.plot_tour(title_prefix=f"Rozwiązanie FARIN TSP z najlepszym wierzchołkiem startowym ({best_start_farin})")
    
    # Przykład z wyborem wierzchołka startowego przez użytkownika
    print("\n--- Przykład z podanymi wierzchołkami ---")
    fixed_points = [
        (3, 17),   # 0
        (5, 49),   # 1
        (26, 17),  # 2
        (6, 2),    # 3
        (24, 34),  # 4
        (35, 21),  # 5
        (21, 3),   # 6
        (10, 8),   # 7
        (21, 35),  # 8
        (21, 44),  # 9
        (15, 10),  # 10
        (0, 27),   # 11
        (49, 5),   # 12
        (38, 24),  # 13
        (4, 0),    # 14
    ]
    
    print("Dostępne wierzchołki:")
    for i, point in enumerate(fixed_points):
        print(f"Wierzchołek {i}: {point}")
    
    valid_choice = False
    user_start_node = None
    user_algorithm = "NEARIN"
    
    # Prosta logika pobierania węzła startowego i algorytmu od użytkownika
    while not valid_choice:
        try:
            user_input = input(f"Wybierz wierzchołek startowy (0-{len(fixed_points)-1}) lub Enter dla losowego: ")
            
            # Jeśli użytkownik nic nie podał, wybieramy losowy
            if user_input.strip() == "":
                valid_choice = True
            else:
                user_start_node = int(user_input)
                if 0 <= user_start_node < len(fixed_points):
                    valid_choice = True
                else:
                    print(f"Błąd: Wybierz wierzchołek z zakresu 0-{len(fixed_points)-1}")
        except ValueError:
            print("Błąd: Wprowadź liczbę całkowitą")
    
    # Wybór algorytmu
    valid_choice = False
    while not valid_choice:
        user_input = input("Wybierz algorytm (NEARIN/FARIN): ").upper()
        if user_input in ["NEARIN", "FARIN"]:
            user_algorithm = user_input
            valid_choice = True
        else:
            print("Błąd: Wprowadź NEARIN lub FARIN")
    
    # Tworzymy solver z punktami
    solver = TSPSolver(points=fixed_points, random_seed=42)
    
    # Mierzymy czas wykonania
    start_time = time.time()
    route = solver.solve(start_node=user_start_node, use_2opt=True, algorithm=user_algorithm)
    execution_time = time.time() - start_time
    
    print(f"Algorytm: {user_algorithm}")
    print(f"Trasa z wierzchołka {user_start_node if user_start_node is not None else 'losowego'}: {route}")
    print(f"Długość trasy: {solver.route_length:.2f}")
    print(f"Czas wykonania: {execution_time:.4f} sekund")
    
    # Rysujemy i wyświetlamy trasę
    solver.plot_tour(title_prefix=f"Rozwiązanie {user_algorithm} TSP z wierzchołkiem startowym {user_start_node if user_start_node is not None else '(losowym)'}")