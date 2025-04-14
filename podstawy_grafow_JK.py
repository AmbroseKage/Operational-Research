from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import random
from tabulate import tabulate
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union


class Graph:
    """
    Reprezentuje graf nieskierowany.

    Atrybuty:
        adjacency_list: Słownik reprezentujący listę sąsiedztwa grafu.
                         Klucze to wierzchołki, a wartości to listy ich sąsiadów.
    """

    def __init__(self):
        """Inicjalizuje pusty graf."""
        self.adjacency_list: Dict[Any, List[Any]] = {}

    def add_vertex(self, vertex: Any) -> 'Graph':
        """
        Dodaje wierzchołek do grafu.

        Args:
            vertex: Wierzchołek do dodania.

        Returns:
            self: Obiekt grafu, umożliwiający łańcuchowe wywoływanie metod.
        """
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
        return self

    def add_edge(self, vertex1: Any, vertex2: Any) -> 'Graph':
        """
        Dodaje krawędź między dwoma wierzchołkami.

        Args:
            vertex1: Pierwszy wierzchołek.
            vertex2: Drugi wierzchołek.

        Returns:
            self: Obiekt grafu, umożliwiający łańcuchowe wywoływanie metod.
        """
        #! Dodaj wierzchołki, jeśli nie istnieją.
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        #! Dodaj krawędzie tylko, jeśli jeszcze nie istnieją.
        if vertex2 not in self.adjacency_list[vertex1]:
            self.adjacency_list[vertex1].append(vertex2)
        if vertex1 not in self.adjacency_list[vertex2]:
            self.adjacency_list[vertex2].append(vertex1)
        return self

    def remove_edge(self, vertex1: Any, vertex2: Any) -> 'Graph':
        """
        Usuwa krawędź między dwoma wierzchołkami, jeśli istnieje.

        Args:
            vertex1: Pierwszy wierzchołek.
            vertex2: Drugi wierzchołek.

        Returns:
            self: Obiekt grafu, umożliwiający łańcuchowe wywoływanie metod.
        """
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            if vertex2 in self.adjacency_list[vertex1]:
                self.adjacency_list[vertex1].remove(vertex2)
            if vertex1 in self.adjacency_list[vertex2]:
                self.adjacency_list[vertex2].remove(vertex1)
        return self

    def remove_vertex(self, vertex: Any) -> 'Graph':
        """
        Usuwa wierzchołek i wszystkie jego krawędzie z grafu.

        Args:
            vertex: Wierzchołek do usunięcia.

        Returns:
            self: Obiekt grafu, umożliwiający łańcuchowe wywoływanie metod.
        """
        if vertex in self.adjacency_list:
            #! Usuń krawędzie z innych wierzchołków.
            for other_vertex in self.adjacency_list:
                if vertex in self.adjacency_list[other_vertex]:
                    self.adjacency_list[other_vertex].remove(vertex)

            #! Usuń sam wierzchołek.
            del self.adjacency_list[vertex]
        return self

    def get_vertices(self) -> List[Any]:
        """
        Zwraca listę wszystkich wierzchołków w grafie.

        Returns:
            List[Any]: Lista wierzchołków.
        """
        return list(self.adjacency_list.keys())

    def get_neighbors(self, vertex: Any) -> List[Any]:
        """
        Zwraca listę sąsiadów dla danego wierzchołka.

        Args:
            vertex: Wierzchołek, którego sąsiedzi mają zostać zwrócone.

        Returns:
            List[Any]: Lista sąsiadów wierzchołka.
        """
        return self.adjacency_list.get(vertex, [])

    def has_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """
        Sprawdza, czy istnieje krawędź między dwoma wierzchołkami.

        Args:
            vertex1: Pierwszy wierzchołek.
            vertex2: Drugi wierzchołek.

        Returns:
            bool: True, jeśli krawędź istnieje, False w przeciwnym razie.
        """
        return vertex1 in self.adjacency_list and vertex2 in self.adjacency_list[vertex1]

    def get_edge_count(self) -> int:
        """
        Oblicza liczbę krawędzi w grafie.

        Returns:
            int: Liczba krawędzi.
        """
        return sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2

    def __str__(self) -> str:
        """
        Zwraca reprezentację grafu jako ciąg znaków.

        Returns:
            str: Reprezentacja grafu jako ciąg znaków.
        """
        result = "Graph:\n"
        for vertex in sorted(self.adjacency_list.keys(), key=lambda x: str(x)):
            neighbors = sorted(self.adjacency_list[vertex], key=lambda x: str(x))
            result += f"{vertex}: {neighbors}\n"
        return result

    def __len__(self) -> int:
        """
        Zwraca liczbę wierzchołków w grafie.

        Returns:
            int: Liczba wierzchołków.
        """
        return len(self.adjacency_list)

    def is_empty(self) -> bool:
        """
        Sprawdza, czy graf jest pusty (nie ma wierzchołków).

        Returns:
            bool: True, jeśli graf jest pusty, False w przeciwnym razie.
        """
        return len(self.adjacency_list) == 0


def bfs(graph: Graph, start_vertex: Any, numbering: Dict[Any, int] = None) -> Tuple[bool, Set[Any], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    Wykonuje przeszukiwanie wszerz (BFS) na grafie.

    Args:
        graph: Graf, na którym ma zostać wykonane BFS.
        start_vertex: Wierzchołek, od którego rozpoczyna się przeszukiwanie.
        numbering: Opcjonalny słownik do przechowywania numeracji wierzchołków.

    Returns:
        Tuple[bool, Set[Any], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
            - has_cycle: True, jeśli znaleziono cykl, False w przeciwnym razie.
            - visited: Zbiór odwiedzonych wierzchołków.
            - bfs_tree: Lista krawędzi w drzewie BFS.
            - cycle_edges: Lista krawędzi tworzących cykle.
    """
    if numbering is None:
        numbering = {}

    parent: Dict[Any, Any] = {vertex: None for vertex in graph.get_vertices()}
    visited: Set[Any] = set()
    bfs_tree: List[Tuple[Any, Any]] = []
    cycle_edges: List[Tuple[Any, Any]] = []
    current_number = max(numbering.values(), default=0) + 1

    #! Inicjalizuj wierzchołek startowy.
    numbering[start_vertex] = current_number
    visited.add(start_vertex)

    #! Kolejka FIFO dla BFS.
    queue = deque([start_vertex])

    while queue:
        current_vertex = queue.popleft()

        #! Przetwarzaj sąsiadów bieżącego wierzchołka.
        for neighbor in sorted(graph.get_neighbors(current_vertex), key=lambda x: str(x)):
            if neighbor not in visited:
                #! Nowy, nieodwiedzony wierzchołek.
                current_number += 1
                numbering[neighbor] = current_number
                parent[neighbor] = current_vertex
                visited.add(neighbor)
                queue.append(neighbor)
                bfs_tree.append((current_vertex, neighbor))
            elif neighbor != parent[current_vertex]:
                #! Wykryto cykl (krawędź do wierzchołka, który nie jest rodzicem).
                if (neighbor, current_vertex) not in cycle_edges and (current_vertex, neighbor) not in cycle_edges:
                    cycle_edges.append((current_vertex, neighbor))

    has_cycle = len(cycle_edges) > 0
    return has_cycle, visited, bfs_tree, cycle_edges


def analyze_graph(graph: Graph, start_vertex: Any = None) -> Dict[str, Any]:
    """
    Analizuje właściwości grafu za pomocą BFS.

    Args:
        graph: Graf do analizy.
        start_vertex: Wierzchołek startowy dla BFS (opcjonalny).

    Returns:
        Dict[str, Any]: Słownik zawierający wyniki analizy:
            - is_empty: True, jeśli graf jest pusty.
            - num_vertices: Liczba wierzchołków.
            - num_edges: Liczba krawędzi.
            - is_connected: True, jeśli graf jest spójny.
            - is_acyclic: True, jeśli graf jest acykliczny (bez cykli).
            - components: Lista spójnych składowych.
            - bfs_trees: Lista drzew BFS (krawędzie).
            - cycle_edges: Lista krawędzi tworzących cykle.
            - numbering: Numeracja wierzchołków w kolejności BFS.
            - formatted_result: Sformatowany tekstowy wynik analizy.
    """
    vertices = graph.get_vertices()
    num_vertices = len(vertices)
    num_edges = graph.get_edge_count()

    #! Sprawdź, czy graf jest pusty.
    if graph.is_empty():
        return {
            "is_empty": True,
            "message": "Graf jest pusty.",
            "num_vertices": 0,
            "num_edges": 0,
            "is_connected": False,
            "is_acyclic": True,
            "components": [],
            "formatted_result": "Graf jest pusty."
        }

    #! Określ wierzchołek startowy, jeśli nie został podany.
    if start_vertex is None and vertices:
        start_vertex = sorted(vertices, key=lambda x: str(x))[0]

    #! Sprawdź, czy podany wierzchołek istnieje w grafie.
    if start_vertex not in vertices:
        return {
            "is_empty": False,
            "message": f"Wierzchołek {start_vertex} nie istnieje w grafie.",
            "num_vertices": num_vertices,
            "num_edges": num_edges,
            "is_connected": None,
            "is_acyclic": None,
            "components": [],
            "formatted_result": f"Wierzchołek {start_vertex} nie istnieje w grafie."
        }

    #! Inicjalizuj dane do analizy.
    numbering: Dict[Any, int] = {}
    components: List[Set[Any]] = []
    component_trees: List[List[Tuple[Any, Any]]] = []
    all_cycle_edges: List[Tuple[Any, Any]] = []
    has_cycle = False

    #! Znajdź wszystkie spójne składowe.
    unvisited = set(vertices)

    while unvisited:
        current_start = next(iter(sorted(unvisited, key=lambda x: str(x))))
        component_has_cycle, component_visited, bfs_tree, cycle_edges = bfs(graph, current_start, numbering)

        #! Zapisz wyniki dla tej składowej.
        components.append(component_visited)
        component_trees.append(bfs_tree)
        all_cycle_edges.extend(cycle_edges)

        if component_has_cycle:
            has_cycle = True

        #! Zaktualizuj zbiór nieodwiedzonych wierzchołków.
        unvisited -= component_visited

    is_acyclic = not has_cycle
    is_connected = len(components) == 1 and num_vertices > 0

    #! Przygotuj dane o składowych.
    component_data = []
    for i, component in enumerate(components):
        component_vertices = sorted(component, key=lambda x: str(x))
        is_isolated = len(component) == 1 and not graph.get_neighbors(list(component)[0])

        vertices_data = []
        for vertex in sorted(component, key=lambda x: str(x)):
            num = numbering.get(vertex, "nieodwiedzony")
            vertices_data.append({"vertex": vertex, "number": num})

        component_data.append({
            "id": i + 1,
            "vertices": component_vertices,
            "is_isolated": is_isolated,
            "vertices_data": vertices_data
        })

    #! Wygeneruj sformatowany wynik analizy.
    result = "\n" + "═" * 50 + "\n"
    result += f"ANALIZA GRAFU\n"
    result += "═" * 50 + "\n\n"

    stats_table = [
        ["Liczba wierzchołków", num_vertices],
        ["Liczba krawędzi", num_edges],
        ["Liczba składowych spójnych", len(components)],
        ["Czy graf jest spójny", "TAK" if is_connected else "NIE"],
        ["Czy graf jest acykliczny", "TAK" if is_acyclic else "NIE"]
    ]

    result += tabulate(stats_table, headers=["Właściwość", "Wartość"], tablefmt="pretty") + "\n\n"

    result += "SKŁADOWE SPÓJNE GRAFU\n"
    result += "═" * 50 + "\n\n"

    for comp in component_data:
        if comp["is_isolated"]:
            result += f"⬤ Składowa {comp['id']}: [IZOLOWANY WIERZCHOŁEK] {comp['vertices']}\n"
        else:
            result += f"⬤ Składowa {comp['id']}: {comp['vertices']}\n"

        vertex_table = [[v["vertex"], v["number"]] for v in comp["vertices_data"]]
        result += tabulate(vertex_table, headers=["Wierzchołek", "Numer BFS"], tablefmt="grid") + "\n\n"

    result += "PODSUMOWANIE\n"
    result += "═" * 50 + "\n"
    if is_connected:
        result += "✓ Graf jest SPÓJNY.\n"
    else:
        result += "✗ Graf jest NIESPÓJNY.\n"
        result += f"  Wykryto {len(components)} składowych spójnych.\n"

    if is_acyclic:
        result += "✓ Graf jest ACYKLICZNY (nie zawiera cykli).\n"
    else:
        result += "✗ Graf zawiera CYKLE.\n"
        result += f"  Wykryto {len(all_cycle_edges)} krawędzi tworzących cykle.\n"

    #! Zwróć wyniki analizy.
    return {
        "is_empty": False,
        "message": None,
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "is_connected": is_connected,
        "is_acyclic": is_acyclic,
        "components": component_data,
        "bfs_trees": component_trees,
        "cycle_edges": all_cycle_edges,
        "numbering": numbering,
        "formatted_result": result
    }


def visualize_graph(graph: Graph, title: str = "Wizualizacja Grafu",
                    save_path: Optional[str] = None, show_graph: bool = True) -> None:
    """
    Wizualizuje graf za pomocą NetworkX i Matplotlib.

    Args:
        graph: Graf do wizualizacji.
        title: Tytuł wykresu.
        save_path: Ścieżka do zapisu wizualizacji do pliku (opcjonalna).
        show_graph: Czy wyświetlić graf w interaktywnym oknie.
    """
    vertices = graph.get_vertices()

    if not vertices:
        print("Graf jest pusty, nie można go zwizualizować.")
        return

    #! Utwórz graf NetworkX.
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(vertices)

    #! Dodaj krawędzie (tylko raz dla każdej krawędzi).
    edges = [(v, n) for v in vertices for n in graph.get_neighbors(v) if v < n]
    nx_graph.add_edges_from(edges)

    #! Przeanalizuj graf.
    analysis = analyze_graph(graph)

    #! Przygotuj kolory.
    #! Użyj ciemniejszych kolorów.
    dark_colors = ['#444444', '#555555', '#666666', '#777777', '#888888', '#999999']
    component_colors = {}
    node_colors = {}
    edge_colors = {}
    edge_styles = {}

    #! Przypisz kolory do składowych.
    for i, component in enumerate(analysis["components"]):
        color_idx = i % len(dark_colors)
        color = dark_colors[color_idx]
        component_colors[i + 1] = color

        for v in component["vertices"]:
            node_colors[v] = color

    #! Domyślne style i kolory dla krawędzi.
    for edge in edges:
        edge_colors[edge] = node_colors.get(edge[0], '#222222')  #! Jeszcze ciemniejszy domyślny
        edge_styles[edge] = 'solid'

    #! Zbierz krawędzie drzewa BFS.
    bfs_edges = []
    for tree in analysis["bfs_trees"]:
        bfs_edges.extend(tree)
        bfs_edges.extend([(v2, v1) for v1, v2 in tree])  #! Dodaj w obu kierunkach

    #! Oznacz krawędzie tworzące cykle.
    cycle_edges = []
    for edge in analysis["cycle_edges"]:
        cycle_edges.append(edge)
        cycle_edges.append((edge[1], edge[0]))  #! Dodaj w obu kierunkach
        edge_colors[(edge[0], edge[1])] = 'red'
        edge_colors[(edge[1], edge[0])] = 'red'
        edge_styles[(edge[0], edge[1])] = 'dashed'
        edge_styles[(edge[1], edge[0])] = 'dashed'

    #! Rozmiar węzłów zależny od stopnia.
    node_sizes = {v: 300 + 100 * len(graph.get_neighbors(v)) for v in vertices}

    #! Wybierz odpowiedni układ grafu.
    if analysis["is_connected"] and analysis["num_edges"] > analysis["num_vertices"] * 1.5:
        pos = nx.kamada_kawai_layout(nx_graph)
    elif analysis["num_vertices"] <= 15:
        pos = nx.spring_layout(nx_graph, seed=42, k=0.5)
    else:
        pos = nx.shell_layout(nx_graph)

    #! Utwórz wykres.
    plt.figure(figsize=(12, 8))

    #! Narysuj krawędzie.
    for edge in edges:
        nx.draw_networkx_edges(
            nx_graph, pos, edgelist=[edge],
            width=2.0 if edge in bfs_edges or (edge[1], edge[0]) in bfs_edges else 1.0,
            alpha=0.7,
            edge_color=edge_colors.get(edge, '#111111'),  #! Najciemniejszy domyślny
            style=edge_styles.get(edge, 'solid')
        )

    #! Narysuj węzły.
    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_color=[node_colors.get(node, '#333333') for node in nx_graph.nodes()],  #! Ciemny domyślny
        node_size=[node_sizes.get(node, 500) for node in nx_graph.nodes()],
        alpha=0.8
    )

    #! Narysuj etykiety węzłów.
    nx.draw_networkx_labels(
        nx_graph, pos,
        font_size=10,
        font_weight='bold',
        font_color='white'  #! Białe etykiety dla kontrastu
    )

    #! Dodaj legendę.
    legend_elements = []

    for comp_id, color in component_colors.items():
        legend_elements.append(mpatches.Patch(
            color=color,
            label=f'Składowa {comp_id}'
        ))

    if analysis["cycle_edges"]:
        legend_elements.append(mpatches.Patch(
            color='red',
            label='Krawędź cyklu'
        ))

    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')

    #! Tytuł i formatowanie.
    plt.title(
        f"{title}\n"
        f"({analysis['num_vertices']} wierzchołków, {analysis['num_edges']} krawędzi, "
        f"{'spójny' if analysis['is_connected'] else 'niespójny'}, "
        f"{'acykliczny' if analysis['is_acyclic'] else 'zawiera cykle'})"
    )

    plt.axis('off')
    plt.tight_layout()

    #! Zapisz wykres do pliku.
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graf zapisany do pliku: {save_path}")

    #! Wyświetl wykres.
    if show_graph:
        plt.show()
    else:
        plt.close()


def generate_random_graph(num_vertices: int, num_edges: int, connected: bool = False) -> Graph:
    """
    Generuje losowy graf.

    Args:
        num_vertices: Liczba wierzchołków.
        num_edges: Liczba krawędzi.
        connected: Czy graf ma być spójny.

    Returns:
        Graph: Wygenerowany losowy graf.

    Raises:
        ValueError: Jeśli parametry są nieprawidłowe.
    """
    #! Sprawdź poprawność parametrów.
    if num_vertices <= 0:
        raise ValueError("Liczba wierzchołków musi być dodatnia")

    if num_edges < 0:
        raise ValueError("Liczba krawędzi nie może być ujemna")

    #! Oblicz maksymalną liczbę krawędzi.
    max_edges = (num_vertices * (num_vertices - 1)) // 2
    if num_edges > max_edges:
        print(f"Uwaga: Zmniejszono liczbę krawędzi z {num_edges} do maksymalnej możliwej: {max_edges}")
        num_edges = max_edges

    #! Utwórz graf.
    graph = Graph()

    #! Dodaj wierzchołki.
    for i in range(num_vertices):
        graph.add_vertex(i)

    #! Dla grafu spójnego, najpierw utwórz drzewo rozpinające.
    if connected and num_vertices > 1:
        #! Utwórz minimalne drzewo rozpinające, aby zapewnić spójność.
        vertices_to_connect = list(range(1, num_vertices))
        for i in range(1, num_vertices):
            #! Wybierz losowy wierzchołek z już połączonych.
            j = random.randint(0, i - 1)
            graph.add_edge(i, j)

        remaining_edges = num_edges - (num_vertices - 1)
    else:
        remaining_edges = num_edges

    #! Dodaj pozostałe losowe krawędzie.
    edges_added = 0
    max_attempts = max_edges * 10  #! Ogranicz liczbę prób, aby uniknąć nieskończonej pętli.
    attempts = 0

    #! Lista wszystkich możliwych krawędzi.
    possible_edges = [(u, v) for u in range(num_vertices) for v in range(u + 1, num_vertices)
                      if not graph.has_edge(u, v)]

    #! Losowo wybierz krawędzie.
    random.shuffle(possible_edges)

    #! Dodawaj krawędzie, aż do osiągnięcia żądanej liczby.
    for u, v in possible_edges:
        if edges_added >= remaining_edges:
            break
        graph.add_edge(u, v)
        edges_added += 1

    return graph


def create_star_graph(n: int) -> Graph:
    """Tworzy graf gwiazdę z n wierzchołkami."""
    g = Graph()
    for i in range(n):
        g.add_vertex(i)  #! Dodaj wierzchołki od 0 do n-1.
    for i in range(1, n):
        g.add_edge(0, i)  #! Wierzchołek 0 łączy się z resztą.
    return g


def create_connected_cyclic_graph() -> Graph:
    """Tworzy graf spójny z cyklami (10 wierzchołków, min 20 krawędzi)."""
    g = Graph()
    for i in range(10):
        g.add_vertex(i)

    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (5, 9),
        (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def create_disconnected_cyclic_graph() -> Graph:
    """Tworzy graf niespójny z cyklami (10 wierzchołków, min 20 krawędzi)."""
    g = Graph()
    #! Pierwsza składowa (wierzchołki 0-4).
    for i in range(5):
        g.add_vertex(i)

    edges1 = [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
        (1, 4), (2, 3), (2, 4), (3, 4)
    ]
    for u, v in edges1:
        g.add_edge(u, v)

    #! Dodatkowe krawędzie w pierwszej składowej.
    g.add_edge(0, 4)
    g.add_edge(1, 0)
    g.add_edge(2, 1)
    g.add_edge(3, 2)
    g.add_edge(4, 3)

    #! Druga składowa (wierzchołki 5-9).
    for i in range(5, 10):
        g.add_vertex(i)

    edges2 = [
        (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8),
        (6, 9), (7, 8), (7, 9), (8, 9)
    ]
    for u, v in edges2:
        g.add_edge(u, v)

    #! Dodatkowe krawędzie w drugiej składowej.
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    g.add_edge(9, 5)
    return g


def create_connected_acyclic_graph() -> Graph:
    """Tworzy graf spójny acykliczny (drzewo) (10 wierzchołków, 9 krawędzi)."""
    g = Graph()
    for i in range(10):
        g.add_vertex(i)

    edges = [
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (4, 8), (5, 9)
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def main():
    """
    Główna funkcja demonstracyjna.
    """
    output_dir = "graph_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #! Przykłady grafów do demonstracji.
    examples = [
        {
            "name": "Graf Gwiazda z 14 wierzchołkami",
            "build": lambda: create_star_graph(14),
            "start": 0
        },
        {
            "name": "Graf spójny z cyklami (10 wierzchołków, min 20 krawędzi)",
            "build": lambda: create_connected_cyclic_graph(),
            "start": 0
        },
        {
            "name": "Graf niespójny z cyklami (10 wierzchołków, min 20 krawędzi)",
            "build": lambda: create_disconnected_cyclic_graph(),
            "start": 0
        },
        {
            "name": "Graf spójny acykliczny (drzewo) (10 wierzchołków, 9 krawędzi)",
            "build": lambda: create_connected_acyclic_graph(),
            "start": 0
        }
    ]

    print("\n" + "=" * 50)
    print(" DEMONSTRACJA ANALIZY I WIZUALIZACJI GRAFÓW")
    print("=" * 50)

    for i, example in enumerate(examples):
        print(f"\n\n{'#' * 50}")
        print(f"# PRZYKŁAD {i + 1}: {example['name']}")
        print(f"{'#' * 50}\n")

        #! Utwórz graf zgodnie z definicją przykładu.
        if "edges" in example:
            g = Graph()
            for v1, v2 in example["edges"]:
                g.add_edge(v1, v2)
        elif "build" in example:
            g = example["build"]()
        else:
            g = Graph()

        #! Wyświetl graf.
        print(g)

        #! Przeanalizuj graf.
        start_vertex = example.get("start", None)
        analysis = analyze_graph(g, start_vertex)
        print(analysis["formatted_result"])

        #! Zwizualizuj graf.
        save_file = f"{output_dir}/graph_{i + 1}_{timestamp}.png"
        visualize_graph(g, title=f"Przykład {i + 1}: {example['name']}", save_path=save_file)


def create_example_graph():
    """Tworzy prosty przykładowy graf do testowania."""
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(4, 5)
    return g


if __name__ == "__main__":
    main()