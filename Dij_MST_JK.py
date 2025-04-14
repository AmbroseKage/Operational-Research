import matplotlib.pyplot as plt
import networkx as nx
import random

def dijkstra_prima_algorithm(G, a, s):
    """
    Znajduje Minimalne Drzewo Rozpinające (MST) metodą Dijkstry-Prima.
    
    Rozpoczyna od wierzchołka s, stopniowo wybierając krawędź o najmniejszej wadze
    łączącą drzewo z nowym wierzchołkiem. Zwraca zbiór krawędzi MST i sumę ich wag.
    """
    suma = 0
    A = set()
    V = list(G.keys())
    alfa = {}
    beta = {}
    
    for u in V:
        alfa[u] = 0
        beta[u] = float('inf')
    
    Q = set(V)
    beta[s] = 0
    Q.remove(s)
    u_star = s
    
    while Q:
        for u in list(Q):
            if u in [neighbor for neighbor, _ in G[u_star]]:
                if a[u_star][u] < beta[u]:
                    alfa[u] = u_star
                    beta[u] = a[u_star][u]
        
        min_beta = float('inf')
        for u in Q:
            if beta[u] < min_beta:
                min_beta = beta[u]
                u_star = u
        
        Q.remove(u_star)
        if u_star != s:
            A.add((alfa[u_star], u_star))
            suma += a[alfa[u_star]][u_star]
    
    return A, suma

def create_weight_function(graph):
    """
    Tworzy słownik wag dla grafu nieskierowanego z listy sąsiedztwa z wagami.
    Zapewnia symetrię krawędzi w obu kierunkach.
    """
    a = {}
    for node in graph:
        if node not in a:
            a[node] = {}
        for neighbor, weight in graph[node]:
            a[node][neighbor] = weight
            if neighbor not in a:
                a[neighbor] = {}
            a[neighbor][node] = weight
    return a

def convert_to_weighted_graph(adj_list):
    """
    Zamienia listę sąsiedztwa bez wag na listę z losowymi wagami (1–10).
    Zapewnia symetrię wag w grafie nieskierowanym.
    """
    weighted_graph = {}
    edge_weights = {}
    node_map = {node: idx for idx, node in enumerate(adj_list.keys())}
    
    for node in adj_list:
        zero_based_node = node_map[node]
        weighted_graph[zero_based_node] = []
    
    for node, neighbors in adj_list.items():
        zero_based_node = node_map[node]
        for neighbor in neighbors:
            zero_based_neighbor = node_map[neighbor]
            edge = (min(zero_based_node, zero_based_neighbor),
                    max(zero_based_node, zero_based_neighbor))
            if edge not in edge_weights:
                weight = random.randint(1, 10)
                edge_weights[edge] = weight
            else:
                weight = edge_weights[edge]
            weighted_graph[zero_based_node].append((zero_based_neighbor, weight))
    
    return weighted_graph

def visualize_graph_and_mst(graph, title_prefix=""):
    """
    Rysuje podany graf wraz z wyznaczonym MST (Dijkstra-Prima).
    Wyświetla informację o liczbie wierzchołków, krawędzi i całkowitej wadze MST.
    """
    a = create_weight_function(graph)
    mst_edges, total_weight = dijkstra_prima_algorithm(graph, a, 0)
    
    print(f"{title_prefix} - Krawędzie w MST:", mst_edges)
    print(f"{title_prefix} - Całkowita waga MST:", total_weight)
    
    G = nx.Graph()
    MST = nx.Graph()
    
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)
    
    for edge in mst_edges:
        u, v = edge
        edge_weight = a[u][v]
        MST.add_edge(u, v, weight=edge_weight)
    
    num_vertices = len(G.nodes())
    num_edges = len(G.edges())
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_weight='bold')
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"{title_prefix} - Graf Oryginalny\n"
              f"Wierzchołki: {num_vertices}, Krawędzie: {num_edges}")
    
    plt.subplot(1, 2, 2)
    nx.draw(MST, pos, with_labels=True, node_color='lightgreen',
            node_size=500, font_weight='bold')
    mst_edge_labels = {(u, v): d["weight"] for u, v, d in MST.edges(data=True)}
    nx.draw_networkx_edge_labels(MST, pos, edge_labels=mst_edge_labels)
    plt.title(f"{title_prefix} - MST\n"
              f"Wierzchołki: {num_vertices}, Krawędzie: {len(MST.edges())}")
    
    plt.tight_layout()
    plt.show()

# Przykład: graf z wagami
graph = {
    0: [(1, 4), (2, 3)],
    1: [(0, 4), (2, 1), (3, 2)],
    2: [(0, 3), (1, 1), (3, 4), (4, 6)],
    3: [(1, 2), (2, 4), (5, 5)],
    4: [(2, 6), (5, 7)],
    5: [(3, 5), (4, 7), (6, 8)],
    6: [(5, 8), (7, 2)],
    7: [(6, 2), (8, 3)],
    8: [(7, 3), (9, 4)],
    9: [(8, 4)]
}

a = create_weight_function(graph)
mst_edges, total_weight = dijkstra_prima_algorithm(graph, a, 0)
print("Krawędzie w MST:", mst_edges)
print("Całkowita waga MST:", total_weight)

G = nx.Graph()
MST = nx.Graph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

for edge in mst_edges:
    u, v = edge
    edge_weight = a[u][v]
    MST.add_edge(u, v, weight=edge_weight)

num_vertices = len(G.nodes())
num_edges = len(G.edges())

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=500, font_weight='bold')
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title(f"Graf Oryginalny\nWierzchołki: {num_vertices}, Krawędzie: {num_edges}")

plt.subplot(1, 2, 2)
nx.draw(MST, pos, with_labels=True, node_color='lightgreen',
        node_size=500, font_weight='bold')
mst_edge_labels = {(u, v): d["weight"] for u, v, d in MST.edges(data=True)}
nx.draw_networkx_edge_labels(MST, pos, edge_labels=mst_edge_labels)
plt.title(f"MST\nWierzchołki: {num_vertices}, Krawędzie: {len(MST.edges())}")

plt.tight_layout()
plt.show()

# Przykład 1
print("\n=== Przykład 1 ===")
GRAPH_1 = {
    0: [1, 2, 3, 4, 6],
    1: [0, 2, 4, 5, 7, 9],
    2: [0, 1, 3, 5, 8],
    3: [0, 2, 4, 6, 7, 8],
    4: [0, 1, 3, 5, 8, 9],
    5: [1, 2, 4, 6, 7, 9],
    6: [0, 3, 5, 7, 8, 9],
    7: [1, 3, 5, 6, 8, 9],
    8: [2, 3, 4, 6, 7, 9],
    9: [1, 4, 5, 6, 7, 8]
}
weighted_graph_1 = convert_to_weighted_graph(GRAPH_1)
visualize_graph_and_mst(weighted_graph_1, "Przykład 1")

# Przykład 2
print("\n=== Przykład 2 ===")
GRAPH_2 = {
    0: [1, 2, 3, 4, 5, 8],
    1: [0, 2, 4, 5, 7, 9],
    2: [0, 1, 3, 5, 6, 8, 9],
    3: [0, 2, 4, 6, 7, 8],
    4: [0, 1, 3, 5, 6, 7, 9],
    5: [0, 1, 2, 4, 6, 8, 9],
    6: [2, 3, 4, 5, 7, 8],
    7: [1, 3, 4, 6, 8, 9],
    8: [0, 2, 3, 5, 6, 7, 9],
    9: [1, 2, 4, 5, 7, 8]
}
weighted_graph_2 = convert_to_weighted_graph(GRAPH_2)
visualize_graph_and_mst(weighted_graph_2, "Przykład 2")

# Przykład 3
print("\n=== Przykład 3 ===")
GRAPH_3 = {
    0: [1, 2, 3, 5, 7, 8],
    1: [0, 2, 3, 4, 6, 9],
    2: [0, 1, 4, 5, 7, 8],
    3: [0, 1, 4, 5, 6, 9],
    4: [1, 2, 3, 6, 7, 8, 9],
    5: [0, 2, 3, 6, 7, 9],
    6: [1, 3, 4, 5, 8, 9],
    7: [0, 2, 4, 5, 8, 9],
    8: [0, 2, 4, 6, 7, 9],
    9: [1, 3, 4, 5, 6, 7, 8]
}
weighted_graph_3 = convert_to_weighted_graph(GRAPH_3)
visualize_graph_and_mst(weighted_graph_3, "Przykład 3")