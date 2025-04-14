# Operational Research

---

## Project Overview

This repository contains full solutions for laboratory assignments from the courses **Operations Research 1** and **Graph Algorithms**. It includes implementations and analysis of classical graph algorithms in Python, with a focus on performance, computational complexity, and practical applications.

---

## Table of Contents

- [Lab 2 – Graph Representation and BFS](#lab-2)
- [Lab 3 – Minimum Spanning Tree (MST)](#lab-3)
- [Lab 4 – Shortest Paths (Floyd-Warshall, A\*)](#lab-4)
- [Lab 5 – Traveling Salesman Problem (TSP)](#lab-5)
- [Technologies and Tools](#technologies-and-tools)
- [Sources](#sources)
- [Conclusion](#conclusion)

---

## Lab 2

### Topic: Graph Representation and BFS

- Graphs implemented as adjacency matrices and adjacency lists
- Advantages and disadvantages of each representation
- Breadth-First Search (BFS) with detection of:
  - graph connectivity,
  - presence of cycles.
- Analysis of three types of graphs:
  - connected with a cycle,
  - disconnected with a cycle,
  - connected acyclic (tree).

---

## Lab 3

### Topic: Minimum Spanning Tree (MST)

- Implementation of the **Dijkstra-Prim** algorithm (a greedy variant of Prim’s algorithm)
- Time complexity: O(V²), with priority handling
- Key graph properties required for proper MST behavior
- Operation description and use-case analysis (network design, transportation, clustering)
- Alternative methods: Kruskal, Christofides, dynamic programming approaches

---

## Lab 4

### Topic: Shortest Paths in a Graph

#### Floyd-Warshall Algorithm

- All-pairs shortest path calculation
- Handles negative weights (no negative cycles)
- Distance and predecessor matrices used for path reconstruction

#### A\* Algorithm

- Heuristic based on geographic distance (Haversine / 130 km/h)
- Extended city graph with attributes: distance, speed, road type
- Realistic travel time analysis with visualization (Folium)

---

## Lab 5

### Topic: Traveling Salesman Problem (TSP)

- Heuristic algorithms implemented:
  - **NEARIN** (nearest insertion),
  - **FARIN** (farthest insertion)
- Optional local optimization using the **2-opt** algorithm
- Analysis of:
  - starting node impact,
  - construction strategy,
  - spatial distribution of points
- Computational complexity: O(n³), with potential for dynamic programming or Christofides' extension

---

## Technologies and Tools

- **Python 3.12.1**
- IDE: **Visual Studio Code**
- Libraries:
  - `networkx` – graph analysis
  - `math`, `heapq`, `collections`, `numpy` – basic computations and data structures
  - `folium` – map-based route visualization
- Tested on: Windows 11 / Linux

---

## Files

- **`A_star_JK.py`**  
  A* (A-star) algorithm with an advanced heuristic estimating travel time. The graph contains attributes such as distance, road type, and speed limits.

- **`Dij_MST_JK.py`**  
  Dijkstra’s algorithm for shortest paths and the Dijkstra-Prim algorithm for minimum spanning trees, with graphical visualization using `networkx` and `matplotlib`.

- **`NEARIN_FARIN_JK.py`**  
  Heuristic approaches to the TSP problem: NEARIN (nearest insertion) and FARIN (farthest insertion), with optional 2-opt optimization.

- **`podstawy_grafow_JK.py`**  
  Foundational functions and graph representations, including adjacency lists and matrices.

- **`README.md`**  
  This file. It contains a full project description, instructions, and file documentation.

---

## Sources

- [GeeksForGeeks](https://www.geeksforgeeks.org/)
- [DataCamp](https://www.datacamp.com/)
- [FavTutor](https://favtutor.com/)
- [Wikipedia – A\*, Floyd-Warshall](https://en.wikipedia.org/wiki/)
- [RedBlobGames – Pathfinding](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- ChatGPT and GitHub Copilot – assistance with testing and code review

---

## Conclusion

This project enabled in-depth learning of graph structures and algorithms, their properties, and their practical applications. The implemented algorithms were tested and analyzed in terms of complexity, efficiency, and usefulness for modeling routes, networks, transportation systems, and optimization problems.

---

> The repository includes source code, test data, and result plots.
