import numpy as np


class Graph:
    def __init__(self, graph_representation):
        if isinstance(graph_representation, list):
            # input is an adjacency list
            self.adj_list = graph_representation
            self.order = len(graph_representation)
            self.size = sum(len(adj_list) for adj_list in graph_representation)
            self.adj_matrix = self._convert_to_adjacency_matrix(graph_representation)
        elif isinstance(graph_representation, (tuple, list)):
            # input is an adjacency matrix
            self.adj_matrix = graph_representation
            self.order = len(graph_representation)
            self.size = sum(1 for row in graph_representation for val in row if val != 0)
            self.adj_list = self._convert_to_adjacency_list(graph_representation)
        else:
            raise ValueError("Invalid input.")
            

        self.vertex_weights = [None] * self.order
        self.edge_weights = self._initialize_edge_weights()

    def _convert_to_adjacency_matrix(self, adj_list):
        order = len(adj_list)
        adj_matrix = [[0] * order for _ in range(order)]
        for i in range(order):
            for neighbor in adj_list[i]:
                adj_matrix[i][neighbor] = 1
        return adj_matrix

    def _convert_to_adjacency_list(self, adj_matrix):
        adj_list = []
        for i in range(len(adj_matrix)):
            neighbors = []
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] != 0:
                    neighbors.append(j)
            adj_list.append(neighbors)
        return adj_list
    
    def _initialize_edge_weights(self):
        edge_weights = [[None] * self.order for _ in range(self.order)]
        for i in range(self.order):
            for neighbor in self.adj_list[i]:
                # Initialize edge weights to None
                edge_weights[i][neighbor] = None  
        return edge_weights

    def get_vertex_weight(self, vertex):
        return self.vertex_weights[vertex]

    def set_vertex_weight(self, vertex, weight):
        self.vertex_weights[vertex] = weight

    def get_edge_weight(self, start_vertex, end_vertex):
        return self.edge_weights[start_vertex][end_vertex]

    def set_edge_weight(self, start_vertex, end_vertex, weight):
        self.edge_weights[start_vertex][end_vertex] = weight

    def add_vertex(self):
        self.order += 1
        self.adj_list.append([])
        self.vertex_weights.append(None)
        for row in self.adj_matrix:
            row.append(0)
        self.adj_matrix.append([0] * self.order)
        for row in self.edge_weights:
            row.append(None)
        self.edge_weights.append([None] * self.order)

    def del_vertex(self, vertex):
        if vertex < self.order:
            del self.adj_list[vertex]
            del self.vertex_weights[vertex]
            del self.adj_matrix[vertex]
            del self.edge_weights[vertex]
            for row in self.adj_matrix:
                del row[vertex]
            for row in self.edge_weights:
                del row[vertex]
            self.order -= 1
        else:
            raise ValueError("Vertex index out of range.")
        
    def add_edge(self, start_vertex, end_vertex):
        if start_vertex < self.order and end_vertex < self.order:
            self.adj_list[start_vertex].append(end_vertex)
            self.adj_matrix[start_vertex][end_vertex] = 1
            self.edge_weights[start_vertex][end_vertex] = None
            self.size += 1
        else:
            raise ValueError("Invalid vertices for adding an edge.")

    def del_edge(self, start_vertex, end_vertex):
        if start_vertex < self.order and end_vertex < self.order:
            if end_vertex in self.adj_list[start_vertex]:
                self.adj_list[start_vertex].remove(end_vertex)
                self.adj_matrix[start_vertex][end_vertex] = 0
                self.edge_weights[start_vertex][end_vertex] = None
                self.size -= 1
            else:
                raise ValueError("Edge does not exist between the given vertices.")
        else:
            raise ValueError("Invalid vertices for deleting an edge.")

    def is_directed(self):
        for i in range(self.order):
            for j in range(self.order):
                if self.adj_matrix[i][j] != self.adj_matrix[j][i]:
                    return True
        return False

    def is_connected(self):
        visited = [False] * self.order

        def dfs(v):
            visited[v] = True
            for neighbor in self.adj_list[v]:
                if not visited[neighbor]:
                    dfs(neighbor)

        dfs(0)
        if all(visited):
            return True
        return False
    
    def is_unilaterally_connected(self):
        if not self.is_directed():
            return False

        def dfs(v, transpose=False):
            visited = [False] * self.order
            stack = [v]
            visited[v] = True

            while stack:
                current = stack.pop()
                for neighbor in self.adj_list[current]:
                    if not visited[neighbor]:
                        if transpose:
                            if self.adj_matrix[neighbor][current] == 1:
                                visited[neighbor] = True
                                stack.append(neighbor)
                        else:
                            if self.adj_matrix[current][neighbor] == 1:
                                visited[neighbor] = True
                                stack.append(neighbor)
            return visited

        for i in range(self.order):
            from_i = dfs(i)
            to_i = dfs(i, transpose=True)
            for j in range(self.order):
                if from_i[j] != to_i[j]:
                    return False
        return True

    def is_weakly_connected(self):
        if not self.is_directed():
            return False

        undirected_adj_matrix = [[0] * self.order for _ in range(self.order)]

        for i in range(self.order):
            for j in range(self.order):
                undirected_adj_matrix[i][j] = max(self.adj_matrix[i][j], self.adj_matrix[j][i])

        undirected_graph = Graph(undirected_adj_matrix)
        return undirected_graph.is_connected()
    
    def is_tree(self):
        if self.order - self.size != 1:
            return False  # A tree has one fewer edges than vertices

        visited = [False] * self.order

        def dfs(v, parent):
            visited[v] = True
            for neighbor in self.adj_list[v]:
                if not visited[neighbor]:
                    if dfs(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False

        if dfs(0, -1) and all(visited):
            return True
        return False

    def components(self):
        visited = [False] * self.order
        count = 0

        def dfs(v):
            visited[v] = True
            for neighbor in self.adj_list[v]:
                if not visited[neighbor]:
                    dfs(neighbor)

        for i in range(self.order):
            if not visited[i]:
                dfs(i)
                count += 1
        return count

    def girth(self):
        min_cycle_length = float('inf')

        def dfs(v, parent, start, depth):
            nonlocal min_cycle_length
            for neighbor in self.adj_list[v]:
                if neighbor != parent:
                    if neighbor == start:
                        min_cycle_length = min(min_cycle_length, depth + 1)
                    else:
                        dfs(neighbor, v, start, depth + 1)

        for i in range(self.order):
            dfs(i, -1, i, 0)
        return min_cycle_length if min_cycle_length != float('inf') else 0

    def circumference(self):
        max_cycle_length = 0

        def dfs(v, parent, start, depth):
            nonlocal max_cycle_length
            for neighbor in self.adj_list[v]:
                if neighbor != parent:
                    if neighbor == start:
                        max_cycle_length = max(max_cycle_length, depth + 1)
                    else:
                        dfs(neighbor, v, start, depth + 1)

        for i in range(self.order):
            dfs(i, -1, i, 0)
        return max_cycle_length
    
    
    def to_numpy_array(self):
        return np.array(self.adj_matrix)
    
"""# Creating a graph using an adjacency list
adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

print("Order:", graph_from_adj_list.order)
print("Size:", graph_from_adj_list.size)
print("Adjacency List:", graph_from_adj_list.adj_list)
print("Adjacency Matrix:", graph_from_adj_list.adj_matrix)

# Example usage:

adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

# Setting weights for vertices and edges
graph_from_adj_list.set_vertex_weight(0, 5)
graph_from_adj_list.set_edge_weight(0, 1, 10)

print("Vertex 0 Weight:", graph_from_adj_list.get_vertex_weight(0))
print("Edge Weight between 0 and 1:", graph_from_adj_list.get_edge_weight(0, 1))

# Adding a vertex
graph_from_adj_list.add_vertex()
print("Order after adding a vertex:", graph_from_adj_list.order)

# Removing a vertex
graph_from_adj_list.del_vertex(0)
print("Order after removing a vertex:", graph_from_adj_list.order)

adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

# Adding an edge
graph_from_adj_list.add_edge(0, 1)
print("Size after adding an edge:", graph_from_adj_list.size)

# Removing an edge
graph_from_adj_list.del_edge(0, 2)
print("Size after removing an edge:", graph_from_adj_list.size)

# Checking if the graph is directed
print("Is Directed:", graph_from_adj_list.is_directed())

# Checking if the graph is connected
print("Is Connected:", graph_from_adj_list.is_connected())


# Example usage:

adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

# Checking if the graph is unilaterally connected
print("Is Unilaterally Connected:", graph_from_adj_list.is_unilaterally_connected())

# Checking if the graph is weakly connected
print("Is Weakly Connected:", graph_from_adj_list.is_weakly_connected())


adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

# Checking if the graph is a tree
print("Is Tree:", graph_from_adj_list.is_tree())

# Finding the number of components in the graph
print("Number of Components:", graph_from_adj_list.components())

# Finding the length of the shortest cycle (girth)
print("Girth:", graph_from_adj_list.girth())

# Finding the length of the longest cycle (circumference)
print("Circumference:", graph_from_adj_list.circumference())

adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

adj_list_representation = [
    [1, 2],     # Node 0 is connected to nodes 1 and 2
    [0, 2],     # Node 1 is connected to nodes 0 and 2
    [0, 1]      # Node 2 is connected to nodes 0 and 1
]

graph_from_adj_list = Graph(adj_list_representation)

# Converting adjacency matrix to a NumPy array
numpy_array = graph_from_adj_list.to_numpy_array()
print("Adjacency Matrix as a NumPy array:")
print(numpy_array)
"""

    