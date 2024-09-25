class Graph(object): 
    def __init__(self, graph_input, is_matrix=False):
        """
        The constructor for the Graph class.
        :param: graph_input: a list of lists or a list of tuples
        :param: is_matrix: a boolean indicating whether the input is a list of lists or a list of tuples
        """
        # vertices
        self.order = 0
        # edges
        self.size = 0

        if is_matrix:
            self.adj_matrix = graph_input
            self.adj_list = self.convert_matrix_to_list(graph_input)
        else: 
            self.adj_list = graph_input
            self.adj_matrix = self.convert_list_to_matrix(graph_input)

    def convert_matrix_to_list(self, adj_matrix):
        adj_list = {}
        for i in range(len(adj_matrix)):
            adj_list[i] = []
            row = adj_matrix[i]
            for j in range(len(row)):
                edge = row[j]
                if edge != 0:
                    adj_list[i].append(j)
        return adj_list
    
    def convert_list_to_matrix(self, adj_list):
        size = len(adj_list)
        adj_matrix = [[0] * size for _ in range(size)]

        for vertex, edges in adj_list.items():
            for edge in edges: 
                adj_matrix[vertex][edge] = 1
        return adj_matrix 
    
    def add_vertex(self):
        self.adj_list[self.order] = []

        self.order += 1
        for row in self.adj_matrix:
            row.append(0)
        self.adj_matrix.append([0] * self.order)
        self.order += 1

    def add_edge(self, v1, v2, weight=1, allow_loops=False):
        if v1 == v2:
            raise ValueError("Theres a loop")
        
        self.adj_list[v1].append(v2)
        self.adj_matrix[v1][v2] = weight
        self.size += 1

    def del_edge(self, v1, v2):
        if v1 in self.adj_list and v2 in self.adj_list[v1]:
            self.adj_list[v1].remove(v2)

        if self.adj_matrix[v1][v2] != 0:
            self.adj_matrix[v1][v2] = 0

        if v1 in self.adj_list and v2 in self.adj_list[v1] or self.adj_matrix[v1][v2] != 0:
            self.size -= 1

    def del_vertex(self, vertex):
        if vertex in self.adj_list:
            for v in list(self.adj_list[vertex]):
                self.del_edge(vertex, v)
            del self.del_edge[vertex]

        for v in list(self.adj_list.keys()):
            if vertex in self.adj_list[v]:
                self.del_edge(v, vertex)
        
        del self.adj_matrix[vertex]
        for row in self.adj_matrix:
            del row[vertex]

        self.order -= 1