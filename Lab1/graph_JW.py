from matrix import Matrix
from vector import Vector

class Graph(object): 
    def __init__(self, graph_input=None) -> None:
        """
        The constructor for the Graph class.
        :param: graph_input: a list of lists or a list of tuples
        """
        self.adj_list = None
        self.adj_matrix = None

        if type(graph_input) is dict:
            self.adj_list = graph_input
            self.convert_list_to_matrix()
        elif type(graph_input) is Matrix:
            self.adj_matrix = graph_input
            self.convert_matrix_to_list()

    @property
    def adj_list(self) -> Vector:
        return self._adj_list
    def adj_list(self, value):
        self._adj_list = value

    def convert_list_to_matrix(self) -> Matrix:
        vertices = list(self.adj_list.keys())
        num_vertices = len(vertices)
        rows = [Vector(*[0] * num_vertices) for _ in range(num_vertices)]
        self.adj_matrix = Matrix(*rows)
        
        for i, vertex in enumerate(vertices):
            for neighbor in self.adj_list[vertex]:
                j = vertices.index(neighbor)
                self.adj_matrix.__setitem__((i,j), 1)
        
        return self.adj_matrix


    
    @property
    def adj_matrix(self) -> Matrix:
        return self.adj_matrix
    def adj_matrix(self, value):
        self.adj_matrix = value

    def convert_matrix_to_list(self) -> dict:
        num_vertices = len(self.adj_matrix)
        self.adj_list = {}

        for i in range(num_vertices):
            self.adj_list[i] = []
            for j in range(num_vertices):
                if self.adj_matrix[i][j] != 0: 
                    self.adj_list[i].append(j)

        return self.adj_list
    
    @property
    def order(self):
        if self.adj_list:
            return len(self.adj_list)
        elif self.adj_matrix:
            return len(self.adj_matrix)


    