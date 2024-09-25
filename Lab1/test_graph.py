import unittest
from graph_JW import Graph
from vector import Vector
from matrix import Matrix

class TestGraph(unittest.TestCase):
    def test_valid_adj_list(self):
        # Test initializing the graph via a valid adjacency list
        adj_list = {
            1: Vector(2, 3, 4),
            2: Vector(1, 3, 4),
            3: Vector(1, 2, 4),
            4: Vector(1, 2, 3)
        }
        
        graph = Graph(adj_list)
        self.assertEqual(adj_list, graph.adj_list)
        
        # Test the order size
        self.assertEqual(len(list(adj_list.keys())), graph.order)
        
        # Test the converted matrix
        adj_matrix = Matrix(
            Vector(0, 1, 1, 1),
            Vector(1, 0, 1, 1),
            Vector(1, 1, 0, 1),
            Vector(1, 1, 1, 0)
        )
        self.assertEqual(adj_matrix, graph.adj_matrix)
        
        # Test the size
        self.assertEqual(graph.order * (graph.order - 1) / 2, graph.size)         

if __name__ == '__main__':
    unittest.main()