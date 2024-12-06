import heapq
import numpy as np

romania_graph = {
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Zerind': [('Oradea', 71), ('Arad', 75)],
    'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Sibiu': [('Oradea', 151), ('Arad', 140), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Craiova': [('Rimnicu Vilcea', 146), ('Pitesti', 138), ('Drobeta', 120)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
    'Giurgiu': [('Bucharest', 90)],
    'Neamt': [('Iasi', 87)],
    'Iasi': [('Neamt', 87), ('Vaslui', 92)],
    'Vaslui': [('Iasi', 92), ('Urziceni', 142)],
    'Urziceni': [('Vaslui', 142), ('Bucharest', 85), ('Hirsova', 98)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Eforie': [('Hirsova', 86)]
}

#### Task 1 ####
def dijkstra(graph, start, dest):
    # set distances to infinity
    distances = {node: float('inf') for node in graph}  
    # start node distance is 0
    distances[start] = 0  
    # (distance, node)
    priority_queue = [(0, start)]  
    # track the path
    previous_nodes = {node: None for node in graph}  
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # reach the destination, stop
        if current_node == dest:
            break
        
        # skip if the nodes distance is larger
        if current_distance > distances[current_node]:
            continue
        
        # update neighbor distance
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # shortest path by backtracking
    path = []
    current = dest
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]
    
    return distances[dest], path

"""
hueristics for distance FROM Bucharest
just chose any distance smaller than the actual distance from Bucharest
"""
heuristic = {
    'Oradea': 420,          # <= 429
    'Zerind': 300,          # <= 493
    'Arad': 360,            # <= 418
    'Timisoara': 100,       # <= 536
    'Lugoj': 404,           # <= 504
    'Mehadia': 202,         # <= 434
    'Drobeta': 321,         # <= 359
    'Sibiu': 123,           # <= 278
    'Rimnicu Vilcea': 100,  # <= 198
    'Craiova': 189,         # <= 239
    'Fagaras': 111,         # <= 211
    'Pitesti': 99,          # <= 101
    'Bucharest': 0,
    'Giurgiu': 80,          # <= 90
    'Neamt': 220,           # <= 406
    'Iasi': 319,            # <= 319
    'Vaslui': 180,          # <= 227
    'Urziceni': 69,         # <= 85
    'Hirsova': 170,         # <= 183
    'Eforie': 268           # <= 269
}

"""
# from worksheet
heuristic = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}

"""

def a_star(graph, start, goal, heuristic):
    # Initialize distances and priority queue
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # (f(n), node), f(n) = g(n) + h(n)
    priority_queue = [(heuristic[start], start)]  
    previous_nodes = {node: None for node in graph}
    
    while priority_queue:
        current_f, current_node = heapq.heappop(priority_queue)
        
        # reach destination stop
        if current_node == goal:
            break
        
        # explore neighbors
        for neighbor, weight in graph[current_node]:
            # g(n)
            g = distances[current_node] + weight 
            # f(n) = g(n) + h(n)
            f = g + heuristic[neighbor]  

            # update if a shorter path is found
            if g < distances[neighbor]:  
                distances[neighbor] = g
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (f, neighbor))
    
    # reconstruct the path
    path = []
    current = goal
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]
    
    return distances[goal], path

def floyd_warshall(graph):
    # extract nodes and create mapping
    nodes = list(graph.keys())
    n = len(nodes)
    node_index = {}
    for i, node in enumerate(nodes):
        node_index[node] = i
    
    # initialize distance matrix
    dist = np.full((n, n), float('inf'))
    np.fill_diagonal(dist, 0)  # Distance to self is 0
    
    # populate initial distances based on the graph
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            dist[node_index[node], node_index[neighbor]] = weight
    
    # floyd warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return dist, nodes


if __name__ == "__main__":
    #### Task 1 ####
    start = 'Arad'
    destination = 'Bucharest'
    shortest_distance, shortest_path = dijkstra(romania_graph, start, destination)
    print(f"Shortest distance from {start} to {destination}: {shortest_distance}")
    # Shortest distance from Arad to Bucharest: 418
    
    #### Task 2 ####
    print(f"Path: {' -> '.join(shortest_path)}")
    # Path: Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest

    #### Task 3 ####
    shortest_distance_a_star, shortest_path_a_star = a_star(romania_graph, start, destination, heuristic)
    
    #### Task 4 ####
    print("\nUsing A* Search:")
    print(f"Shortest distance from {start} to {destination}: {shortest_distance_a_star}")

    #### Task 5 ####
    print(f"Path: {' -> '.join(shortest_path_a_star)}")

    #### Task 6 ####
    dist_matrix, nodes = floyd_warshall(romania_graph)
    
    #### Task 7 ####
    print("\nFloyd-Warshall Distance Matrix:")
    print("   " + "   ".join(nodes))
    for i, row in enumerate(dist_matrix):
        row_values = []
        for cell in row:
            if cell < float('inf'):
                # format finite distances
                row_values.append(f"{int(cell):>3}")  
            else:
                # format infinite distances
                row_values.append("inf")  
        # combine values into a single string
        row_string = " ".join(row_values)  
        # print the node and the row values
        print(f"{nodes[i]:>9} {row_string}")  

#### Task 9 ####
"""
Of the three algorithms, I believe that the floyd warshall is the most appropriate for solving this type
of problem. Even though it is not the most efficient it makes up for it with the extra information that it
generates compared to the other two. In terms of solving distance to and from a destination I assume
that seeing all the possible distances in one place is very helpful to analysts. The other two algorithms are 
faster but in the context of this type of problem I don't think efficiency at the computer level is valued
because the time saved in human time is probably negligible. I am aware that with extremely large cases
the efficiency will become noticeable in human time but I can't think of a scenario where a data set 
solving for distances between locations would be large enough to warrant the use of a more efficient 
algorithm, I could be wrong though. In the context of this problem I think you can ignore efficiency because
I view efficiency as a quality of life feature that in large programs is important for users in a consumer
setting. In this case the user is the analyst and the program is the tool and as long as they get the 
correct data before their deadline, how fast they obtain it is not important.
"""