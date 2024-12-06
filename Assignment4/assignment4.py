# Jevin Wilson
# Assignment 4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Lab1.graph_JW import Graph
import numpy as np

########## (i) ##########
def compute_laplacian_from_matrix(adj_matrix):
    """Calculates the Laplacian matrix from a hardcoded adjacency matrix."""
    # Compute the degree matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - adj_matrix
    return laplacian_matrix

# create adjacency matrix for Kn
def generate_kn_matrix(n):
    adj_matrix = []
    for i in range(n):
        # empty list for the current row
        row = []  
        for j in range(n):
            if i != j:
                # Add 1 for edges between different vertices
                row.append(1)  
            else:
                # Add 0 for diagonal elements
                row.append(0)  
        # Add the completed row to the matrix
        adj_matrix.append(row)  
    # Convert list of lists to a numpy array
    return np.array(adj_matrix)  

# calculate eigenvalues
def compute_eigenvalues(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.sort(eigenvalues)

# uses kirchhoff's matrix tree to count the number of spanning trees.
def count_spanning_trees(laplacian):
    # Remove the first row and column to compute the cofactor
    minor = np.delete(np.delete(laplacian, 0, axis=0), 0, axis=1)
    # Compute the determinant of the minor
    num_spanning_trees = round(abs(np.linalg.det(minor)))  # Take the absolute value
    return num_spanning_trees

def do_Kn():
    n = int(input("n (vertices): "))
    adj_matrix = generate_kn_matrix(n)
    print("\nAdjacency Matrix:")
    print(adj_matrix)

    # Compute and display Laplacian matrix and results
    results(adj_matrix) 

########## (ii) ##########
def generate_pn_matrix(n):
    adj_matrix = np.zeros((n, n))
    # add edges between consecutive vertices
    for i in range(n - 1):
        adj_matrix[i, i + 1] = adj_matrix[i + 1, i] = 1
    return adj_matrix

def do_Pn():
    n = int(input("n (vertices): "))
    adj_matrix = generate_pn_matrix(n)
    print("\nAdjacency Matrix:")
    print(adj_matrix)

    # Compute and display results
    results(adj_matrix)


def results(adj_matrix):
    # Compute Laplacian matrix
    laplacian = compute_laplacian_from_matrix(adj_matrix)
    print("\nLaplacian Matrix:")
    print(laplacian)

    # Compute eigenvalues of adjacency matrix
    adjacency_eigenvalues = compute_eigenvalues(adj_matrix)
    print("\nEigenvalues (adjacency matrix):")
    print(adjacency_eigenvalues)
    # Compute eigenvalues of Laplacian matrix
    laplacian_eigenvalues = compute_eigenvalues(laplacian)
    print("Eigenvalues (Laplacian matrix):")
    print(laplacian_eigenvalues)

    # Count the number of spanning trees
    num_trees = count_spanning_trees(laplacian)
    print("\nNumber of Spanning Trees:")
    print(num_trees)

def main():
    while True:
        print("Assignmetm number:")
        print("i   - Complete Graph (Kn)")
        print("ii  - Path Graph (Pn)")
        print("q   - Quit")

        choice = input("Number: ").strip().lower()
        if choice == "i":
            do_Kn()
        elif choice == "ii":
            do_Pn()
        elif choice == "q":
            break


if __name__ == "__main__":
    main()