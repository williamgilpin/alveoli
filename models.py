
import numpy as np

import networkx as nx

# from networkx.drawing.nx_agraph import graphviz_layout

def plot_circle(n, nodecolors=None, ax=None):
    """Plot nodes as a circle"""
    if nodecolors is None:
        nodecolors = np.ones(n)
    pos = np.zeros((n, 2))
    for i in range(n):
        pos[i, 0] = np.cos(2 * np.pi * i / n)
        pos[i, 1] = np.sin(2 * np.pi * i / n)
    plt.scatter(pos[:, 0], pos[:, 1], c=nodecolors, cmap="RdBu", vmin=0, vmax=1, ax=ax)
    return pos


def tree_coupling_matrix(n):
    """
    Create a tree-like adjacency matrix for n nodes. The first node is the root.

    Args:
        n (int): number of nodes
        
    Returns:
        matrix (ndarray): n x n coupling matrix

    """
    if n < 1:
        return "n should be greater than or equal to 1"
    
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n), dtype=int)
    
    # Create tree-like connections
    for i in range(1, n):
        matrix[i, (i-1)//2] = 1
        matrix[(i-1)//2, i] = 1
    
    return matrix

def plot_tree(n, nodecolors=None, treestyle="full", pos=None, random_state=None, ax=None, **kwargs):
    """
    Given a balanced tree coupling matrix, plot the tree with matplotlib
    """
    # Get the adjacency matrix
    if treestyle == "full":
        matrix = tree_coupling_matrix(n)
    elif treestyle == "leaf":
        matrix = hyperbolic_coupling_matrix(n)
    else:
        matrix = tree_coupling_matrix(n)
    
    # Create a graph from the adjacency matrix
    G = nx.Graph(matrix)
    
    if pos is None:
        # Create a tree layout for our graph
        # pos = nx.tree_layout(G)
        pos = nx.spring_layout(G, iterations=100, seed=random_state)
        # pos = graphviz_layout(G, prog="twopi")

    if nodecolors is None:
        nodecolors = np.random.random(n)

    nx.draw(G, pos, with_labels=False, node_color=nodecolors, ax=ax, **kwargs)
    
    # Draw the graph
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue")
    
    # # Show the plot
    # plt.show()

    return pos

def weighted_shortest_distance(n):
    depth = int(np.log2(n))
    total_nodes = 2**depth - 1
    leaf_indices = [total_nodes + i for i in range(n)]

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            xi, xj = leaf_indices[i], leaf_indices[j]
            distance = 0

            while xi != xj:
                if xi > xj:
                    xi = (xi - 1) // 2
                else:
                    xj = (xj - 1) // 2
                distance += 2
            
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix

def hyperbolic_coupling_matrix(n):
    a = weighted_shortest_distance(n)
    np.fill_diagonal(a, np.inf)
    a = 1 / a
    np.fill_diagonal(a, 0)
    return a / np.sum(a)
