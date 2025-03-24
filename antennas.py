import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import dimod
from qdeepsdk import QDeepHybridSolver

# Create the graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 7)])

# Maximum Independent Set (MIS) QUBO formulation:
# For each node i, assign binary variable x_i = 1 if node i is in the independent set.
# Objective: maximize sum_i x_i subject to x_i + x_j <= 1 for each edge (i, j)
# This can be formulated as minimizing:
#    Q(x) = - sum_i x_i + A * sum_{(i,j) in E} x_i * x_j
# where A is a penalty constant (set A = 2 here).

A = 2

# Build the QUBO dictionary
Q = {}
nodes = list(G.nodes())
for i in nodes:
    # Linear term for each node: coefficient -1
    Q[(i, i)] = -1

for i, j in G.edges():
    # For each edge, add the penalty term A * x_i * x_j
    Q[(i, j)] = Q.get((i, j), 0) + A

# Map node labels to indices (0 to n-1) because our solver expects a matrix index
mapping = {node: idx for idx, node in enumerate(nodes)}
n = len(nodes)
matrix = np.zeros((n, n))
for (i, j), coeff in Q.items():
    idx_i = mapping[i]
    idx_j = mapping[j]
    matrix[idx_i, idx_j] = coeff

# Initialize the QDeepHybridSolver
solver = QDeepHybridSolver()
solver.token = "your-auth-token-here"  # Replace with your actual token

# Solve the QUBO by passing the NumPy matrix to the solver
result = solver.solve(matrix)
sample = result.get('sample')

# Convert the sample (with keys 0,..., n-1) back to the original node labels
independent_set = [nodes[i] for i in range(n) if sample.get(i, 0) == 1]

print("Maximum independent set found (MIS) has size", len(independent_set))
print("MIS:", independent_set)

# Visualize the graph with the independent set highlighted
pos = nx.spring_layout(G)
plt.figure()
node_colors = ['red' if node in independent_set else 'blue' for node in G.nodes()]
nx.draw_networkx(G, pos=pos, with_labels=True, node_color=node_colors)
plt.title("Maximum Independent Set")
plt.savefig("mis.png")
print("Plot saved as mis.png")
