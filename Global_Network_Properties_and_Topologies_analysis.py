import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. Loading and Cleaning the GML file
file_path = 'Downloads/network.gml'

def load_cleaned_gml(path):
    """Filters out problematic attributes like _pos that cause tokenization errors."""
    cleaned_lines = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove lines containing _pos which causes the 'cannot tokenize' error
                if '_pos' not in line:
                    cleaned_lines.append(line)
        return nx.parse_gml("".join(cleaned_lines))
    except Exception as e:
        print(f"Error parsing GML: {e}")
        return nx.Graph()

G_real = load_cleaned_gml(file_path)

# Verify if the graph is loaded correctly
if G_real.number_of_nodes() == 0:
    print("Warning: Graph is empty. Please check the GML file content.")
    # Fallback to manual edge list provided in your source if file fails
    edges = [(0,25),(0,26),(0,27),(1,25),(1,28),(2,29),(2,25),(2,30),(3,25),(3,31),(3,32),(4,29),(4,25),(4,32),(5,25),(5,33),(5,34),(5,35),(6,33),(6,27),(6,34),(7,36),(7,33),(7,37),(7,34),(8,25),(8,38),(9,29),(9,25),(9,38),(10,33),(10,27),(11,25),(11,33),(11,26),(11,38),(12,29),(12,25),(12,33),(12,35),(12,28),(12,31),(12,30),(13,29),(13,25),(13,39),(13,31),(13,30),(14,29),(14,25),(14,39),(14,37),(14,32),(15,29),(15,25),(15,26),(15,35),(15,28),(15,30),(16,33),(16,38),(16,28),(16,31),(16,30),(17,36),(17,25),(17,33),(17,27),(17,38),(18,29),(18,25),(18,33),(18,35),(18,30),(19,25),(19,33),(19,37),(20,25),(20,37),(20,30),(21,29),(21,25),(21,39),(21,30),(22,36),(22,25),(22,33),(22,39),(22,30),(23,29),(23,25),(23,31),(24,29),(24,25),(24,28)]
    G_real.add_edges_from(edges)

G_real = G_real.to_undirected()
n = G_real.number_of_nodes()
m = G_real.number_of_edges()
avg_degree = 2 * m / n

def get_network_stats(G, name):
    """Calculates core topological metrics for the network"""
    if not nx.is_connected(G):
        G_cc = G.subgraph(max(nx.connected_components(G), key=len))
    else:
        G_cc = G
        
    return {
        'Model': name,
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Avg Clustering': round(nx.average_clustering(G), 4),
        'Avg Path Length': round(nx.average_shortest_path_length(G_cc), 4),
        'Diameter': nx.diameter(G_cc)
    }

# 2. Generate Baseline Models
p = m / (n * (n - 1) / 2)
G_er = nx.erdos_renyi_graph(n, p, seed=42)
k = int(round(avg_degree))
if k % 2 != 0: k += 1
G_ws = nx.watts_strogatz_graph(n, k, p=0.1, seed=42)
m_ba = max(1, int(round(avg_degree / 2)))
G_ba = nx.barabasi_albert_graph(n, m_ba, seed=42)

# 3. Comparison Table
models = [(G_real, 'Real (CEO Club)'), (G_er, 'Erdös-Renyi'), (G_ws, 'Watts-Strogatz'), (G_ba, 'Barabasi-Albert')]
results = [get_network_stats(g, label) for g, label in models]
print("\n--- Network Model Comparison Table ---")
print(pd.DataFrame(results).to_string(index=False))

# 4. Visualization: Degree Distribution
plt.figure(figsize=(10, 6))
for g, label in models:
    degrees = [d for n, d in g.degree()]
    counts = np.bincount(degrees)
    plt.plot(range(len(counts)), counts, label=label, marker='o', alpha=0.7)

plt.title("Degree Distribution Comparison")
plt.xlabel("Degree (k)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Downloads/comparison_plot.png")
print("\nResults saved to 'comparison_plot.png'")