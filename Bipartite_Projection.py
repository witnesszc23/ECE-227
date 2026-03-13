import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

# ==========================================
# Data Loading & Initialization
# ==========================================
nodes_df = pd.read_csv('Downloads/nodes.csv')
edges_df = pd.read_csv('Downloads/edges.csv')

source_col = edges_df.columns[0]
target_col = edges_df.columns[1]

print("Twin Cities Corporate Network Analysis")
print("=" * 45, "\n")

# ==========================================
# Part 1: Bipartite Projection (CEO Strong Ties)
# ==========================================
B = nx.Graph()

# Define node sets: 0-24 are CEOs, 25-39 are clubs
ceo_nodes = range(25)
club_nodes = range(25, 40)

B.add_nodes_from(ceo_nodes, bipartite=0)
B.add_nodes_from(club_nodes, bipartite=1)
B.add_edges_from(zip(edges_df[source_col], edges_df[target_col]))

# Create weighted projection for CEOs based on shared clubs
ceo_projected_graph = bipartite.weighted_projected_graph(B, ceo_nodes)

print("--- Part 1: Strongest CEO Ties ---")
edges_with_weights = list(ceo_projected_graph.edges(data=True))
sorted_edges = sorted(edges_with_weights, key=lambda x: x[2]['weight'], reverse=True)

print("Top 5 most connected CEO pairs:")
for u, v, data in sorted_edges[:5]:
    # Adding 1 to align with 1-based indexing for reporting
    print(f"  CEO Node {u+1} <---> CEO Node {v+1} | Shared clubs: {data['weight']}")
print("\n")

# ==========================================
# Part 2: Louvain Community Detection
# ==========================================
# Build the general graph directly from the edge list
G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col)

# Run Louvain community detection
communities = nx.community.louvain_communities(G)

# Calculate community assignment and degree for each node
node_stats = []
sorted_communities = sorted(communities, key=len, reverse=True)

for comm_id, comm_nodes in enumerate(sorted_communities):
    for node in comm_nodes:
        node_stats.append({
            'node_id': node, 
            'Community': comm_id + 1,
            'Degree': G.degree(node)
        })

stats_df = pd.DataFrame(node_stats)

# Merge network metrics with original node attributes
node_id_col = nodes_df.columns[0] 
final_df = nodes_df.merge(stats_df, left_on=node_id_col, right_on='node_id', how='inner')

print("--- Part 2: Core Nodes by Community ---")
for i in range(1, 4):
    print(f"Community #{i}: Top 5 Nodes by Degree")
    top5 = final_df[final_df['Community'] == i].nlargest(5, 'Degree')
    
    for _, row in top5.iterrows():
        # Fallback to node_id if a secondary label/name column doesn't exist
        name_or_label = row.iloc[1] if len(row) > 1 else row['node_id']
        print(f"  Name/Label: {name_or_label} | Degree: {row['Degree']}")
    print("-" * 45)

output_filename = 'Community_Analysis_Result.csv'
final_df.to_csv(output_filename, index=False)
print(f"✅ Analysis complete. Results saved to '{output_filename}'\n")

# ==========================================
# Part 3: Network Visualization
# ==========================================
print("--- Part 3: Generating Visualization ---")

# Create a relabeled copy of B for visualization (1-based indexing)
mapping = {node: node + 1 for node in B.nodes()}
B_vis = nx.relabel_nodes(B, mapping)

# Shift the sets for visualization
ceos_shifted = [n + 1 for n in ceo_nodes]
clubs_shifted = [n + 1 for n in club_nodes]

# Set up the plot
plt.figure(figsize=(12, 10), dpi=300)

# Bipartite layout with shifted CEO nodes on the left
pos = nx.bipartite_layout(B_vis, ceos_shifted)

# Draw CEOs (Left side)
nx.draw_networkx_nodes(B_vis, pos, nodelist=ceos_shifted, 
                       node_color='#87CEEB', edgecolors='black', 
                       node_size=600, label='CEOs (Nodes 1-25)')

# Draw Clubs (Right side)
nx.draw_networkx_nodes(B_vis, pos, nodelist=clubs_shifted, 
                       node_color='#FF7F50', edgecolors='black', 
                       node_size=800, label='Organizations (Nodes 26-40)')

# Draw edges and labels
nx.draw_networkx_edges(B_vis, pos, alpha=0.3, edge_color='gray')
nx.draw_networkx_labels(B_vis, pos, font_size=9, font_weight='bold')

# Styling
plt.title("Bipartite Network of CEO Club Memberships (Twin Cities, 1985)", fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
plt.axis('off')

# Save and Show
plt.tight_layout()
plt.savefig('Downloads/bipartite_network.png', format='png', bbox_inches='tight')
print("✅ Visualization complete! Saved as 'bipartite_network.png'")

plt.show()
