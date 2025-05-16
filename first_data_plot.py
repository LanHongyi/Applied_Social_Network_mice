import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

def load_and_prepare_graph(df1_path, df2_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    nodes1 = pd.unique(df1[['from', 'to']].values.ravel())
    nodes2 = pd.unique(df2[['from', 'to']].values.ravel())
    common_nodes = set(nodes1) & set(nodes2)

    df1_filtered = df1[df1['from'].isin(common_nodes) & df1['to'].isin(common_nodes)]
    df2_filtered = df2[df2['from'].isin(common_nodes) & df2['to'].isin(common_nodes)]

    edge_weights1 = {(row['from'], row['to']): row['weight'] for _, row in df1_filtered.iterrows()}
    edge_weights2 = {(row['from'], row['to']): row['weight'] for _, row in df2_filtered.iterrows()}

    common_edges = set(edge_weights1.keys()) & set(edge_weights2.keys())

    G = nx.Graph()
    for u, v in common_edges:
        G.add_edge(u, v, weight1=edge_weights1[(u, v)], weight2=edge_weights2[(u, v)])

    return G

# File paths
g1_path = '/Users/steven/Desktop/ASN/rdata.csv'
g2_path = '/Users/steven/Desktop/ASN/filtered_weights_q25.csv'
shared_path = '/Users/steven/Desktop/ASN/sdata.csv'

# Load graphs
G1 = load_and_prepare_graph(g1_path, shared_path)
G2 = load_and_prepare_graph(g2_path, shared_path)

# Combine all weights for color normalization
all_weights1 = [G1[u][v]['weight1'] for u, v in G1.edges()] + [G2[u][v]['weight1'] for u, v in G2.edges()]
norm = mcolors.Normalize(vmin=min(all_weights1), vmax=max(all_weights1))
cmap = cm.plasma

# Fix node positions across both graphs
all_nodes = set(G1.nodes()).union(G2.nodes())
layout_graph = nx.Graph()
layout_graph.add_nodes_from(all_nodes)
layout_graph.add_edges_from(G1.edges())  # Helps shape layout
fixed_pos = nx.spring_layout(layout_graph, seed=42)

# Function to draw graph
def draw_graph(ax, G, title, pos):
    weights1 = [G[u][v]['weight1'] for u, v in G.edges()]
    weights2 = [G[u][v]['weight2'] for u, v in G.edges()]

    edge_colors = [cmap(norm(w)) for w in weights1]
    min_w2, max_w2 = min(weights2), max(weights2)
    edge_widths = [1 + 4 * ((w - min_w2) / (max_w2 - min_w2)) if max_w2 > min_w2 else 1 for w in weights2]

    nx.draw(G, pos,
            with_labels=False,
            edge_color=edge_colors,
            width=edge_widths,
            node_color='lightgray',
            node_size=70,
            ax=ax)
    ax.set_title(title)
    ax.axis('off')

# Create figure and axes
fig = plt.figure(figsize=(22, 20))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], wspace=0.1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
cax = fig.add_axes([0.92, 0.35, 0.01, 0.3])  # [left, bottom, width, height] for colorbar

# Draw both graphs
draw_graph(ax1, G1, "", fixed_pos)
draw_graph(ax2, G2, "", fixed_pos)

# Shared colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cax, label='Scaled Genetic Similarity')

# Adjust layout to fit everything
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
plt.show()
