import networkx as nx
import matplotlib.pyplot as plt

"""
    Plots a graph with 2 degrees of clustering for Figure 2.2.
"""

node_color = 'red'
node_border_color = 'black'
node_border_width = .6
edge_color = 'black'

figsize = 15
sizes = [5, 5, 5]

probs1 = [[1, 0.05, 0.05],
          [0.05, 1, 0.05],
          [0.05, 0.05, 1]]

probs2 = [[1, 0.5, 0.5],
          [0.5, 1, 0.5],
          [0.5, 0.5, 1]]


def draw(G, pos, ax):
    #  Plots a graph.
    nodes1 = nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, ax=ax)
    nodes1.set_edgecolor(node_border_color)
    nodes1.set_linewidth(node_border_width)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.8,
                           ax=ax)
    ax.axis('off')

    return ax


fig, axs = plt.subplots(1, 2, figsize=(figsize, figsize))

G1 = nx.stochastic_block_model(sizes, probs1, seed=0)
pos = nx.kamada_kawai_layout(G1)
draw(G1, pos, axs[0])
axs[0].set_title('Network with high degree of clustering')

G2 = nx.stochastic_block_model(sizes, probs2, seed=0)
draw(G2, pos, axs[1])
axs[1].set_title('Network with low degree of clustering')

plt.show()
