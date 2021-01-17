import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

node_color = 'red'
node_border_color = 'black'
node_border_width = .6
edge_color = 'black'

n = 5
num = 3
radius = np.linspace(0, 1, num=num)


def draw(G, pos, ax):
    nodes1 = nx.draw_networkx_nodes(G, pos=pos, node_size=500,
                                    node_color='white', ax=ax)
    nodes1.set_edgecolor(node_border_color)
    nodes1.set_linewidth(node_border_width)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.8,
                           ax=ax)
    z = zip(G.nodes, list(range(0, len(G.nodes))))
    idx = dict(z)
    nx.draw_networkx_labels(G, pos, idx, font_size=9, ax=ax)
    # ax.axis('off')


fig, axs = plt.subplots(num)

G = nx.generators.geometric.random_geometric_graph(n, radius[0])
pos = nx.get_node_attributes(G, 'pos')
draw(G, pos, axs[0])

for i in range(1, len(radius)):
    G = nx.generators.geometric.random_geometric_graph(n, radius[i], pos=pos)
    draw(G, pos, axs[i])
    # axs[i].axis('off')
plt.show()
