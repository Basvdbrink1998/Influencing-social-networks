import networkx as nx
import matplotlib.pyplot as plt

node_color = 'red'
node_border_color = 'black'
node_border_width = .6
edge_color = 'black'


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
    ax.axis('off')


fig, ax = plt.subplots()
G = nx.house_graph()
pos = nx.spring_layout(G)
idx = draw(G, pos, ax)
print(nx.adjacency_matrix(G).todense())

plt.show()
