import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

node_color = 'red'
node_border_color = 'black'
node_border_width = .6
edge_color = 'black'

def round_dict(dic, n):
    for key in dic:
        dic[key] = round(dic[key], n)
    return dic

def draw(G, pos, centrality, ax):
    # nodes1 = nx.draw_networkx_nodes(G, pos=pos, node_color=list(centrality.values()), ax=ax)
    nodes1 = nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='white', ax=ax)
    nodes1.set_edgecolor(node_border_color)
    nodes1.set_linewidth(node_border_width)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.8,
                           ax=ax)
    nx.draw_networkx_labels(G, pos, round_dict(centrality, 2), font_size=9, ax=ax)
    ax.axis('off')
    return ax

fig, axs = plt.subplots(1, 2)
G = nx.generators.social.florentine_families_graph()
pos = nx.spring_layout(G)
evc = nx.eigenvector_centrality_numpy(G)
cc = nx.degree_centrality(G)
draw(G, pos, evc, axs[0])
axs[0].set_title('Eigenvector centrality')

draw(G, pos, cc, axs[1])
axs[1].set_title('Degree centrality')

plt.show()
