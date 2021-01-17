import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

node_color= 'red'
node_border_color = 'black'
node_border_width = .6
edge_color = 'black'
N = 10
num_graphs = 6
N_columns = 3
N_rows = 2
P = np.linspace(0.0, 1.0, num=num_graphs)
print(P)


def draw(G, pos, ax):
    nodes1 = nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, ax=ax)
    nodes1.set_edgecolor(node_border_color)
    nodes1.set_linewidth(node_border_width)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.8,
                           ax=ax)
    ax.axis('off')

    return ax


fig, axs = plt.subplots(N_columns, N_rows)

G = nx.fast_gnp_random_graph(N, P[0], seed=0)
pos = nx.spring_layout(G)
c = 0
for i in range(N_columns):
    for j in range(N_rows):

        G = nx.fast_gnp_random_graph(N, P[c], seed=0)
        # pos = nx.random_layout(G)
        # pos = nx.spring_layout(G)
        draw(G, pos, axs[i,j])
        txt="I need the caption to be present a little below X-axis"
        axs[i,j].text(0.5,-0.3, "P = " + str(round(P[c],1)), size=12, ha="center",
             transform=axs[i,j].transAxes)
        # axs[0].set_title('Network with high degree of clustering')
        c += 1

plt.subplots_adjust(hspace=0.3)
plt.show()
