import numpy as np
from matplotlib import cm
import networkx as nx
import copy
from Influence.sdnet.utils import make_dist_matrix, euclidean_dist
from sdnet import SDA


def make_graph(A, C, belief_name, media_locs=[]):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    for i in range(len(A)):
        G.nodes[i][belief_name] = C[i]
        G.add_weighted_edges_from([(i, i, 1)])
    if media_locs:
        # print("media_locs", media_locs)
        for idx in media_locs:
            media_name = list(G.nodes.keys())[idx]
            G.nodes[media_name]['color'] = 'magenta'
            neighbors = copy.deepcopy(G.adj[media_name])
            for i in neighbors:
                G.remove_edge(media_name, i)
    return G


def viz_space(ax, X, labels=None, edgecolors='#000000', linewidths=.6, **kwds):
    x, y = X[:, 0], X[:, 1]
    if labels is not None:
        labels = [cm.Set1(x) for x in labels]
    ax.scatter(x, y, c=labels, edgecolors=edgecolors, linewidths=linewidths,
               **kwds)
    ax.set_axisbelow(True)
    ax.grid(zorder=0, linestyle='--')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax


def viz_degseq(ax, degseq):
    ax.hist(degseq, color=cm.Set1(1), edgecolor='#000000', linewidth=1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax


def viz_sda_graph(ax, G, pos=None, with_labels=False, node_color=None,
                  node_border_color='#000000', node_border_width=.6,
                  edge_color='#000000', nodes_kws=None, edges_kws=None,
                  size_scaler=lambda x: np.sqrt(x)*2):
    if nodes_kws is None:
        nodes_kws = {}
    if edges_kws is None:
        edges_kws = {}
    if node_color is None:
        node_color = np.array([n['color'] for n in G.nodes.values()])
    if pos is None:
        pos = nx.drawing.kamada_kawai_layout(G)
    nodes = \
        nx.draw_networkx_nodes(G, pos, node_color=node_color,
                               with_labels=with_labels, ax=ax, node_size=size_scaler, **nodes_kws)
    nodes.set_edgecolor(node_border_color)
    nodes.set_linewidth(node_border_width)
    # nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.2,
                           ax=ax, **edges_kws)
    return ax


def plot_results(ax, results, node_names, show_legend=False,
                 remove_dupes=False, colors=None, idx=0):
    """
    """
    for i in range(len(results)):
        if remove_dupes:
            if not results[i, -1] == results[i, 0]:
                ax.plot(results[i], label=node_names[i], color=colors[node_names[i]])
        else:
            ax.plot(results[i], label=node_names[i])
    if show_legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


def insert_node(beliefs, b):
    """
        Appends a b vector to a belief matrix.
    """
    beliefs = np.append(beliefs, [b], axis=0)
    return beliefs, len(beliefs) - 1


def score(G, b, dist=euclidean_dist, belief_name='belief'):
    """
        Returns the distance according to a given distance function between
        the average belief of a given graph and a given belief vector.
    """
    gb = list(nx.get_node_attributes(G, belief_name).values())
    mean = np.mean(gb, axis=0)
    s = np.sum(abs(mean-b))
    return s


def generate_network(N, NDIM, clusters, degseq, alpha, K,  belief_name,
                     media_beliefs=None, media_changes=None):
    """
        Returns a network generated using the Social Distance Attachment model
        and inserts influencer node(s) if media beliefs are given.
    """
    media_locs = None
    if isinstance(media_beliefs, np.ndarray):
        clusters = np.append(clusters, media_beliefs, axis=0)
        media_locs = list(range(len(clusters)-len(media_beliefs),
                                len(clusters)))
        N += len(media_beliefs)
    D = make_dist_matrix(clusters, euclidean_dist,
                         symmetric=True).astype(np.float32)
    sda = SDA.from_dist_matrix(D, alpha=alpha, k=K, directed=False)
    A = sda.adjacency_matrix(sparse=False)
    G = make_graph(A, clusters, belief_name, media_locs=media_locs)
    return G, media_locs
