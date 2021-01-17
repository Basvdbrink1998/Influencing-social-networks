#!/usr/bin/env python

"""
    enviroment.py:
"""

from Influence import Learning_SDC as ls
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Influence import DeGroot as dg
from sdnet.utils import make_dist_matrix, euclidean_dist
from Influence.sdnet import SDA


class SDA_enviroment:
    """
        Class which performes different experiments to evaluate different
        influencing strategies of networkks generated by the SDA model.
    """
    def __init__(self, Belief_name='belief', weight_name='weight'):
        """
            Sets the name of the attribute where the beliefs of nodes and
            weights of connections are saved.
        """
        self.BN = Belief_name
        self.WN = weight_name

    def init_features(self, N_nodes, NDIM, CENTERS, K, ALPHA):
        """
            Initialises the different parameters of the SDA model.
        """
        self.N = N_nodes
        self.NDIM = NDIM
        self.NMED = 0
        self.CENTERS = CENTERS
        self.K = K
        self.ALPHA = ALPHA
        self.step_n = 0

        self.graphs = []
        self.media = []
        self.media_locs = []
        self.media_changes = []
        self.clusters = None
        self.cluster_labs = None

    def generate_nodes(self):
        """
            Generates nodes which are placed in a belief space.
        """
        self.clusters = ls.generate_clusters(self.N, self.NDIM, self.CENTERS)
        return self.clusters

    def insert_nodes(self, nodes, categories=None):
        """
            Inserts given nodes in the belief space.
        """
        self.clusters = nodes
        self.cluster_labs = categories

    def connect_nodes(self, clusters):
        """
            Creates connections between placed nodes according to the Social
            Distance Attachment (SDA) model and retruns the resulting graph.
        """
        D = make_dist_matrix(clusters, euclidean_dist,
                             symmetric=True).astype(np.float32)
        sda = SDA.from_dist_matrix(D, alpha=self.ALPHA, k=self.K,
                                   directed=False)
        A = sda.adjacency_matrix(sparse=False)
        G = ls.make_graph(A, clusters, self.BN, media_locs=self.media_locs)
        return G

    def connect_graph(self, clusters=[]):
        """
            Connects the nodes in given clusters. If no clusters are given,
            the saved clusters are used.
        """
        if len(clusters) == 0:
            G = self.connect_nodes(self.clusters)
        else:
            G = self.connect_nodes(clusters)
        self.graphs = [G]
        return G

    def generate_new_graph(self):
        """
            Resets the connections in the currently saved graph.
        """
        self.generate_nodes()
        self.graphs = [self.connect_nodes(self.clusters)]
        return self.graphs[0]

    def insert_media(self, belief, changes=None):
        """
            Inserts a given influencer node in the clusters of nodes and save
            the location and future changes.
        """
        self.clusters = np.vstack([self.clusters, belief])
        self.media_locs.append(len(self.clusters)-1)
        self.media_changes.append(changes)

    def step(self, node_idx=0, action=0, dynamic=True):
        """
            Performes 1 iteration where the following steps are performed.

            1. Updates the location of the influencer node(s).

            2. Updates the connections between all the nodes.

            3. Updates the location of all the nodes except the influencer
            nodes in the belief space according to the DeGroot algorithm.
        """
        if dynamic:
            clusters = np.array(list(nx.get_node_attributes(self.graphs[-1],
                                                            self.BN).values()))
            D = make_dist_matrix(clusters, euclidean_dist,
                                 symmetric=True).astype(np.float32)
            sda = SDA.from_dist_matrix(D, alpha=self.ALPHA, k=self.K,
                                       directed=False)
            A = sda.adjacency_matrix(sparse=False)
            graph = ls.make_graph(A, clusters, self.BN,
                                  media_locs=self.media_locs)
        else:
            graph = self.graphs[-1]
        new_graph = dg.DeGroot(graph, 1, self.BN, self.WN, self.step_n,
                               self.media_locs, self.media_changes)
        self.graphs.append(new_graph[-1])
        self.step_n += 1

    def multiple_steps(self, iter, node_idx=0, action=0, dynamic=False):
        """
            Performes the step function multiple times.
        """
        for i in range(iter):
            self.step(node_idx, action, dynamic)

    def score(self, idx, b, dist_func, max_dist=0.1):
        """
            Returns the difference between the average belief of the network
            on a given iteration and the goal of the influencer node(s).
        """
        gb = np.array(list(nx.get_node_attributes(self.graphs[idx], self.BN).values()))
        mean = np.mean(gb, axis=0)
        s = np.linalg.norm(b-mean)
        return s

    def plot_G(self, G, ax):
        """
            Plots a given graph.
        """
        pos = nx.get_node_attributes(G, self.BN)
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=50)
        nx.draw_networkx_edges(G, pos=pos, ax=ax)
        ax.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    def plot_fancy_G(self, G, ax):
        """
            Saves a plot of a given graph using the agraph layout.
        """
        A = nx.nx_agraph.to_agraph(G)
        A.layout()
        A.draw('figures/fancy_G.png')
        return plt.imread('figures/fancy_G.png')
