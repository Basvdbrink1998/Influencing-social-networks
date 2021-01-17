#!/usr/bin/env python

"""
    DeGroot.py: Contains multiple function performing the DeGroot algorithm on
    networkx graphs.
"""

import networkx as nx
import numpy as np
import copy


def update_graph(G, belief_name, weight_name):
    """
        Performs a iteration according to the DeGroot model.
    """
    G2 = G.copy()
    for node in G.nodes:
        neighbors = G.adj[node]
        believe = 0
        weight_size = 0
        if len(neighbors) < 1:
            believe = G.nodes[node][belief_name]
        else:
            for n in neighbors:
                weight_size += neighbors[n][weight_name]
            for n in neighbors:
                believe += G.nodes[n][belief_name] * (neighbors[n][weight_name]/weight_size)
        G2.nodes[node][belief_name] = believe
    return G2


def update_media(G, names, changes, belief_name, iter):
    """
        Updates the influencer nodes in a network according to the array of
        changes given the names of the influencer nodes and the name of the
        belief attribute.
    """
    for i in range(len(names)):
        for j in range(len(changes)):
            G.nodes[names[i]][belief_name] = changes[i][iter]
    return G


def DeGroot(G, max_iter, belief_name, weight_name, iter=0, media_names=None,
            media_changes=None):
    """
        Returns a list of graphs where the DeGroot algorithm is performed for
        each position.
    """
    Graphs = [G]
    for i in range(max_iter):
        if type(Graphs[0]) == list:
            Graphs = Graphs[0]
        current_G = copy.deepcopy(Graphs[-1])
        if media_changes is not None:
            current_G = update_media(current_G, media_names, media_changes, belief_name, iter+1)
        Graphs.append(update_graph(current_G, belief_name, weight_name))
    return Graphs


def plot_changes(Graphs, belief_name, weight_name, idx):
    """
        Return a summary of the development of one dimension of the belief
        space from a list of graphs.
    """
    res = np.empty((len(Graphs[0].nodes), len(Graphs)))
    node_names = list(Graphs[0].nodes.keys())
    for i in range(len(Graphs)):
        believes = nx.get_node_attributes(Graphs[i], belief_name)
        for j in range(len(node_names)):
            name = node_names[j]
            res[j,i] = believes[name][idx]
    return res
