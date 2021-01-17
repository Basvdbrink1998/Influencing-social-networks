#!/usr/bin/env python

"""
    Static.py:
"""

from enviroment import SDA_enviroment
from sklearn.preprocessing import MinMaxScaler
import _
import matplotlib.pyplot as plt
import numpy as np
import DeGroot as dg
import Learning_SDC as ls
from sklearn.cluster import KMeans
from sdnet.utils import euclidean_dist

import time


def lin(pos, goal, x, n_steps):
    """
        Returns the y value of a linear function based which goes from y=pos
        on x=0 to y=goal on x=n_steps.
    """
    return pos + (1/(n_steps-1) * (goal-pos))*x


def lin_m(pos, goal, n_steps):
    """
        Returns an array of y values of a linear function based which goes
        from y=pos on x=0 to y=goal on x=n_steps.
    """
    res = []
    for i in range(n_steps):
        res.append(lin(pos, goal, i, n_steps))
    return np.array(res)


def static(val, Non, length):
    """
        Returns an array containing only the values val.
    """
    return np.zeros((length, len(val))) + val


def k_means(clusters, n_features):
    """
        Performes the K_means clustering algorithm on the clusters of nodes.
    """
    kmeans = KMeans(n_clusters=n_features).fit(clusters)
    return kmeans


def start_at_goal(clusters, goal=None):
    """
        Returns the give goal variable.
    """
    return goal


def find_most(vec):
    """
        Return the vaulue of the most occuring value in an array.
    """
    counts = np.bincount(vec)
    return np.argmax(counts)


def get_largest_center(clusters, n_features=10, goal=None):
    """
        Returns the index of the largest cluster in a group of nodes.
    """
    kmeans = KMeans().fit(clusters)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    largest = find_most(labels)
    return centers[largest]


def get_closest_center(clusters, n_features=10, goal=None):
    """
        Returns the location of the center of the cluster which is clostest to the goal.
    """
    kmeans = KMeans(n_clusters=n_features).fit(clusters)
    centers = kmeans.cluster_centers_
    return centers[np.argmin(np.sum(np.absolute(centers-goal), axis=1))]


def get_closest_node(clusters, n_features=10, goal=None):
    """
        Returns the location of the node which is clostest to the goal.
    """
    return clusters[np.argmin(np.sum(np.absolute(clusters-goal), axis=1))]


def avg(clusters, n_features=10, goal=None):
    """
        Returns the average location of all the nodes.
    """
    return np.mean(clusters, axis=0)


def influence_cluster(env, start_function, change_function, clusters, GOAL,
                      NIT, dynamic=True, n_clusters=3):
    """
        Influences the inputted graph according to the inputted start and
        change function towards the goal.
    """
    if start_function:
        loc = start_function(clusters, goal=GOAL)
        changes = change_function(loc, GOAL, NIT+1)
        env.insert_media(loc, changes)
    env.connect_graph()
    env.multiple_steps(NIT, dynamic=dynamic)

    score = env.score(-1, GOAL, euclidean_dist)
    res = dg.plot_changes(env.graphs, env.BN, env.WN, 0)

    return res, score


if __name__ == "__main__":
    n_features = 3
    N = 20
    NDIM = 2
    NMED = 1
    NIT = 20
    CENTERS = 3
    K = 3
    ALPHA_SMALL = 2
    ALPHA_LARGE = np.inf
    ALPHA = 2
    ALPHAS = np.linspace(0, 3, num=9)
    MIN = -8
    MAX = 8
    belief_name = 'belief'
    weight_name = 'weight'
    scaler = MinMaxScaler()
    N_exp = 300
    dynamic = False

    strategies = ['closest_center', 'Largest_center', 'closest_center static', 'Largest_center static']
    start_functions = [get_closest_center, get_largest_center]
    change_functions = [lin_m, lin_m]
    locs = []
    changes = []

    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM,
                                                         centers=CENTERS,
                                                         center_box=(MIN, MAX))
    scaler.fit(clusters)
    norm_clusters = scaler.transform(clusters)
    start_time = time.time()
    scores = np.zeros((len(ALPHAS), len(strategies), N_exp))
    n_clusters = np.amax(clusters_labs) + 1
    print(n_clusters)
    for k in range(len(ALPHAS)):
        for i in range(len(strategies)):
            for j in range(N_exp):
                # GOAL = np.random.random(NDIM)
                GOAL = np.array([20, 20])
                env = SDA_enviroment()
                env.init_features(N, NDIM, CENTERS, K, ALPHAS[k])
                env.insert_nodes(norm_clusters, clusters_labs)
                res, score = influence_cluster(env, start_functions[i],
                                               change_functions[i],
                                               norm_clusters, GOAL, NIT,
                                               dynamic=dynamic, n_clusters=n_clusters)
                scores[k, i, j] = score
    scores = np.array(scores)

    v = []
    m = []
    for s in scores:
        v.append(np.var(s, axis=1))
        # m.append(np.mean(s, axis=1))
        m.append(np.median(s, axis=1))
    m = np.array(m).transpose()
    v = np.array(v).transpose()
    for s in range(len(m)):
        plt.plot(ALPHAS, m[s], label=strategies[s])
        # plt.fill_between(ALPHAS, m[s]-v[s], m[s]+v[s])
    plt.ylabel("Score")
    plt.xlabel("Degree of homophily")
    plt.legend()
    end_time = time.time()
    print("\n\n Time gone by:", end_time-start_time)
    plt.show()
