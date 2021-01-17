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


def func(x):
    a = 1
    b = 1
    c = 1
    return a-(1/(b*(x+1)**c))


def move_to_goal(pos, goal, x, c):
    a = goal
    b = 1/(goal-pos)
    return a-1/(b*np.power(x+1, float(c)))


def lin(pos, goal, x, n_steps):
    return pos + (1/(n_steps-1) * (goal-pos))*x


def lin_m(pos, goal, n_steps):
    res = []
    for i in range(n_steps):
        res.append(lin(pos, goal, i, n_steps))
    return np.array(res)


def static(pos, goal, n_steps):
    return np.zeros((n_steps, len(pos))) + pos


def k_means(clusters, n_features):
    kmeans = KMeans(n_clusters=n_features).fit(clusters)
    return kmeans


def start_at_goal(clusters, n_features=10, goal=None):
    return goal


def find_most(vec):
    counts = np.bincount(vec)
    return np.argmax(counts)


def get_largest_center(clusters, n_features=10, goal=None):
    kmeans = KMeans(n_clusters=n_features).fit(clusters)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    largest = find_most(labels)
    return centers[largest]


def get_closest_node(clusters, n_features=10, goal=None):
    return clusters[np.argmax(np.sum(np.absolute(clusters-goal), axis=1))]


def avg(clusters, n_features=10, goal=None):
    return np.mean(clusters, axis=0)


def influence_cluster(env, start_function, change_function, clusters, GOAL, NIT, dynamic=True):
    if start_function:
        loc = start_function(clusters, goal=GOAL)
        changes = change_function(loc, GOAL, NIT)
        env.insert_media(loc, changes)
    env.connect_graph()
    env.multiple_steps(NIT, dynamic=dynamic)

    score = env.score(-1, GOAL, euclidean_dist)
    res = dg.plot_changes(env.graphs, env.BN, env.WN, 0)

    return res, score


def plot_results(res, ax):
    ls.plot_results(ax, res)


if __name__ == "__main__":
    n_features = 3
    N = 20
    NDIM = 2
    NMED = 1
    NIT = 20
    CENTERS = 3
    NS = np.arange(5, 2000, 100)
    ALPHA_SMALL = 2
    ALPHA_LARGE = np.inf
    ALPHA = 2
    MIN = -8
    MAX = 8
    belief_name = 'belief'
    weight_name = 'weight'
    scaler = MinMaxScaler()
    N_exp = 1
    dynamic = False

    strategies = ['Largest_center', 'mean', 'start at goal', 'get_closest_node', 'None']
    start_functions = [get_largest_center, avg, start_at_goal, get_closest_node, None]
    change_functions = [lin_m, lin_m, static, lin_m, None]
    locs = []
    changes = []

    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, centers=CENTERS, center_box=(MIN, MAX))
    scaler.fit(clusters)
    norm_clusters = scaler.transform(clusters)
    start_time = time.time()
    scores = np.zeros((len(NS), len(strategies), N_exp))
    for k in range(len(NS)):
        for i in range(len(strategies)):
            for j in range(N_exp):
                GOAL = np.random.random(NDIM)
                # GOAL = np.array([20,20])
                env = SDA_enviroment()
                env.init_features(NS[k], NDIM, CENTERS, K, ALPHA)
                env.insert_nodes(norm_clusters, clusters_labs)
                res, score = influence_cluster(env, start_functions[i], change_functions[i], norm_clusters, GOAL, NIT, dynamic)
                scores[k, i, j] = score
    scores = np.array(scores)

    v = []
    m = []
    for s in scores:
        v.append(np.var(s, axis=1))
        m.append(np.mean(s, axis=1))
    m = np.array(m).transpose()
    v = np.array(v).transpose()
    for s in range(len(m)):
        plt.plot(NS, m[s], label=strategies[s])
        # plt.fill_between(ALPHAS, m[s]-v[s], m[s]+v[s])
    correlation = np.corrcoef(NS, m[s])
    print("correlation, ", correlation)
    plt.ylabel("Score")
    plt.xlabel("Network size")
    plt.legend()
    end_time = time.time()
    print("\n\n Time gone by:", end_time-start_time)
    plt.show()
