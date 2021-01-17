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
    N_exp = 500
    dynamic = False
    n_plots = 3


    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, centers=CENTERS, center_box=(MIN, MAX))
    scaler.fit(clusters)
    norm_clusters = scaler.transform(clusters)

    plt.scatter(norm_clusters[:, 0], norm_clusters[:, 1])
    plt.xlabel('belief (x)')
    plt.ylabel('belief (y)')
    plt.show()

    start_time = time.time()
    env = SDA_enviroment()
    env.init_features(N, NDIM, CENTERS, K, ALPHA)
    env.insert_nodes(norm_clusters, clusters_labs)
    G = env.connect_graph()
    ax = plt.subplot()
    env.plot_G(G, ax)

    plt.xlabel('belief (x)')
    plt.xlabel('belief (y)')
    plt.show()

    env.multiple_steps(NIT, dynamic=dynamic)

    its = np.linspace(0, NIT, n_plots+1)[1:]
    fig, axs = plt.subplots(n_plots)
    for i in range(n_plots):
        env.plot_G(env.graphs[int(its[i])], axs[i])

        axs[i].set_xlabel('belief (x)')
        axs[i].set_xlabel('belief (y)')
        axs[i].axes.set_xlim((-0.05, 1.05))
        axs[i].axes.set_ylim((-0.05, 1.05))
        axs[i].set_title('iteration: ' + str(int(its[i])))
    plt.subplots_adjust(hspace=0.5)
    plt.show()
