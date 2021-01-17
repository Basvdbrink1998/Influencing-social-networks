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


    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, centers=CENTERS, center_box=(MIN, MAX))
    scaler.fit(clusters)
    print(clusters_labs)
    norm_clusters = scaler.transform(clusters)

    plt.scatter(clusters[:, 0], clusters[:, 1])

    start_time = time.time()
    env = SDA_enviroment()
    env.init_features(N, NDIM, CENTERS, K, ALPHA)
    env.insert_nodes(norm_clusters, clusters_labs)
    plot_G(self, G, ax)



    plt.show()
