#!/usr/bin/python

from Influence import Static as st
from Influence import _
from Influence.enviroment import SDA_enviroment
import numpy as np
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def simulation(N, NIT, CENTERS, K, a_min, a_max, n_a, MIN, MAX, N_exp, dynamic,
               goal):
    NDIM = 2
    scaler = MinMaxScaler()

    ALPHAS = np.linspace(a_min, a_max, num=n_a)

    strategies = ['Largest_center', 'get_closest_center', 'mean',
                  'get_closest_node', 'start at goal', 'None']
    start_functions = [st.get_largest_center, st.get_closest_center, st.avg,
                       st.get_closest_node, st.start_at_goal, None]
    change_functions = [st.lin_m, st.lin_m, st.lin_m, st.lin_m, st.static,
                        None]
    start_time = time.time()
    scores = np.zeros((len(ALPHAS), len(strategies), N_exp))

    for j in range(N_exp):
        clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM,
                                                             centers=CENTERS,
                                                             center_box=(MIN,
                                                                         MAX))
        scaler.fit(clusters)
        norm_clusters = scaler.transform(clusters)
        if goal == 'r':
            GOAL = np.random.binomial(1, 0.5, NDIM)
        elif goal == 'vr':
            GOAL = (np.random.binomial(1, 0.5, NDIM) - 0.5) * 40
        else:
            GOAL = np.random.random(NDIM)
        for k in range(len(ALPHAS)):
            for i in range(len(strategies)):
                env = SDA_enviroment()
                env.init_features(N, NDIM, CENTERS, K, ALPHAS[k])
                env.insert_nodes(norm_clusters, clusters_labs)
                res, score = st.influence_cluster(env, start_functions[i],
                                                  change_functions[i],
                                                  norm_clusters, GOAL, NIT,
                                                  dynamic=dynamic,
                                                  n_clusters=CENTERS)
                scores[k, i, j] = score
    scores = np.array(scores)

    v = []
    m = []
    for s in scores:
        v.append(np.var(s, axis=1))
        m.append(np.median(s, axis=1))
    m = np.array(m).transpose()
    v = np.array(v).transpose()
    for s in range(len(m)):
        plt.plot(ALPHAS, m[s], label=strategies[s])
    plt.ylabel("Score")
    plt.xlabel("Degree of homophily")
    plt.legend()
    end_time = time.time()
    print("\n\n Time gone by:", end_time-start_time)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs a simulation which evaluates different strategies in influencing social networks built using the Social Distance attachment and the DeGroot model.''')
    parser.add_argument('-n', '--n_nodes', type=int, help='Amount of nodes in the generated network. must be an inteture.', default=20)
    parser.add_argument('-nit', '--n_iterations', type=int, help='Amount of DeGroot iterations that the simulation computes.', default=20)
    parser.add_argument('-c', '--n_centers', type=int, help='Amount of clusters of nodes that are generated.', default=3)
    parser.add_argument('-k', '--k', type=int, help='The average amount of connections each node makes.', default=3)
    parser.add_argument('-a_min', '--alpha_min', type=float, help='The lowest degree of homophily that is used in in the simulation.', default=0)
    parser.add_argument('-a_max', '--alpha_max', type=float, help='The lowest highest of homophily that is used in in the simulation.', default=3)
    parser.add_argument('-n_a', '--n_alpha', type=int, help='The number of alphas tested in the simulation', default=20)
    parser.add_argument('-n_min', '--node_minimum', type=float, help='The lower boundary of the placement of the center of the clusters', default=-8)
    parser.add_argument('-n_max', '--node_maximum', type=int, help='The higher boundary of the placement of the center of the clusters', default=8)
    parser.add_argument('-n_exp', '--n_experiments', type=int, help='The amount of experiments performed for each datapoint', default=200)
    parser.add_argument('-dyn', '--dynamic', type=int, help='If the connections between the nodes change with each DeGroot iteration', default=False)
    parser.add_argument('-g', '--goal', type=str, help='The kind of influencing which is done. vr for very radical, r for radical, nr for non radical', default='r')
    args = parser.parse_args()
    simulation(args.n_nodes, args.n_iterations, args.n_centers, args.k, args.alpha_min, args.alpha_max, args.n_alpha, args.node_minimum, args.node_maximum, args.n_experiments, args.dynamic, args.goal)
