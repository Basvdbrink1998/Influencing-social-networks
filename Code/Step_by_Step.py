#!/usr/bin/python

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import argparse

from Influence.enviroment import SDA_enviroment
from Influence import _


def plot_nodes(norm_clusters):
    """
        Plots all of the given nodes without connections.
    """
    plt.scatter(norm_clusters[:, 0], norm_clusters[:, 1])
    plt.xlabel('belief (x)')
    plt.ylabel('belief (y)')
    plt.show()


def plot_network(env, G):
    """
        Plots a network including the connections.
    """
    ax = plt.subplot()
    env.plot_G(G, ax)
    plt.xlabel('belief (x)')
    plt.xlabel('belief (y)')
    plt.show()


def plot_iterations(env, NIT, n_plots):
    """
        Plots a network at different DeGroot iterations.
    """
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


def simulation(N, NIT, CENTERS, K, ALPHA, MIN, MAX, dynamic, n_plots):
    """
        Simulates a network troughout different DeGroot iterations and plot
        different stages of generating the network and different DeGroot
        iterations.
    """
    NDIM = 2
    scaler = MinMaxScaler()
    n_plots = 3

    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM,
                                                         centers=CENTERS,
                                                         center_box=(MIN, MAX))
    scaler.fit(clusters)
    norm_clusters = scaler.transform(clusters)

    plot_nodes(norm_clusters)

    env = SDA_enviroment()
    env.init_features(N, NDIM, CENTERS, K, ALPHA)
    env.insert_nodes(norm_clusters, clusters_labs)
    G = env.connect_graph()
    plot_network(env, G)

    env.multiple_steps(NIT, dynamic=dynamic)

    plot_iterations(env, NIT, n_plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs a simulation which evaluates different strategies in influencing social networks built using the Social Distance attachment and the DeGroot model.''')
    parser.add_argument('-n', '--n_nodes', type=int, help='Amount of nodes in the generated network. must be an inteture.', default=20)
    parser.add_argument('-nit', '--n_iterations', type=int, help='Amount of DeGroot iterations that the simulation computes.', default=10)
    parser.add_argument('-c', '--n_centers', type=int, help='Amount of clusters of nodes that are generated.', default=3)
    parser.add_argument('-k', '--k', type=int, help='The average amount of connections each node makes.', default=3)
    parser.add_argument('-a', '--alpha', type=float, help='', default=np.inf)
    parser.add_argument('-n_min', '--node_minimum', type=float, help='The lower boundary of the placement of the center of the clusters', default=-8)
    parser.add_argument('-n_max', '--node_maximum', type=int, help='The higher boundary of the placement of the center of the clusters', default=8)
    parser.add_argument('-n_exp', '--n_experiments', type=int, help='The amount of experiments performed for each datapoint', default=200)
    parser.add_argument('-dyn', '--dynamic', type=int, help='If the connections between the nodes change with each DeGroot iteration', default=False)
    parser.add_argument('-n_p', '--n_plots', type=str, help='', default='3')
    args = parser.parse_args()
    simulation(args.n_nodes, args.n_iterations, args.n_centers, args.k, args.alpha, args.node_minimum, args.node_maximum, args.dynamic,
               args.n_plots)
