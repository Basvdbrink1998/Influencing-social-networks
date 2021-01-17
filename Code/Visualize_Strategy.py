#!/usr/bin/python

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import argparse

from Influence import Static as st
from Influence import _
from Influence.enviroment import SDA_enviroment


def get_strategies():
    strats = dict()
    strats['start_at_goal'] = st.start_at_goal
    strats['get_closest_node'] = st.get_closest_node
    strats['get_closest_center'] = st.get_closest_center
    strats['avg'] = st.avg
    strats['lin_m'] = st.lin_m
    strats['static'] = st.static
    return strats


def visualize_strategy(N, NIT, CENTERS, K, ALPHA, MIN, MAX, start_name,
                       change_name, dynamic, goal):
    NDIM = 2
    scaler = MinMaxScaler()
    strat_dict = get_strategies()
    if start_name in list(strat_dict.keys()):
        start_function = strat_dict[start_name]
    else:
        print("Start name does not exist")
        return
    if change_name in list(strat_dict.keys()):
        change_function = strat_dict[change_name]
    else:
        print("Change name does not exist")
        return
    clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM,
                                                         centers=CENTERS,
                                                         center_box=(MIN, MAX))
    scaler.fit(clusters)
    norm_clusters = scaler.transform(clusters)

    if goal == 'r':
        GOAL = np.random.binomial(1, 0.5, NDIM)
    elif goal == 'vr':
        GOAL = (np.random.binomial(1, 0.5, NDIM) - 0.5) * 40
    else:
        GOAL = np.random.random(NDIM)

    env = SDA_enviroment()
    env.init_features(N, NDIM, CENTERS, K, ALPHA)
    env.insert_nodes(norm_clusters, clusters_labs)
    res, score = st.influence_cluster(env, start_function, change_function,
                                      norm_clusters, GOAL, NIT,
                                      dynamic=dynamic)

    fig, axs = plt.subplots(len(env.graphs), figsize=(10, 10))
    for i in range(len(env.graphs)):
        G = env.graphs[i]
        ax = axs[i]
        belief = nx.get_node_attributes(G, 'belief')
        nx.draw_networkx_nodes(G, belief, nodelist=list(set(G.nodes)-
                               set(env.media_locs)), ax=ax, node_color='b',
                               label='Regular node')
        nx.draw_networkx_nodes(G, belief, nodelist=env.media_locs, ax=ax,
                               node_color='r', label='Influencer node')
        nx.draw_networkx_edges(G, belief, width=1.0, alpha=0.5, ax=ax)
        ax.scatter(GOAL[0], GOAL[1], facecolors='none', edgecolors='g', s=500,
                   label='Goal')

        ax.axis('on')
        ax.set_xlabel('belief (x)')
        ax.set_ylabel('belief (y)')
        ax.axes.set_xlim((-0.15, 1.15))
        ax.axes.set_ylim((-0.15, 1.15))
        ax.title.set_text('Iteration: ' + str(i))
        ax.tick_params(left=True, bottom=True, labelleft=True,
                       labelbottom=True)
    axs[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs a simulation which evaluates different strategies in influencing social networks built using the Social Distance attachment and the DeGroot model.''')
    parser.add_argument('-n', '--n_nodes', type=int, help='Amount of nodes in the generated network. must be an inteture.', default=20)
    parser.add_argument('-nit', '--n_iterations', type=int, help='Amount of DeGroot iterations that the simulation computes.', default=3)
    parser.add_argument('-c', '--n_centers', type=int, help='Amount of clusters of nodes that are generated.', default=3)
    parser.add_argument('-k', '--k', type=int, help='The average amount of connections each node makes.', default=3)
    parser.add_argument('-a', '--alpha', type=float, help='', default=0)
    parser.add_argument('-n_min', '--node_minimum', type=float, help='The lower boundary of the placement of the center of the clusters', default=-8)
    parser.add_argument('-n_max', '--node_maximum', type=int, help='The higher boundary of the placement of the center of the clusters', default=8)
    parser.add_argument('-n_exp', '--n_experiments', type=int, help='The amount of experiments performed for each datapoint', default=200)
    parser.add_argument('-dyn', '--dynamic', type=int, help='If the connections between the nodes change with each DeGroot iteration', default=False)
    parser.add_argument('-g', '--goal', type=str, help='The kind of influencing which is done. vr for very radical, r for radical, nr for non radical', default='r')
    parser.add_argument('-sn', '--start_name', type=str, help='', default='start_at_goal')
    parser.add_argument('-cn', '--change_name', type=str, help='', default='static')
    args = parser.parse_args()
    visualize_strategy(args.n_nodes, args.n_iterations, args.n_centers, args.k,
                       args.alpha, args.node_minimum, args.node_maximum,
                       args.start_name, args.change_name, args.dynamic,
                       args.goal)
