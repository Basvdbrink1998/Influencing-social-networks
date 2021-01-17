from enviroment import SDA_enviroment
from sklearn.preprocessing import MinMaxScaler
import _
import matplotlib.pyplot as plt
import numpy as np
import Static as st
import networkx as nx


N = 10
NDIM = 2
NMED = 1
NIT = 3
CENTERS = 3
K = 3
ALPHA = np.inf
MIN = 0
MAX = 1
padding = 0.3
belief_name = 'belief'
weight_name = 'weight'
scaler = MinMaxScaler()

locs = []
changes = []

# start_function = st.start_at_goal
start_function = st.get_closest_node
# start_function = st.get_largest_center
# start_function = st.avg
change_function = st.lin_m

clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, centers=CENTERS, center_box=(MIN, MAX))
scaler.fit(clusters)
norm_clusters = scaler.transform(clusters)

GOAL = np.array([1,1])
env = SDA_enviroment()
env.init_features(N, NDIM, CENTERS, K, ALPHA)
env.insert_nodes(norm_clusters, clusters_labs)
res, score = st.influence_cluster(env, start_function, change_function, norm_clusters, GOAL, NIT, dynamic=True)


fig, axs = plt.subplots(len(env.graphs), figsize=(10, 10))
for i in range(len(env.graphs)):
    G = env.graphs[i]
    ax = axs[i]
    belief = nx.get_node_attributes(G, 'belief')
    nx.draw_networkx_nodes(G, belief, nodelist=list(set(G.nodes)-set(env.media_locs)), ax=ax, node_color='b', label='Regular node')
    nx.draw_networkx_nodes(G, belief, nodelist=env.media_locs, ax=ax, node_color='r', label='Influencer node')
    nx.draw_networkx_edges(G, belief, width=1.0, alpha=0.5, ax=ax)
    ax.scatter(GOAL[0], GOAL[1], facecolors='none', edgecolors='g', s=500, label='Goal')

    ax.axis('on')
    ax.set_xlabel('belief (x)')
    ax.set_ylabel('belief (y)')
    ax.axes.set_xlim((-0.15, 1.15))
    ax.axes.set_ylim((-0.15, 1.15))
    ax.title.set_text('Iteration: ' + str(i))
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
axs[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()
