import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from Influence import Learning_SDC as ls
from Influence import _
from Influence.enviroment import SDA_enviroment
from Influence import DeGroot as dg


N = 5
NDIM = 2
NIT = 20
CENTERS = 2
K = 2
ALPHA = np.inf
belief_name = 'belief'
weight_name = 'weight'
scaler = MinMaxScaler()

clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, centers=CENTERS)
scaler.fit(clusters)
norm_clusters = scaler.transform(clusters)
env = SDA_enviroment()
env.init_features(N, NDIM, CENTERS, K, ALPHA)
env.insert_nodes(norm_clusters, clusters_labs)
env.connect_graph()
env.multiple_steps(NIT, dynamic=False)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)
node_names = list(env.graphs[0].nodes.keys())
colors = nx.get_node_attributes(env.graphs[0], 'color')

ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(env.plot_fancy_G(env.graphs[0], ax0))
ax0.axis('off')
ax0.title.set_text('Connections in the network')

ax1 = fig.add_subplot(gs[0, 1])
env.plot_G(env.graphs[0], ax1)
ax1.set_xlabel('x Belief')
ax1.set_ylabel('y Belief')
ax1.title.set_text('Starting beliefs')

ax2 = fig.add_subplot(gs[1, 0])
results = dg.plot_changes(env.graphs, belief_name, weight_name, 0)
ls.plot_results(ax2, results, node_names, show_legend=True, remove_dupes=False,
                colors=None, idx=0)
ax2.title.set_text('x beliefs over time')

ax3 = fig.add_subplot(gs[1, 1])
results2 = dg.plot_changes(env.graphs, belief_name, weight_name, 1)
ls.plot_results(ax3, results2, node_names, show_legend=True,
                remove_dupes=False, colors=None, idx=1)
ax3.title.set_text('y beliefs over time')

plt.show()
