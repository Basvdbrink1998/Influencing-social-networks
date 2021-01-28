import networkx as nx

G = nx.DiGraph()
G.add_nodes_from(['s(1)', 'n(0.5)'])
G.add_edges_from([('s(1)', 's(1)'), ('n(0.5)', 'n(0.5)'), ('s(1)', 'n(0.5)')])

A = nx.nx_agraph.to_agraph(G)
A.layout('dot')
A.draw('Small_graph_2.png')
