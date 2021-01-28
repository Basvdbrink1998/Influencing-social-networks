import networkx as nx

G = nx.DiGraph()
G.add_nodes_from(['s(1)', 'n(0)'])
G.add_edges_from([('s(1)', 's(1)'), ('n(0)', 'n(0)')])

A = nx.nx_agraph.to_agraph(G)
A.layout('dot')
A.draw('Small_graph.png')
