import networkx as nx

"""
    Small_graph.py: Plots a small example of a iteration of the Degroot model
     for Figure 3.4
"""

G = nx.DiGraph()
G.add_nodes_from(['s(1)', 'n(0.5)'])
G.add_edges_from([('s(1)', 's(1)'), ('n(0.5)', 'n(0.5)'), ('s(1)', 'n(0.5)')])

A = nx.nx_agraph.to_agraph(G)
A.layout('dot')
A.draw('Small_graph_2.png')
