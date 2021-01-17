import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_nodes_from(['S(1)', 'N(0)'])
G.add_edges_from([('S(1)', 'S(1)'), ('N(0)', 'N(0)')])

A = nx.nx_agraph.to_agraph(G)
A.layout('dot')
A.draw('Small_graph.png')
