import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_nodes_from(['S(1)', 'N(0.5)'])
G.add_edges_from([('S(1)', 'S(1)'), ('N(0.5)', 'N(0.5)'), ('S(1)', 'N(0.5)')])

A = nx.nx_agraph.to_agraph(G)
A.layout('dot')
A.draw('Small_graph_2.png')
