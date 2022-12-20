import numpy as np

import cdcps as cps

E = [(1, 2, 1.0),
     (2, 1, 1.0),
     (1, 3, 1.0),
     (3, 1, 1.0),
     (2, 4, 1.0),
     (4, 2, 1.0),
     (2, 5, 1.0),
     (5, 2, 1.0),
     (3, 2, 1.0),
     (2, 3, 1.0),
     (4, 1, 1.0),
     (1, 4, 1.0),
     (5, 4, 1.0),
     (4, 5, 1.0)
     ]
G1 = cps.Graph.get_graph_from_edges(E)
G1.show_graph_data()
G1.plot_graph("spring")

xgrid = np.array([0, 1, 3])
ygrid = np.array([0, 1, 0])
sg1 = cps.Signals.get_PC_Function(xgrid, ygrid, 0)
print(sg1)
print("time   | ", "{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}".format(0, 1, 2, 3, 4))
print("values | ", sg1(0), sg1(1), sg1(2), sg1(3), sg1(4))
