
import numpy as np
import cdcps as cps

if __name__ == '__main__':
    E = [(1, 2, 0.4),
         (1, 3, 0.4),
         (2, 4, 0.5),
         (2, 5, 0.3),
         (3, 2, 0.4),
         (4, 1, 0.8),
         (5, 4, 0.3)
         ]
    NET1 = cps.Graph.get_graph_from_edges(E)
    NET_SYS = cps.MultiAgent.get_networked_system(graphs=[NET1])
    NET_SYS.initial_state = np.array([-2, -1, 0, 1.5, 2])
    NET_SYS.t_final = 30
    NET_SYS.plot_consensus()




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
    G1.show_graph_plot("spring")

    pass

