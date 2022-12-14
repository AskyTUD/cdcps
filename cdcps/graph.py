#
#   This file is part of cdcps
#
#   cdcps is a package providing methods for
#   the lecture cdcps like simulating linear dynamical systems, consensus and synchronization problems and analyzing graphs
#
#   Copyright (c) 2022 Andreas Himmel
#                      All rights reserved
#
#   cdcps is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   cdcps is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with cdcps. If not, see <http://www.gnu.org/licenses/>.
#

from tabulate import tabulate
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Graph:
    def __init__(self):
        self.graph = []
        self.adjacency = []
        self.incidence = []
        self.inDegree = []
        self.totDegree = []
        self.inLaplacian = []
        self.totLaplacian = []
        self.reachability = []
        self.weight_balanced = []
        self.spanning_tree = []
        self.leader = []
        self.root_nodes = []
        self.leader_node = []
        self.strongly_connected = []
        self.LaplacianSpectrum = []
        self.algConnectivity = []
        self.subGraphs = []

    def __repr__(self):
        return 'graph_{}N_{}E'.format(self.graph.number_of_nodes(), self.graph.number_of_edges())

    def __str__(self):
        return 'graph_{}N_{}E'.format(self.graph.number_of_nodes(), self.graph.number_of_edges())

    @classmethod
    def get_graph_from_edges(cls, edge_list):
        """ This function returns a graph from the set of edges """
        G = nx.DiGraph()
        G.add_weighted_edges_from(edge_list)

        sortGraph = nx.DiGraph()
        sortGraph.add_nodes_from(sorted(G.nodes(data=True)))
        sortGraph.add_edges_from(G.edges(data=True))
        localGraph = cls()
        return localGraph.__addInfo(sortGraph)

    @classmethod
    def get_graph_from_adjacency(cls, A):
        """ This function returns a graph from an adjacency matrix """
        G = nx.DiGraph()

        edge_list = [(col + 1, row + 1, A[row, col]) for col in range(A.shape[1]) for row in range(A.shape[0]) if A[row, col] != 0.0]
        edge_list_new = edge_list + cls.__NodeList(A, edge_list)
        G.add_weighted_edges_from(edge_list_new)

        sortGraph = nx.DiGraph()
        sortGraph.add_nodes_from(sorted(G.nodes(data=True)))
        sortGraph.add_edges_from(G.edges(data=True))
        localGraph = cls()
        return localGraph.__addInfo(sortGraph)

    @classmethod
    def get_graph_from_Laplacian(cls, L):
        """ This function returns a graph from a Laplacian matrix """
        G = nx.DiGraph()
        A = -L+np.diag(np.diag(L))
        G.add_weighted_edges_from(
            [(col + 1, row + 1, A[row, col]) for col in range(A.shape[1]) for row in range(A.shape[0]) if
             A[row, col] != 0.0])

        sortGraph = nx.DiGraph()
        sortGraph.add_nodes_from(sorted(G.nodes(data=True)))
        sortGraph.add_edges_from(G.edges(data=True))
        localGraph = cls()
        return localGraph.__addInfo(sortGraph)

    @classmethod
    def get_graph_from_union(cls, graph_list):
        """ This function returns a graph from a graph union """
        """ !! TRANSFER THIS FUNCTION INTO __add__ !!"""
        NX = graph_list[0].graph.number_of_nodes()
        A_new = np.zeros([NX, NX])
        for i_g in graph_list:
            A_new = A_new + i_g.adjacency
        for i_row in range(NX):
            for i_col in range(NX):
                if A_new[i_row, i_col] > 0:
                    A_new[i_row, i_col] = 1.0
        return cls.get_graph_from_adjacency(A_new)

    def get_Matrix_Representation(self):
        """ This function yields the adjacency, degree and Laplacian matrix. """
        self.adjacency = nx.to_numpy_matrix(self.graph).transpose()
        deg_in = np.sum(nx.to_numpy_matrix(self.graph).transpose(), axis=1)
        self.inDegree = np.diag(np.asarray(deg_in).flatten())
        self.totDegree = np.diag(np.array([[list(self.graph.degree)[i][1]] for i in range(len(self.graph.nodes))]).T[0])

        inc_nw = np.array(nx.incidence_matrix(self.graph, oriented=True)).tolist().todense()
        inc_w = np.array(nx.incidence_matrix(self.graph, oriented=True)).tolist().todense()
        for row in range(len(self.graph.nodes)):
            for col in range(len(self.graph.edges)):
                if row + 1 in list(self.graph.edges)[col]:
                    inc_w[row, col] = inc_nw[row, col] * self.graph[list(self.graph.edges)[col][0]][list(self.graph.edges)[col][1]]['weight']
        inc_nw = inc_nw.T
        self.incidence = inc_w.T
        self.inLaplacian = self.inDegree - self.adjacency
        self.totLaplacian = self.incidence.T @ self.incidence

    def get_reachability(self):
        """ determine the reachability matrix """
        R = np.linalg.matrix_power(self.adjacency, 0)
        for i in range(1, np.shape(self.adjacency)[0]):
            R = R + np.linalg.matrix_power(self.adjacency, i)
        self.reachability = R

    def get_LaplacianSpectrum(self):
        """ determine the spectrum of the Laplacian matrix """
        eigenValues = np.linalg.eig(self.inLaplacian)[0]
        self.LaplacianSpectrum = np.round(eigenValues[eigenValues.argsort()],4).argsort()
        self.algConnectivity = self.LaplacianSpectrum[1]

    @staticmethod
    def is_Laplacian(A):
        """ Check if A is a Laplacian matrix """
        # check for pos and neg entires
        CHECK_A_diag = [[1] if A[row, col] >= 0 else [0] for row in range(np.size(A, axis=0)) for col in
                        range(np.size(A, axis=1)) if row == col]
        CHECK_A_Ndiag = [[1] if A[row, col] <= 0 else [0] for row in range(np.size(A, axis=0)) for col in
                         range(np.size(A, axis=1)) if row != col]
        CHECK_B = np.all(np.absolute(np.around(A @ np.ones(np.size(A, axis=0)), decimals=8)) == 0)
        if np.all(np.array(CHECK_A_diag) == 1) and np.all(np.array(CHECK_A_Ndiag) == 1) and CHECK_B:
            return True
        return False

    @staticmethod
    def get_Structure(A):
        N_row = A.shape[0]
        N_col = A.shape[1]
        A_st = np.zeros([N_row, N_col])
        for i_row in range(N_row):
            for i_col in range(N_col):
                if A[i_row, i_col] != 0:
                    A_st[i_row, i_col] = 1
        return A_st

    def get_Conected_Subgraphes(self):
        """ This function yields all strongly connected subgraphs """
        subGraphs = []
        if not self.strongly_connected:  # nx.is_strongly_connected(G):
            HELP = list(nx.strongly_connected_components(self.graph))
            for ig in range(len(HELP)):
                subGraphs.append(self.graph.subgraph(HELP[ig]))
        else:
            subGraphs = self.graph
        self.subGraphs = tuple(subGraphs)

    def is_weight_balanced(self):
        """ This function checks if the graph is weight-balanced according to the definition of the lecture. """
        Acol = np.sum(self.adjacency, axis=0)
        Arow = np.sum(self.adjacency, axis=1).transpose()
        self.weight_balanced = np.array_equal(Arow, Acol)

    def has_spanning_tree(self):
        """ Check if the graph has a spanning tree """

        self.root_nodes = [i_node for i_node in self.graph.nodes if np.all(self.reachability[:, i_node - 1].round(10) > 0)]
        if len(self.root_nodes) != 0:
            self.spanning_tree = True
        else:
            self.spanning_tree = False

    def has_leader(self):
        """  Check if the graph has a leader, i.e., a node that does not receive any input """

        self.leader = False
        self.leader_node = []
        if not self.strongly_connected and len(self.root_nodes) == 1:
            path2root = [i_node for i_node in self.graph.nodes if
                         i_node != self.root_nodes[0] and self.is_path(initial_node=i_node, final_node=self.root_nodes[0])]
            if len(path2root) == 0:
                self.leader = True
                self.leader_node = self.root_nodes[0]

    def is_strongly_connected(self):
        """ Check if the graph is strongly connected """
        if np.all(self.reachability.round(10) > 0):
            self.strongly_connected = True
        else:
            self.strongly_connected = False

    def is_path(self, initial_node=1, final_node=2):
        """ Check if there exists a path from an initial node to a final one """
        if self.reachability[final_node - 1, initial_node - 1] > 0:
            return True
        else:
            return False

    def show_graph_data(self):
        """ This function returns all data of a graph """
        prim_dataNodeEdge = [['total graph', list(self.graph.nodes), list(self.graph.edges)]]

        ADJ_LAP2 = -self.totLaplacian + np.diag(np.diag(self.totLaplacian))
        DEG_LAP2 = np.diag(np.diag(self.totLaplacian))

        prim_dataMatrix_baic = [[self.adjacency, self.incidence, self.inDegree, self.totDegree]]
        prim_dataMatrix_buildA = [[self.inLaplacian]]
        prim_dataMatrix_buildB = [[self.totDegree, ADJ_LAP2, DEG_LAP2]]
        prim_dataPred = [['predecessors of ' + str(i + 1), set(self.graph.predecessors(i + 1))]
                         for i in range(self.graph.number_of_nodes()) if i + 1 in self.graph.nodes]
        prim_dataConBal = [['is strongly connected __:', self.strongly_connected],      # nx.is_strongly_connected(self.graph)
                           ['is weight balanced _____:', self.weight_balanced]]         # self.is_weight_balanced()

        prim_dataEIG = [
            ['eigenvalue ' + str(i + 1), "{num.real:+0.04f} {num.imag:+0.04f}i".format(num=self.LaplacianSpectrum[i])]
            if self.LaplacianSpectrum[i].imag != 0.0 else
            ['eigenvalue ' + str(i + 1), "{num.real:+0.04f}".format(num=self.LaplacianSpectrum[i])]
            for i in range(len(self.LaplacianSpectrum))
        ]

        prim_dataSub = [['subgraph ' + str(iG), set(self.subGraphs[iG].nodes), set(self.subGraphs[iG].edges)] for iG in
                        range(len(self.subGraphs)) if not self.strongly_connected]

        # with np.printoptions(precision=2, suppress=True):
        # ===============================================================================================
        print(tabulate(prim_dataNodeEdge, headers=[' ', 'Node', 'Edges']))
        print('\n Predecessor sets: \n')
        print(tabulate(prim_dataPred, headers=[' ', 'Nodes']))
        print('\n Structure describing matrices: \n')
        print(tabulate(prim_dataMatrix_baic, headers=['adjacency matrix (A)',
                                                      'incidence matrix (I)',
                                                      'in-deree matrix (D_in)',
                                                      'degree matrix (D_tot)']))
        print('\n Laplacian matrices (using the incoming edges): \n')
        print(tabulate(prim_dataMatrix_buildA, headers=['Laplacian matrix (D_in-A)']))
        print('\n Laplacian matrices (using the incoming and outgoing edges): \n')
        print(tabulate(prim_dataMatrix_buildB, headers=['Laplacian matrix (I.T I)',
                                                        'undirected adjacency matrix (A_ud)',
                                                        'degree matrix (D_ud)']))

        print(tabulate(prim_dataConBal, headers=[' ', ' '], tablefmt="plain"))
        print('\n Laplacian spectrum: \n')
        print(tabulate(prim_dataEIG, headers=[' ', 'Value']))
        print('\n Strongly connected subgraphs: \n')
        print(tabulate(prim_dataSub, headers=[' ', 'Node', 'Edges']))

        if not np.all(np.around(np.sum(self.inLaplacian, axis=0), decimals=8) == 0) == prim_dataConBal[1][1]:
            print('\n !!! is_weight_balanced check is wrong !!!')

    def plot_graph(self, mode):
        """ This function plot the graph by predefined properties. """
        help = self.graph.copy()
        for ie in self.graph.edges():
            if self.adjacency[ie[1]-1, ie[0]-1] == 0:
                help.remove_edge(ie[0], ie[1])

        if mode == "spring":
            pos = nx.spring_layout(help)
        if mode == "circular":
            pos = nx.circular_layout(help)
        if mode == "random":
            pos = nx.random_layout(help)
        nx.draw_networkx_nodes(help, pos, node_size=500, node_color='#eeeeee')
        nx.draw_networkx_edges(help, pos, edgelist=help.edges(), edge_color='black', arrowsize=20)
        nx.draw_networkx_labels(help, pos)
        nx.draw_networkx_edge_labels(help, pos, edge_labels=nx.get_edge_attributes(help, 'weight'))
        plt.show()

    def __addInfo(self, sortGraph):
        self.graph = sortGraph
        self.get_Matrix_Representation()  # ADJ, INC_nw, INC_w, DEG_in, DEG_tot, LAP1_in, LAP2 =
        self.get_reachability()
        self.get_LaplacianSpectrum()
        self.has_spanning_tree()
        self.has_leader()
        self.is_strongly_connected()
        self.is_weight_balanced()
        self.get_Conected_Subgraphes()
        return self

    @staticmethod
    def __NodeList(A, edge_list):
        d_list = []
        if np.any(np.where(~A.any(axis=0))[0] == np.where(~A.any(axis=1))[0]):
            d_list = []
            for ii in range(A.shape[0] - 1):
                if not (ii + 1, ii + 2) in [(element[0], element[1]) for element in edge_list]:
                    d_list.append((ii + 1, ii + 2, 0.0))
        return d_list


