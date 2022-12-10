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

# from bokeh.io import show
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import Range1d, Arrow, OpenHead, NormalHead, VeeHead, Panel, Tabs

# from bokeh.models.annotations import Label

from tabulate import tabulate
import scipy as sc
from scipy.linalg import block_diag

import numpy as np
import casadi as ca

from .system import System
from .graph import Graph

class MultiAgent(System):

    def __init__(self):
        super(MultiAgent, self).__init__()
        self.consensus = dict(graph=[], val=[])
        self.synchronization = dict(graphs=[],
                                    systems=[],
                                    state_projection=[],
                                    input_projection=[],
                                    ouput_projection=[],
                                    feedback_gain=[],
                                    reference_system=[],
                                    synchronized=[],
                                    has_intersection=[],
                                    all_agents_identical=[])
        self.synchronized = []

    def __repr__(self):
        return str('multiAgent_system')

    def __str__(self):
        return str('multiAgent_system')

    @classmethod
    def get_networked_system(cls, graphs, systems=None):
        """ Create a system object describing the networked system """

        if systems:
            mode = "synchronization"
            for isystem in systems:
                if systems.__len__() != graphs[0].graph.number_of_nodes():
                    print("dimension mismatch in: get_networked_system")
        else:
            mode = "consensus"

        if mode == "consensus":
            sys_loc_T = cls()
            sys_loc_T.type = "consensus_continuous"
            sys_loc_T.consensus["graph"] = graphs
            sys_loc_T.Ngrid = 50
            for i_graph in graphs:
                sys_loc_T.state_matrix.append(-i_graph.inLaplacian)
                sys_loc_T.input_matrix.append(np.zeros([i_graph.graph.number_of_nodes(), 1]))
                sys_loc_T.output_matrix.append(np.eye(i_graph.graph.number_of_nodes()))

        if mode == "synchronization":
            k = ca.SX.sym('controller_gain')
            N = graphs[0].graph.number_of_nodes()
            L = -k*graphs[0].inLaplacian
            Anet, Bnet, Cnet, Xpr, Upr, Ypr = cls.__get_tot_system(systems, L, k)
            Asyn = Anet + Bnet@L@Cnet
            # build networked system ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sys_loc_T = cls()
            sys_loc_T.type = "synchronize_continuous"
            sys_loc_T.synchronization["graphs"] = graphs
            sys_loc_T.synchronization["systems"] = systems
            sys_loc_T.synchronization["state_projection"] = Xpr
            sys_loc_T.synchronization["input_projection"] = Upr
            sys_loc_T.synchronization["output_projection"] = Ypr
            sys_loc_T.synchronization["feedback_gain"] = k
            sys_loc_T.control_val = 1
            sys_loc_T.Ngrid = 200
            for i_graph in graphs:
                sys_loc_T.state_matrix.append(Asyn)
                sys_loc_T.input_matrix.append(np.zeros([Asyn.shape[0], 1]))
                sys_loc_T.output_matrix.append(np.eye(Asyn.shape[0]))
            # get reference system ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            identical_sys, has_intersection = cls.get_reference_system(sys_loc_T)
            # analysis for synchronization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if identical_sys:
                is_synchron = cls.is_synchronized(sys_loc_T, identicalSystem=identical_sys)
            else:
                # only the case of identical agents is implemented
                is_synchron = None
            sys_loc_T.synchronization["synchronized"] = is_synchron
            sys_loc_T.synchronization["has_intersection"] = has_intersection
            sys_loc_T.synchronization["all_agents_identical"] = identical_sys
        return sys_loc_T

    def plot_consensus(self, sigma=None, Tspace=True, Xspace=False):
        """  simmulate the multi-agents system starting at x0 to check for the consensus value  """
        # **************************************************
        if sigma is not None:
            flag_switch = True
            wB = []
            for i_graph in self.consensus["graph"]:
                wB.append(i_graph.weight_balanced)
            graph_tot = Graph.get_graph_from_union(self.consensus["graph"])
            if all(wB):
                wB_tot = True
            else:
                wB_tot = False
            if graph_tot.strongly_connected:
                sc_tot = True
            else:
                sc_tot = False
            if sc_tot and wB_tot:
                self.consensus["val"] = np.sum(np.array(self.initial_state)) / self.consensus["graph"][0].graph.number_of_nodes()
        else:
            flag_switch = False
            if self.consensus["graph"][0].spanning_tree:
                st_tot = True
            else:
                st_tot = False
            self.__get_consensus()

        # **************************************************
        if Tspace or Xspace:
            if flag_switch:
                T_sample = sigma.data["xdata"][1]
                t_span_tot = np.array([0])
                x_span_tot = self.initial_state
                sigma_span_tot = sigma.signal(0)
                for k in range(sigma.data["xdata"].__len__()-1):
                    t_span_loc = np.linspace(sigma.data["xdata"][k], sigma.data["xdata"][k+1], self.Ngrid)
                    t_span_tot = np.concatenate((t_span_tot, t_span_loc[1:]))
                    index = sigma.signal(t_span_loc[0]).__int__()
                    x_span_loc = self.__sim_networked_system(time_hist=t_span_loc, index=index)
                    sigma_span_loc = sigma.signal(t_span_loc)
                    self.initial_state = x_span_loc[:, -1]
                    x_span_tot = ca.horzcat(x_span_tot, x_span_loc[:, 1:])
                    sigma_span_tot = ca.horzcat(sigma_span_tot, sigma_span_loc[1:].T)
            else:
                t_span_tot = np.linspace(0, self.t_final, self.Ngrid)
                x_span_tot = self.__sim_networked_system(time_hist=t_span_tot)
                index = 0
            self.TimeGrid = t_span_tot
            self.StateHistory = x_span_tot
        # **************************************************
        if Tspace:
            Npoints = self.TimeGrid.shape[0]
            X_line = np.zeros([Npoints, 1])
            if self.consensus["val"]:
                X_line = self.consensus["val"] * np.ones([Npoints, 1])
            # ******************************************************
            if flag_switch:
                pSg = figure(title='switching function')
                pSg.background_fill_color = 'darkslategray'
                pSg.step(self.TimeGrid, list(np.array(sigma_span_tot.T)),  line_width=2, line_color='black')
                pSg.yaxis.axis_label = 'sigma'
                pSg.xaxis.axis_label = 'time'
                pSg.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))

            pS = figure(title='system states')
            pS.background_fill_color = 'darkslategray'
            help_min = 0
            help_max = 0
            for i in range(self.consensus["graph"][index].graph.number_of_nodes()): # LAP1.shape[0]
                pS.line(self.TimeGrid, list(np.array(self.StateHistory[i, :].T)), line_width=3)
                if np.array(self.StateHistory[i, :].T).min() - 0.2 < help_min:
                    help_min = np.array(self.StateHistory[i, :].T).min() - 0.2
                if np.array(self.StateHistory[i, :].T).max() + 0.2 > help_max:
                    help_max = np.array(self.StateHistory[i, :].T).max() + 0.2
            pS.line(self.TimeGrid, list(X_line),  line_width=2, line_dash=[5, 10], line_color='black')
            pS.yaxis.axis_label = 'states'
            pS.xaxis.axis_label = 'time'
            pS.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
            pS.y_range = Range1d(help_min, help_max)

            if flag_switch:
                show(gridplot([[pSg], [pS]], width=500, height=200))
            else:
                show(pS)

        if Xspace:
            # ******************************************************
            pSS = figure(title='system states')
            pSS.background_fill_color = 'darkslategray'
            pSS.line(list(np.array(self.StateHistory[0, :].T)), list(np.array(self.StateHistory[1, :].T)), line_width=3)
            pSS.line( np.linspace(help_min,help_max,100), np.linspace(help_min,help_max,100),  line_width=2, line_dash=[5, 10], line_color='black')
            pSS.circle(np.array(self.initial_state)[0], np.array(self.initial_state)[1], size=10, color="red", alpha=0.5)
            pSS.add_layout(Arrow(end=VeeHead(size=10), line_color="black",
                                 x_start=self.StateHistory[0, -0.3*self.Ngrid].__float__(),
                                 y_start=self.StateHistory[1, -0.3*self.Ngrid].__float__(),
                                 x_end=self.StateHistory[0, -1].__float__(),
                                 y_end=self.StateHistory[1, -1].__float__()
                                 ))
            pSS.xaxis.axis_label = 'state 1'
            pSS.yaxis.axis_label = 'state 2'
            pSS.x_range = Range1d(help_min, help_max)
            pSS.y_range = Range1d(help_min, help_max)
            show(pSS)

    def __get_consensus(self, index=0):

        w, vl, vr = sc.linalg.eig(self.consensus["graph"][index].inLaplacian, left=True)
        # LL = np.matrix(self.consensus["graph"][index].inLaplacian)
        VL = np.matrix(vl)
        VR = np.matrix(vr)
        ONEvec = np.ones([self.consensus["graph"][index].graph.number_of_nodes(), 1])

        ORDER = w.argsort()
        # lambda_1 = w[ORDER[0]]
        VL_1 = VL[:, ORDER[0]]
        # VR_1 = VR[:, ORDER[0]]
        # lambda_2 = w[ORDER[1]]
        # VL_2 = VL[:, ORDER[1]]
        # VR_2 = VR[:, ORDER[1]]

        # get consensus value
        if self.consensus["graph"][index].spanning_tree:
            if self.consensus["graph"][index].weight_balanced:
                self.consensus["val"] = np.sum(np.array(self.initial_state))/self.consensus["graph"][index].graph.number_of_nodes()
            else:
                if not self.consensus["graph"][index].leader:
                    W_hat = 1 / (VL_1.H@ONEvec)*VL_1.H
                    self.consensus["val"] = np.real(W_hat@np.array(self.initial_state)).__float__()
                else:
                    self.consensus["val"] = self.initial_state[self.consensus["graph"][index].leader_node-1].__float__()

    def __sim_networked_system(self, time_hist, index=0, para_sym=None, para_val=None, ref_sys=False):  #
        """   """
        if para_sym == None:
            x = ca.SX.sym('x', self.state_matrix[index].shape[0])
            tf_val = time_hist[1] - time_hist[0]
            F = ca.integrator('F', 'cvodes', dict(x=x, ode=self.state_matrix[index] @ x), dict(t0=0, tf=tf_val))
            Ftot = F.mapaccum(self.Ngrid - 1)
            sol = Ftot(x0=self.initial_state)
        else:
            # simulate the networked system  / if para_sym.is_symbolic():
            x = ca.SX.sym('x', self.state_matrix[index].shape[0])
            tf_val = time_hist[1] - time_hist[0]
            F = ca.integrator('F', 'cvodes', dict(x=x, ode=self.state_matrix[index] @ x, p=para_sym), dict(t0=0, tf=tf_val))
            Ftot = F.mapaccum(self.Ngrid - 1)
            sol = Ftot(x0=self.initial_state, p=para_val)

        # simulate reference system
        if ref_sys:
            Ar = self.synchronization["reference_system"].state_matrix
            xr = ca.SX.sym('xr', Ar.shape[0])
            Fr = ca.integrator('Fr', 'cvodes', dict(x=xr, ode=Ar @ xr), dict(t0=0, tf=tf_val))
            Frtot = Fr.mapaccum(self.Ngrid - 1)
            if isinstance(self.synchronization["reference_system"].initial_state, ca.Function):
                xr0 = self.synchronization["reference_system"].initial_state(self.initial_state)
            else:
                xr0 = self.synchronization["reference_system"].initial_state
            solr = Frtot(x0=xr0)
            return ca.horzcat(self.initial_state, sol['xf']), ca.horzcat(xr0, solr['xf'])
        else:
            return ca.horzcat(self.initial_state, sol['xf'])

    @staticmethod
    def __get_tot_system(systems, L, k):
        """ build total system """
        # put all system parameters into a list ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Atotd = []
        [Atotd.append(i_sys.state_matrix) for i_sys in systems]
        Btotd = []
        [Btotd.append(i_sys.input_matrix) for i_sys in systems]
        Ctotd = []
        [Ctotd.append(i_sys.output_matrix) for i_sys in systems]

        # build uo the network parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Anet = block_diag(*Atotd)
        Bnet = block_diag(*Btotd)
        Cnet = block_diag(*Ctotd)

        # create the states ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        NX = Anet.shape[0]
        Xnet = ca.SX.sym('Xnet', NX)
        nx = [isystem.state_matrix.shape[0] for isystem in systems]

        # extract the states of the individual systems ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        posX = [0]
        [posX.append(posX[-1]+inx) for inx in nx]
        X_states = [Xnet[posX[ii]:posX[ii+1]] for ii in range(systems.__len__())]

        # build up projection functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xpr = []
        [Xpr.append(ca.Function("X2X",[Xnet], [X_states[ii]],['X_tot'],['x_'+str(ii)])) for ii in range(systems.__len__())]
        Upr = ca.Function("X2U", [Xnet, k], [-L@Cnet@Xnet], ['X_tot', 'k'], ['U'])
        Ypr = ca.Function("X2Y", [Xnet], [Cnet@Xnet], ['X_tot'], ['Y'])

        return Anet, Bnet, Cnet, Xpr, Upr, Ypr

    @staticmethod
    def show_switching_data(graph_list):
        """ Show properties of the graphs of a switching newtwork """
        head = [" "]
        dataA = ["weight_balanced    "]
        dataB = ["strongly_connected "]
        dataC = ["spanning_tree      "]
        i_g = 1
        for i_graph in graph_list:
            dataA.append(i_graph.weight_balanced)
            dataB.append(i_graph.strongly_connected)
            dataC.append(i_graph.spanning_tree)
            head.append("graph " + str(i_g))
            i_g += 1
        print(tabulate([dataA, dataB, dataC], headers=head))