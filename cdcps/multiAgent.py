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

from tabulate import tabulate
import scipy as sc
from scipy.linalg import block_diag

import numpy as np
import casadi as ca

from .system import System
from .graph import Graph
import cdcps.basic_fun as basic


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
                                    has_identical_agents=[])
        #self.synchronized = []

    def __repr__(self):
        return str('multiAgent_system')

    def __str__(self):
        return str('multiAgent_system')

    def has_intersection(self):
        return self.synchronization["has_intersection"]

    def has_identical_agents(self):
        return self.synchronization["has_identical_agents"]

    def synchronized(self):
        return self.synchronization["synchronized"]

    @staticmethod
    def get_system_intersection(spectrum_data):
        """ design a state matrix from the intersection of the eigenvalue data """
        # number of systems ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        N_sys = spectrum_data.__len__()
        # build for each system a list of all Jordan blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tot_block = []
        for i_sys in spectrum_data:
            all_sys_block = []
            for i_J_block in i_sys:
                for i_key in i_J_block[3].keys():
                    if i_key.startswith('of'):
                        block = i_J_block[3].__getitem__(i_key).tolist()
                        nr_block = i_J_block[3]['NofSize' + str(int(i_key[-1]))]
                        [all_sys_block.append(block) for ii in range(nr_block)]
            tot_block.append(all_sys_block)
        # remove multiple blocks from the 1st system ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        block1_l_red = []
        [block1_l_red.append(i_block) for i_block in tot_block[0] if i_block not in block1_l_red]
        # check if elements of the first system appear in the other systems ~~~~~~~~~~~~~

        diag4ref = []
        for i_blocks1 in block1_l_red:
            N = [tot_block[0].count(i_blocks1)]
            # is block in the Jordan blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i_sys in tot_block[1:]:
                if i_blocks1 in i_sys:
                    # number how often it appears ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    N.append(i_sys.count(i_blocks1))
            # if block is in all systems ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if N.__len__() == spectrum_data.__len__():
                take_n = np.min(N)
                [diag4ref.append(i_blocks1) for ii in range(take_n)]
        # convert block with complex eigenvalues to real on ~~~~~~~~~~~~~~~~~~~~~
        Aref = np.matrix(block_diag(*diag4ref))
        Cref = 0.1 * np.ones([1, Aref.shape[1]])

        return Aref, Cref

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
            sys_loc_T.synchronization["has_identical_agents"] = identical_sys
        return sys_loc_T

    def get_reference_system(self):
        self.synchronization["systems"]
        list_A = []
        [list_A.append(i_sys.state_matrix.tolist())
         for i_sys in self.synchronization["systems"]
         if i_sys.state_matrix.tolist() not in list_A]

        # check if all systems identical ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if list_A.__len__() == 1:
            # get normalized left eigenvector ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            _, vl, _ = sc.linalg.eig(self.synchronization["graphs"][0].inLaplacian, left=True)
            omega_hat = vl[0, :] / vl[0, :].sum()  # if vl[:, 0].sum()!= 0 else vl[:, 0]
            # get function to eval the initial value of the reference system ~~~~~~~~~~~~
            X_tot = ca.SX.sym('X_tot',
                           self.synchronization["systems"][0].state_matrix.shape[0],
                           self.synchronization["graphs"][0].inLaplacian.shape[0])
            x0_ref = np.zeros([self.synchronization["systems"][0].state_matrix.shape[0], 1])

            for index, i_comp in enumerate(omega_hat):
                x0_ref = x0_ref + np.real(i_comp) * X_tot[:, index]
            get_init_ref = ca.Function('get_reference_x0',
                                       [ca.reshape(X_tot, X_tot.size(1) * X_tot.size(2), 1)],
                                       [x0_ref], ['X_tot'], ['x0_ref_val'])
            # build reference matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Aref, Cref = (self.synchronization["systems"][0].state_matrix, self.synchronization["systems"][0].output_matrix)
            flag_identiacal = True
        else:
            # get all system matrices of the individual systems ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            A_org = [i_sys.state_matrix for i_sys in self.synchronization["systems"]]
            # convert system matrices into Jordan form ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            A_Jordan = self.get_Jordan(A_org)
            # get data from the Jordan matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            spectrum_data = self.get_system_spectrum_data(A_Jordan)
            # build reference matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Aref, Cref = MultiAgent.get_system_intersection(spectrum_data)
            get_init_ref = ca.DM(np.ones([Aref.shape[0], 1]))
            flag_identiacal = False

        # build reference system ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ref_sys = System()
        ref_sys.state_matrix = Aref
        ref_sys.input_matrix = np.zeros([Aref.shape[0], 1])
        ref_sys.output_matrix = Cref
        ref_sys.initial_state = get_init_ref
        self.synchronization['reference_system'] = ref_sys

        has_intersection = True if Aref.tolist()[0].__len__() > 0 else False
        return flag_identiacal, has_intersection

    def is_synchronized(self, identicalSystem=True):
        """ check if the networked system is synchronizable"""
        if identicalSystem:
            A = self.synchronization["systems"][0].state_matrix
            B = self.synchronization["systems"][0].input_matrix
            C = self.synchronization["systems"][0].output_matrix

            for i_lambda in np.linalg.eig(self.synchronization["graphs"][0].inLaplacian)[0]:
                if np.real(i_lambda).__round__(3).__abs__() != 0.0:
                    A_bar = A - self.control_val*i_lambda*B@C
                    lambda_A_bar = np.linalg.eig(A_bar)[0].real
                    if np.any(lambda_A_bar >= 0):
                        return False
                    else:
                        return True
        else:
            return None

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
                pSg.background_fill_color = basic.background_color()
                pSg.step(self.TimeGrid, list(np.array(sigma_span_tot.T)),  line_width=2, line_color='black')
                pSg.yaxis.axis_label = 'sigma'
                pSg.xaxis.axis_label = 'time'
                pSg.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))

            pS = figure(title='system states')
            pS.background_fill_color = basic.background_color()
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
            pSS.background_fill_color = basic.background_color()
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

    def plot_synchronize(self, sigma=None, Tspace=True):
        """  simmulate the synchronization problem of a multi-agents system """
        # **************************************************
        if sigma is not None:
            flag_switch = True
        else:
            flag_switch = False
        # **************************************************
        if Tspace:
            if flag_switch:
                T_sample = sigma.data["xdata"][1]
                t_span_tot = np.array([0])
                x_span_tot = self.initial_state
                sigma_span_tot = sigma.signal(0)
                for k in range(sigma.data["xdata"].__len__() - 1):
                    t_span_loc = np.linspace(sigma.data["xdata"][k], sigma.data["xdata"][k + 1], self.Ngrid)
                    t_span_tot = np.concatenate((t_span_tot, t_span_loc[1:]))
                    index = sigma.signal(t_span_loc[0]).__int__()
                    x_span_loc = self.__sim_networked_system(time_hist=t_span_loc, index=index)
                    sigma_span_loc = sigma.signal(t_span_loc)
                    self.initial_state = x_span_loc[:, -1]
                    x_span_tot = ca.horzcat(x_span_tot, x_span_loc[:, 1:])
                    sigma_span_tot = ca.horzcat(sigma_span_tot, sigma_span_loc[1:].T)
            else:
                t_span_tot = np.linspace(0, self.t_final, self.Ngrid)
                ref_sys_val = False
                x_span_tot = self.__sim_networked_system(time_hist=t_span_tot,
                                                         para_sym=self.synchronization["feedback_gain"],
                                                         para_val=self.control_val,
                                                         ref_sys=ref_sys_val)
                if ref_sys_val:
                    yr_tot = self.synchronization["reference_system"].output_matrix @ x_span_tot

            self.TimeGrid = t_span_tot
            self.StateHistory = x_span_tot
            self.OutputHistory = self.synchronization["output_projection"](x_span_tot)
            self.InputHistory = self.synchronization["input_projection"](x_span_tot, self.control_val)
            x_span_loc = [iprojection(x_span_tot) for iprojection in self.synchronization["state_projection"]]

        tab = []
        help_min_tot = 0
        help_max_tot = 0
        pOtot = figure(title='network output')
        pOtot.background_fill_color = basic.background_color()
        for i_sys in range(self.synchronization["state_projection"].__len__()):
            i_color = next(basic.colors_generator())
            nxi = self.synchronization["state_projection"][i_sys].nnz_out()

            pI = figure(title='agents input')
            pI.background_fill_color = basic.background_color()
            help_min = np.array(self.InputHistory[i_sys, :].T).min() - 0.5
            help_max = np.array(self.InputHistory[i_sys, :].T).max() + 0.5
            pI.line(self.TimeGrid, list(np.array(self.InputHistory[i_sys, :].T)), line_width=3)
            pI.yaxis.axis_label = 'input'
            pI.xaxis.axis_label = 'time'
            pI.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
            pI.y_range = Range1d(help_min, help_max)

            pO = figure(title='agents output')
            pO.background_fill_color = basic.background_color()
            help_min = np.array(self.OutputHistory[i_sys, :].T).min() - 0.5
            help_max = np.array(self.OutputHistory[i_sys, :].T).max() + 0.5
            pO.line(self.TimeGrid, list(np.array(self.OutputHistory[i_sys, :].T)), line_width=3)
            pO.yaxis.axis_label = 'output'
            pO.xaxis.axis_label = 'time'
            pO.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
            pO.y_range = Range1d(help_min, help_max)

            pS = figure(title='agents states')
            pS.background_fill_color = basic.background_color()
            help_min = 0
            help_max = 0
            for i_state in range(nxi):
                pS.line(self.TimeGrid, list(np.array(x_span_loc[i_sys][i_state, :].T)), line_width=3)
                if np.array(x_span_loc[i_sys][i_state, :].T).min() - 0.2 < help_min:
                    help_min = np.array(x_span_loc[i_sys][i_state, :].T).min() - 0.2
                if np.array(x_span_loc[i_sys][i_state, :].T).max() + 0.2 > help_max:
                    help_max = np.array(x_span_loc[i_sys][i_state, :].T).max() + 0.2
            pS.yaxis.axis_label = 'states'
            pS.xaxis.axis_label = 'time'
            pS.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
            pS.y_range = Range1d(help_min, help_max)
            if np.array(self.OutputHistory[i_sys, :].T).min() - 0.5 < help_min_tot:
                help_min_tot = np.array(self.OutputHistory[i_sys, :].T).min() - 0.5
            if np.array(self.OutputHistory[i_sys, :].T).max() + 0.5 > help_max_tot:
                help_max_tot = np.array(self.OutputHistory[i_sys, :].T).max() + 0.5
            pOtot.line(self.TimeGrid, list(np.array(self.OutputHistory[i_sys, :].T)), line_width=3,
                       legend_label="output " + str(i_sys + 1),
                       line_color=i_color)
            tab.append(Panel(child=gridplot([[pI], [pS], [pO]], width=500, height=200), title="agent " + str(i_sys + 1)))

        pOtot.yaxis.axis_label = 'output'
        pOtot.xaxis.axis_label = 'time'
        pOtot.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
        pOtot.y_range = Range1d(help_min_tot, help_max_tot)
        pOtot.legend.location = "top_left"
        tab.append(Panel(child=gridplot([[pOtot]], width=500, height=400), title="network output"))

        show(Tabs(tabs=tab))

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