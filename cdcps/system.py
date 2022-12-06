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

import random
from copy import deepcopy

from tabulate import tabulate
import scipy as sc
from scipy.linalg import block_diag

import numpy as np
from numpy.polynomial import polynomial as P

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import Range1d, Arrow, OpenHead, NormalHead, VeeHead, Panel, Tabs
from bokeh.models.annotations import Label

import control.matlab as coma
import casadi as ca


class System:
    def __init__(self):
        self.type = "continuous"
        self.category = "state_spce"
        self.state_matrix = []
        self.input_matrix = []
        self.output_matrix = []
        self.initial_state = []
        self.input_val = []
        self.control_val = []       # scaling the control law: u = control_val * k(x)
        self.Ngrid = []
        self.t_final = []
        # -------------------------------
        self.impulse_response = []
        self.control_system = dict(ss=[], G0=[], F=[], Gcl= [])
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
        # -------------------------------
        self.observable = []
        self.synchronized = []
        # -------------------------------
        self.TimeGrid = []
        self.InputHistory = []
        self.StateHistory = []
        self.OutputHistory = []
        # -------------------------------
        self.OneStepIntegrator = []
        self.OutputFunction = []
        self.TimeGridIntegrator = []
        # -------------------------------
        self.ref_point = []

    def __repr__(self):
        return str('LTI_system')

    def __str__(self):
        return str('LTI_system')

    def build_system(self):
        """  Return the one-step integrator and an integration map for the predefined time grid """
        nx = self.state_matrix.shape[0]
        x = ca.SX.sym('x', nx)
        # Create the time grid
        t_span = np.linspace(0, self.t_final, self.Ngrid)
        # Create Output Function
        Y = ca.Function('Y', [x], [self.output_matrix @ x])
        # Formulate the ODE
        if isinstance(self.input_val, ca.Function):
            time = ca.SX.sym('time')
            rhs = self.state_matrix @ x + self.input_matrix @ self.input_val(time)
            ode = dict(x=x, t=time, ode=rhs)
            # Create input grid
            u_span = self.input_val(t_span).T
            # Create solver instance
            x0 = ca.MX.sym('x0', nx)
            x_now = x0
            x_span = []
            for k in range(len(t_span) - 1):
                options = dict(t0=t_span[k], tf=t_span[k + 1])
                F = ca.integrator('F', 'cvodes', ode, options)
                x_now = F(x0=x_now)["xf"]
                x_span = ca.horzcat(x_span, x_now)
            F = []
            Ftot = ca.Function('F', [x0], [x_span], ['x0'], ['xf'])
        else:
            u = ca.SX.sym('u')
            rhs = self.state_matrix @ x + self.input_matrix@u
            ode = dict(x=x, p=u, ode=rhs)
            # Create input grid
            u_span = ca.Function('u_span', [u], [self.input_val])(t_span).T
            # Create solver instance
            options = dict(t0=0, tf=t_span[1])
            F = ca.integrator('F', 'cvodes', ode, options)
            Ftot = F.mapaccum(self.Ngrid - 1)

        self.InputHistory = u_span
        self.TimeGrid = t_span
        self.OneStepIntegrator = F
        self.OutputFunction = Y
        self.TimeGridIntegrator = Ftot

    def simulate_system(self):
        """ Return the solution of the system (state and output trajectory) """
        if isinstance(self.input_val, ca.Function):
            sol = self.TimeGridIntegrator(x0=self.initial_state)
        else:
            sol = self.TimeGridIntegrator(x0=self.initial_state, p=self.InputHistory[1:])
        x_span = ca.horzcat(self.initial_state, sol['xf'])

        self.StateHistory = x_span
        self.OutputHistory = self.OutputFunction(x_span)

    def get_discrete_system(self, T=1, simulate=False, plot=False, sigma=None):
        """ under construction -> at the moment only autonomous systems are considered """
        if sigma is not None:
            T = sigma.data["xdata"][1]
            sys_loc = System()
            sys_loc.type = "consensus_discrete"
            for i_sys in range(len(self.state_matrix)):
                sys_loc.state_matrix.append(self.get_Matrix_exponential(self.state_matrix[i_sys],T))  #sc.linalg.expm(self.state_matrix[i_sys] * T))
            sys_loc.t_final = self.t_final
            sys_loc.initial_state = self.initial_state
        else:
            sys_loc = System()
            sys_loc.type = "discrete"
            sys_loc.state_matrix = sc.linalg.expm(self.state_matrix * T)
            NX = np.shape(sys_loc.state_matrix)[0]

            Ainv = np.linalg.inv(self.state_matrix)
            Aint = Ainv @ (sc.linalg.expm(self.state_matrix*T) - np.identity(NX))
            sys_loc.input_matrix = Aint @ self.input_matrix
            sys_loc.output_matrix = self.output_matrix

            sys_loc.t_final = self.t_final
            sys_loc.initial_state = self.initial_state

            if isinstance(self.input_val, ca.Function):
                sys_loc.input_val = self.input_val(0).full()
            else:
                sys_loc.input_val = self.input_val

        if simulate or plot:
            sys_loc.TimeGrid = np.append(np.arange(0, sys_loc.t_final, T), sys_loc.t_final)
            x_span = sys_loc.initial_state
            for i in range(len(sys_loc.TimeGrid) - 1):
                Xnext = sys_loc.state_matrix @ x_span[:, -1] + sys_loc.input_matrix * sys_loc.input_val
                x_span = ca.horzcat(x_span, Xnext)
            sys_loc.StateHistory = x_span
        if plot:
            self.build_system()
            self.simulate_system()

            matrix = [[None for i in range(1)] for j in range(NX)]
            for i in range(NX):
                if i == 0:
                    pS = figure(title='system states')
                else:
                    pS = figure()
                pS.line(self.TimeGrid, list(np.array(self.StateHistory[i, :].T)), line_width=3)
                pS.dot(sys_loc.TimeGrid, list(np.array(sys_loc.StateHistory[i, :].T)), size=25, color="navy", alpha=0.5)
                pS.yaxis.axis_label = 'state ' + str(i + 1)
                pS.xaxis.axis_label = 'time'
                pS.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
                pS.y_range = Range1d(np.array(self.StateHistory[i, :].T).min() - 0.2,
                                     np.array(self.StateHistory[i, :].T).max() + 0.2)
                matrix[i][0] = pS

            grid = gridplot(matrix, width=500, height=450)
            show(grid)

        return sys_loc

    def get_weight_function(self):
        """ get the impulse response function """
        t = ca.MX.sym('t')

        # transformation of the system matrix into a diagonal form and compute fundamental matrix
        ew, ev = np.linalg.eig(self.state_matrix)
        LAMBDA = ca.diag(ca.exp(ew*t))
        PHI = ev@LAMBDA@np.linalg.inv(ev)

        # compute the weight function
        g_prime = self.output_matrix @ PHI @ self.input_matrix
        g_ext = ca.if_else(t >= 0, g_prime, 0)

        self.impulse_response = ca.Function('irf', [t], [g_ext], ['time'], ['weight_factor'])

    def get_control_system(self):
        self.control_system["ss"] = coma.ss(self.state_matrix, self.input_matrix, self.output_matrix,
                                            np.zeros((self.output_matrix.shape[0], self.input_matrix.shape[1])))
        self.control_system["G0"] = coma.tf(self.control_system["ss"])
        self.control_system["F"] = 1 + self.control_system["G0"]
        self.control_system["Gcl"] = coma.feedback(self.control_system["G0"])

    @classmethod
    def get_transfer_function(cls, num, den, type=["c", "c"]):
        # type c ... coefficient | r ... roots
        sys = cls()
        sys.category = "transfer_function"
        if type[0] == "r":
            num =  np.flip(P.polyfromroots(num))
        if type[1] == "r":
            den = np.flip(P.polyfromroots(den))

        sys.control_system["G0"] = coma.tf(num, den)
        sys.control_system["F"] = 1 + sys.control_system["G0"]
        sys.control_system["Gcl"] = coma.feedback(sys.control_system["G0"])
        return sys

    @staticmethod
    def get_Matrix_exponential(A, T):
        return sc.linalg.expm(A * T)

    def plot_system(self, *argv):
        """ Plot the input, state and trajectory """
        default = ['ISO' if len(argv) < 1 else argv[0]][0]

        nu = int(self.input_matrix.shape[1])
        nx = int(self.state_matrix.shape[0])
        ny = int(self.output_matrix.shape[0])

        nrow = max([nu, nx, ny])
        ncol = len(default)
        matrix = [[None for i in range(ncol)] for j in range(nrow)]

        i_col = 0
        if "I" in default:
            for i in range(nrow):
                if i<nu:
                    if i == 0:
                        pI = figure(title='system inputs')
                    else:
                        pI = figure()
                    pI.line(self.TimeGrid, list(np.array(self.InputHistory[i, :].T)), line_width=3)
                    pI.yaxis.axis_label = 'input ' + str(i + 1)
                    pI.xaxis.axis_label = 'time, (s)'
                    pI.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
                    pI.y_range = Range1d(np.array(self.InputHistory[i, :].T).min() - 0.2,
                                         np.array(self.InputHistory[i, :].T).max() + 0.2)
                    matrix[i][i_col] = pI
            i_col += 1

        if "S" in default:
            for i in range(nrow):
                if i < nx:
                    if i == 0:
                        pS = figure(title='system states')
                    else:
                        pS = figure()
                    pS.line(self.TimeGrid, list(np.array(self.StateHistory[i, :].T)), line_width=3)
                    pS.yaxis.axis_label = 'state ' + str(i + 1)
                    pS.xaxis.axis_label = 'time, (s)'
                    pS.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
                    pS.y_range = Range1d(np.array(self.StateHistory[i, :].T).min() - 0.2,
                                         np.array(self.StateHistory[i, :].T).max() + 0.2)
                    matrix[i][i_col] = pS
            i_col += 1

        if "O" in default:
            for i in range(nrow):
                if i < ny:
                    if i == 0:
                        pO = figure(title='system output')
                    else:
                        pO = figure()
                    pO.line(self.TimeGrid, list(np.array(self.OutputHistory[i, :].T)), line_width=3)
                    pO.yaxis.axis_label = 'output ' + str(i + 1)
                    pO.xaxis.axis_label = 'time, (s)'
                    pO.x_range = Range1d(np.min(self.TimeGrid), np.max(self.TimeGrid))
                    pO.y_range = Range1d(np.array(self.OutputHistory[i, :].T).min()-0.2, np.array(self.OutputHistory[i, :].T).max()+0.2)
                    matrix[i][i_col] = pO
            i_col += 1

        grid = gridplot(matrix, width=250, height=200)
        show(grid)

    def plot_bode(self, domain, type="ss", at_omega=1, show_omega=False, show_phase_margin=False):
        # for the plot
        if self.control_system[type]==[]:
            type = "G0"
        use_sys = self.control_system[type]

        mag, phase, omega = coma.bode(use_sys, np.logspace(domain[0], domain[1]), plot=False)
        mag_dB = 20 * np.log10(mag)
        phase_deg = 180 * phase / np.pi

        # point evaluation
        if show_omega:
            mag_p, phase_p, omega_p = coma.bode(use_sys, [at_omega, at_omega], plot=False)
            mag_p_dB = 20 * np.log10(mag_p)
            phase_p_deg = 180 * phase_p / np.pi

        # margin
        if show_phase_margin:
            gm, pm, wg, wp = coma.margin(use_sys)

        p1 = figure(title="Bode Plot", x_axis_type="log")
        p1.line(omega, mag_dB, line_width=3)
        if show_phase_margin:
            p1.circle(wp, 0, size=10, color="red", alpha=0.5)
        if show_omega:
            p1.line(omega_p, [0, mag_p_dB[0]], line_width=2, line_color='black')
        p1.yaxis.axis_label = 'Magnitude, (dB)'

        p2 = figure(x_axis_type="log")
        p2.line(omega, phase_deg, line_width=3)
        if show_phase_margin:
            p2.line(omega, -180, line_width=1)
            p2.add_layout(Arrow(end=VeeHead(size=10), line_color="blue",
                                x_start=wp, y_start=-180, x_end=wp, y_end=-180 + pm))
            p2.add_layout(Label(x=wp, y=-180 + pm, x_offset=-50, text=str(round(pm, 1)), text_baseline="top"))

        if show_omega:
            p2.line(omega_p, [0, phase_p_deg[0]], line_width=2, line_color='black')
        p2.yaxis.axis_label = 'Phase, (deg)'
        p2.xaxis.axis_label = 'Frequency, (rad/sec)'
        grid = gridplot([[p1], [p2]], width=1000, height=300)
        show(grid)

        if show_omega:
            val_at_p = [[mag_p[0], mag_p_dB[0], phase_p[0], phase_p_deg[0]]]
            print('\n Values of the Transfer Function at omega=', str(at_omega), ' (rad): \n', )
            print(tabulate(val_at_p, headers=['Magnitude\n |G|', '\n |G| (dB)', 'Phase\n (rad)', '\n \phi (deg)']))

    def plot_nyquist(self, type="ss", at_omega=1, show_omega=False, show_phase_margin=False):
        # for the plot
        if self.control_system[type] == []:
            type = "G0"
        use_sys = self.control_system[type]

        real, imag, omega = coma.nyquist(use_sys, plot=False)

        # point evaluation
        if show_omega:
            # real_p, imag_p, omega_p = coma.nyquist(system, [at_omega, at_omega], plot=False)
            point_tar = coma.evalfr(use_sys, complex(0, at_omega))
            real_p = np.real(point_tar)
            imag_p = np.imag(point_tar)

        # margin
        if show_phase_margin:
            gm, pm, wg, wp = coma.margin(self.control_system)
            point_wp = coma.evalfr(use_sys, complex(0, wp))
            real_wp = np.real(point_wp)
            imag_wp = np.imag(point_wp)

        p1 = figure(title="Nyquist Plot")
        p1.line(real, imag, line_width=3)
        p1.line(real, -imag, line_width=3, line_dash=[5, 10])
        p1.circle(-1, 0, size=10, color="red", alpha=0.5)
        p1.add_layout(Arrow(end=VeeHead(size=15), line_color="blue",
                            x_start=real[0], y_start=imag[0], x_end=real[1], y_end=imag[1]))
        p1.add_layout(Arrow(end=VeeHead(size=15), line_color="blue",
                            x_start=real[round(len(real) - 0.5 * len(real))],
                            y_start=imag[round(len(real) - 0.5 * len(real))],
                            x_end=real[round(len(real) - 0.5 * len(real)) + 1],
                            y_end=imag[round(len(real) - 0.5 * len(real)) + 1]))
        if show_phase_margin:
            p1.line(np.cos(np.linspace(0, np.pi, 100)), np.sin(np.linspace(0, np.pi, 100)), line_width=1,
                    line_dash=[5, 10], line_color='black')
            p1.line(np.cos(np.linspace(0, np.pi, 100)), -np.sin(np.linspace(0, np.pi, 100)), line_width=1,
                    line_dash=[5, 10], line_color='black')
            p1.circle(real_wp, imag_wp, size=10, color="green", alpha=0.5)
            p1.add_layout(Arrow(end=VeeHead(size=10), line_color="green",
                                x_start=0, y_start=0, x_end=real_wp, y_end=imag_wp))
        if show_omega:
            p1.add_layout(Arrow(end=VeeHead(size=10), line_color="black",
                                x_start=0, y_start=0, x_end=real_p, y_end=imag_p))

        p1.yaxis.axis_label = 'Imaginary axis'
        p1.xaxis.axis_label = 'Real axis'
        show(p1)

