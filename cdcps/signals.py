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

from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

from casadi import casadi as ca

class Signals:
    def __init__(self, signal_fun, signal_data, signal_type):
        self.signal = signal_fun
        self.data = signal_data
        self.type = signal_type

    def __call__(self, *args, **kwargs):
        return self.signal(args[0])

    def __repr__(self):
        return str(self.signal)

    def __str__(self):
        return str(self.signal)

    @classmethod
    def get_PC_Function(cls, xdata, ydata, *argv):
        """ Return a piecewise-constant function """
        default = [0 if len(argv) < 1 else argv[0]][0]
        help_exp = default
        help_var = ca.SX.sym('help')
        for i_step in range(len(xdata)):
            help_exp = ca.if_else(help_var >= xdata[i_step], ydata[i_step], help_exp)

        raw_data = {'xdata': xdata, 'ydata': ydata, 'default': default}
        return cls(ca.Function('PC_fun', [help_var], [help_exp], ['time'], ['value']), raw_data, "PC_function")

    @classmethod
    def get_PL_Function(cls, xdata, ydata, *argv):
        """ Return a piecewise-linear function """
        default = [0 if len(argv) < 1 else argv[0]][0]
        help_exp = default
        help_var = ca.SX.sym('help')
        for i_step in range(xdata.shape[0]):
            if xdata[i_step, 0] == xdata[i_step, 1]:
                help_exp = ca.if_else(help_var >= xdata[i_step, 0], ydata[i_step, 1], help_exp)
            else:
                m = (ydata[i_step, 1] - ydata[i_step, 0]) / (xdata[i_step, 1] - xdata[i_step, 0])
                n = ydata[i_step, 0] - m * xdata[i_step, 0]
                help_exp = ca.if_else(help_var >= xdata[i_step, 0], m * help_var + n, help_exp)
        help_exp = ca.if_else(help_var >= xdata[i_step, 1], default, help_exp)

        raw_data = {'xdata': xdata, 'ydata': ydata, 'default': default}
        return cls(ca.Function('PC_fun', [help_var], [help_exp], ['time'], ['value']), raw_data, "PL_function")

    @classmethod
    def get_EDS_Function(cls, parameter):
        """ Return of an exponentially damped sine function """
        if parameter.get('u0') is None:
            parameter['u0'] = 1
        if parameter.get('omega') is None:
            parameter['omega'] = 0
        if parameter.get('phi') is None:
            parameter.update({'phi': 0})
        if parameter.get('delta') is None:
            parameter.update({'delta': 0})

        help_var = ca.SX.sym('help')
        help_exp = parameter['u0'] * ca.exp(parameter['delta'] * help_var) * ca.sin(
            parameter['omega'] * help_var + parameter['phi'])
        return cls(ca.Function('EDS_fun', [help_var], [help_exp], ['time'], ['value']), parameter, "EDS_function")

    @classmethod
    def get_convolution(cls, G, U, t_start=0, t_final=10, step=0.1):
        """ get the convolution of the signals G and U """
        tau = ca.MX.sym('tau')
        t = ca.MX.sym('t')

        ode = dict(t=tau, x=ca.MX.sym('x'), p=t, ode=G(t-tau)*U(tau))
        tau_span = np.arange(t_start, t_final, step)
        x_span = 0
        for k in range(len(tau_span) - 1):
            F = ca.integrator('F', 'cvodes', ode, dict(t0=tau_span[k], tf=tau_span[k+1]))
            x_span = F(x0=x_span, p=t)["xf"]

        return cls(ca.Function('conv', [t], [x_span], ['t'], ['G*U']), dict(G=G, U=U), "convolution")

    def show_convolution(self, t_span):
        """ This function plot the convolution of two signals """

        if self.type == "convolution":
            Gval = self.data.get("G")(t_span).full()
            Uval = self.data.get("U")(t_span).full()
            Yval = self.signal(t_span).full()

            fig, ax = plt.subplots()
            ax.axis([min(t_span), max(t_span),
                     min([min(Gval), min(Uval), min(Yval)])-0.1,
                     max([max(Gval), max(Uval), max(Yval)])+0.1])
            ax.plot(t_span, Uval, label='input u') #Gval
            ax.grid(color='lightgray', linestyle='--', linewidth=1)
            l1, = ax.plot([], [], label='weight g')
            l2, = ax.plot([], [], label='output y')
            l3, = ax.plot([], [],'k--')
            ax.legend()
            def animate(i):
                # l1.set_data(t_span, U(t_span - t_span[i]).full())
                l1.set_data(t_span, self.data.get("G")(t_span[i]-t_span).full())
                l2.set_data(t_span[:i], Yval[:i])
                l3.set_data([t_span[i], t_span[i]], [-0.1, 1])

            ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(t_span))
            return HTML(ani.to_jshtml())
