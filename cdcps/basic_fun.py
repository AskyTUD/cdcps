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
import numpy as np

def vercat(*args):
    return np.array([[ii_number] for i_element in args for ii_number in i_element])

def colors_generator(firstval=0, step=1):
    # colors = itertools.cycle(inferno(self.synchronization["graphs"][0].graph.number_of_nodes()))# create a color iterator
    colorsA = ["black", "darkslategray", "slategray", "lightslategray", "dimgray", "darkgray"]
    colorsB = ["midnightblue", "navy", "steelblue", "lightslategray", "cornflowerblue", "skyblue"]
    colorsC = ["darkred", "firebrick", "indianred"]
    colorsD = ["purple", "indigo", "darkmagenta", "darkorange"]
    colorsE = ["darkgreen", "seagreen", "mediumseagreen", "darkseagreen", "darkolivegreen", "olivedrab", "teal"]
    colorstot = colorsA + colorsB + colorsC + colorsD + colorsE
    xx = firstval
    yield random.sample(colorstot, colorstot.__len__())[xx]
    xx += step

def background_color():
    return 'darkgray' #'slategray' #'darkslategray'
