from braket.ahs.atom_arrangement import AtomArrangement

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
import os
import sys

from quera_ahs_utils.plotting import plot_avg_density

def test_plot_avg_density():
    # create register
    register = AtomArrangement()

    # Intentionally non-rectilinear ordering of coordinates
    pos_list = [
        (6.7e-06, 1.34e-05),
        (1.34e-05, 0.0),
        (0.0, 0.0),
        (1.34e-05, 6.7e-06),
        (6.7e-06, 6.7e-06),
        (0.0, 1.34e-05),
        (6.7e-06, 0.0),
        (1.34e-05, 1.34e-05),
        (0.0, 6.7e-06)
    ]

    for pos in pos_list:
        register.add(pos)

    # hardcoded simulation densities
    densities = np.array([0.0075, 0.98, 0.98, 0.01, 0.96, 0.98, 0.0075, 0.9825, 0.01])

    # create matplotlib plot
    fig, ax = plt.subplots()

    # generate plot on custom axis
    plot_avg_density(densities, register, custom_axes=ax)

    # save plot with date and time to file
    now = datetime.now()
    # NOTE: You CANNOT store the date as any format with slashes, this gets misinterpreted as a filepath!
    date_and_time = now.strftime("%m-%d-%Y %H:%M:%S")
    filename = "aux_files/" + date_and_time + " " + "plot-avg-density-test.png"
    fig.savefig( filename, format="png")

test_plot_avg_density()