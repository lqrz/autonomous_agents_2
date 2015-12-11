from __future__ import division
import timeit
import math
import itertools
import random
import numpy as np
from representations import *
from graphics.plot import value_heatmap, value_heatmap_animated, checkerboard_table, plotIterationErrors,plotDifferences
import pickle

if __name__ == '__main__':
    field = (10., 10.)
    iterations = 20

    sample_stepsize = 1.0
    k_samples = 10
    state_rep = SquareRepresentation()

    experiment_name = "stepsize-%s_k-%s_staterep-%s_sampling-uniform" % (sample_stepsize, k_samples, state_rep.__class__.__name__,)
    with open('results/params_%s.p' % experiment_name, 'rb') as f:
        values, theta, differences, errors, thetas = pickle.load(f)


    # MAKE ANIMATED HEATMAPS WITH UNIFORM COLORBARS
    vmin = 1.0 #np.min(reduce(np.minimum, all_values))
    vmax = -.2 #np.max(reduce(np.maximum, all_values))
    value_heatmap_animated(values, field, fps=2, filename="heatmap_%s.gif" % experiment_name, vmin=vmin, vmax=vmax)
