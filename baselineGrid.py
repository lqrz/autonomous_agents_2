__author__ = 'lqrz'

from maincont import *


if __name__ == '__main__':
    field = (10., 10.)

    #-------- Pick State representation here --------#
    # state_rep = LinearRepresentation()
    state_rep = SquareRepresentation()
    # state_rep = PolynomialRepresentation()

    values = np.zeros((field[0] + 1, field[1] + 1))
    theta = pickle.load(open('theta.p','r'))
    for (x, y) in getStates(cell_size=1.0):
        values[x, y] = value(np.array([x, y]), theta, state_rep)

    checkerboard_table(values)
    value_heatmap_animated(values, field, fps=2)