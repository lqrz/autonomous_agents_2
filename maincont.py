from __future__ import division
import timeit
import math
import itertools
import random
import numpy as np
from representations import *
from graphics.plot import value_heatmap, value_heatmap_animated, checkerboard_table, plotIterationErrors,plotDifferences
import pickle


def discretize_pred_actions(field):
    max_radius = 1.5
    angle_steps = 12
    distance_steps = 12
    actions = [(0., 0.)]
    angle_stepsize = ((2 * math.pi) / angle_steps)

    for radius in np.arange(0.1, max_radius, 1.4 / distance_steps):
        for angle in np.arange(0, 2 * math.pi, angle_stepsize):
            move_x = math.cos(angle) * radius
            move_y = math.sin(angle) * radius
            actions.append([move_x, move_y])
    return actions


def toroidalify(field, state):
    state = np.array([state[0] % field[0], state[1] % field[1]])
    return state


def value(new_stoc_state, old_theta, state_rep):
    return np.dot(old_theta.T, state_rep.rep(new_stoc_state))


def reward(new_stoc_state):
    if np.sqrt(np.sum([(c - (field[0] / 2)) ** 2 for c in new_stoc_state])) <= 1.0:
        return 1.0
    return 0.0


def sample_states_randomly(field, prey_loc, sample_size=100, ):
    """
    Samples states within field randomly. The last sample is the goal state (5.0, 5.0).
    :param field: tuple of field size e.g. (10,10)
    :param prey_loc: coordinates of the prey which will always be added as a sample point
    :param sample_size: number of samples
    :return: list of sampled states [(1.5, 2.3), (5.2, 7.8), ...]
    """
    sampled_states = [(random.uniform(0, field[0]), random.uniform(0, field[1])) for _ in range(sample_size - 1)] \
                     + [prey_loc]
    return sampled_states


def sample_states(field, sample_stepsize):
    """
    We represent the state as the relative distance, incremented by 5 on both axis
    So 5,5 means you are on the prey, 0,0 is you're 5 units away from the pray in both dimensions
    10,10 and 0,0 are the same relative distances on the toridal grid.
    This allows us to use modulo without any issues.
    Essentially we have changed the problem to where you have to move to 5,5, but everytime you move
    the prey movement adds some noise to your own movement.
    :param field:
    :param sample_stepsize:
    :return:
    """
    # TODO: make different functions for sampling
    x_range, y_range = np.arange(0, field[0], sample_stepsize), np.arange(0, field[1], sample_stepsize)
    sampled_states = list(itertools.product(x_range, y_range))
    if (5.0,5.0) not in sampled_states:
        sampled_states.append((5.0,5.0))
    return sampled_states


def getClosestDistance(x, y):
    '''
    Converts coordinates x, y to relative distance. The state is the distance to the prey,
    the distance may turn from positive to negative if the fastest route is by going around the board.
    '''
    global field
    return ((x + field[0] / 2) % field[0] - field[0] / 2, (y + field[1] / 2) % field[1] - field[1] / 2)


def getStates(cell_size=1.0):
    '''
    Returns the absolute state space.
    '''
    steps = (field[0] / cell_size) + 1, (field[1] / cell_size) + 1
    # return set([self.getClosestDistance(x-a,y-b) for x in range(self.field[0]) for y in range(self.field[1]) for a in range(self.field[0]) for b in range(self.field[1])])
    return set([(x, y) for x in np.linspace(0., field[0], steps[0]) for y in np.linspace(0., field[1], steps[1])])


def update_value_function(sampled_states, old_theta, field, actions, state_rep, discount_factor, k_samples=10, ):
    sampled_values = compute_exact_values(sampled_states, actions, k_samples, discount_factor, old_theta)
        # print "Sampling state: %s" % str(old_state)
    new_theta = np.linalg.lstsq(state_rep.rep(np.array(sampled_states)), sampled_values)
    return new_theta

def compute_exact_values(sampled_states, actions, k_samples, discount_factor, old_theta):
    sampled_values = []
    for old_state in sampled_states:
        old_state_reward = reward(old_state)
        q_of_as = []
        for action in actions:
            # Compute the deterministic state (if the prey wouldnt move)
            new_det_state = (np.array(old_state) + np.array(action))
            new_det_state = toroidalify(field, new_det_state)
            # Approximate Q value for taking action in old_state:
            qa = 0
            # Now add the noise of the prey moving around and sample from this
            for i in range(k_samples):
                prey_movement = np.random.normal(loc=0.0, scale=1.0, size=2)
                new_stoc_state = toroidalify(field, new_det_state + prey_movement)
                qa += old_state_reward + discount_factor * value(new_stoc_state, old_theta, state_rep)
            qa = qa / k_samples
            q_of_as.append(qa)
        sampled_values.append(max(q_of_as))  # Pick the maximum action
    return sampled_values

def compute_fitted_values(shape, state_rep, theta):
    values = np.zeros(shape)
    # for i, x in enumerate(np.linspace(0., 10., 50)):
    #     for j, y in enumerate(np.linspace(0., 10., 50)):
    for i, x in enumerate(np.linspace(0., field[0], 50)):
        for j, y in enumerate(np.linspace(0., field[1], 50)):
            values[i, j] = value(np.array([x, y]), theta, state_rep)
    return values


def plot_valueheatmap(theta, state_rep):
    values = compute_fitted_values(shape=(50, 50), state_rep=state_rep, theta=theta)
    value_heatmap(values, path="plot.png")


def run(state_rep, k_samples, sample_stepsize, discount_factor, iterations=20, convergence_epsilon=0.0001):
    global field
    assert field[0] == field[1]

    start_predator_loc = [0., 0.]
    start_prey_loc = [5., 5.]
    max_iterations = iterations
    value_list = []
    differences = []

    # field = (10., 10.)
    # Our states are encoded as the relative distance from pred to prey + .5*fieldsize
    actions = discretize_pred_actions(field)
    predator_loc = np.array(start_predator_loc)
    prey_loc = np.array(start_prey_loc)

    theta = state_rep.theta()

    # Choose mode of sampling here
    sampled_states = sample_states(field, sample_stepsize=sample_stepsize)

    # sampled_states = sample_states_randomly(field, sample_size=100, prey_loc=tuple(start_prey_loc))

    iterationErrors = []
    thetas = []
    for i in range(max_iterations):

        theta_old = np.copy(theta)
        theta, _, _, _ = update_value_function(sampled_states, theta_old, field, actions, discount_factor=discount_factor, state_rep=state_rep, k_samples=k_samples)

        value_list.append(compute_fitted_values(shape=(50,50), state_rep=state_rep, theta=theta))

        # get approximation error
        error, _ = evaluateApproximation(theta, state_rep)
        iterationErrors.append(error)

        difference = np.sum(np.abs(theta - theta_old))
        differences.append(difference)
        print 'Iter ', i + 1, ':', difference, theta, 'error:',error
        # if difference < convergence_epsilon:  # delta must be 0.0001 to be comparable w/ exact VI.
        #     break
        thetas.append(thetas)
    return value_list, theta, differences, iterationErrors, thetas


def evaluateApproximation(theta, state_rep):
    '''
    Evaluates an approximation value function wrt the exact V.
    Returns the sum of errors.
    '''
    global field

    values = np.zeros((field[0] + 1, field[1] + 1))
    error = 0
    baseline_theta = pickle.load(open('theta-euclid.p', 'r'))
    baseline_state_rep = EuclidRepresentation()
    for (x, y) in getStates(cell_size=1.0):
        baseline = value(np.array([x, y]), baseline_theta, baseline_state_rep)
        values[x, y] = value(np.array([x, y]), theta, state_rep)
        error += np.abs(values[x, y] - baseline)

    return error, values


if __name__ == '__main__':
    field = (10., 10.)
    iterations = 20
    k_samples = 10
    discount_factor = .5
    state_rep = EuclidRepresentation()

    for sample_stepsize  in [2.0]:
        model_params = {sample_stepsize,k_samples,state_rep.__class__.__name__,iterations,field,discount_factor}

        experiment_name = "stepsize-%s_k-%s_staterep-%s_sampling-uniform" % (sample_stepsize, k_samples, state_rep.__class__.__name__,)

        start = timeit.default_timer()

        values, theta, differences, errors, thetas = run(state_rep, sample_stepsize=sample_stepsize, k_samples=k_samples, discount_factor=discount_factor, iterations=iterations)
        results = (values, theta, differences, errors, thetas, model_params),
        with open('results/results_%s.p' % experiment_name,'wb') as f:
            pickle.dump(results,f)

        # get approximation error and approximated values
        square_error, square_eval_values = evaluateApproximation(theta, state_rep)
        print 'Approximation error: ', square_error
        print "finished after", round(timeit.default_timer() - start, 3), "seconds."
        # # display approxiation function on grid
        # checkerboard_table(square_eval_values)
        # # plot approximation error vs iteration
        # plotIterationErrors(errors)
        # # plot differences
        # plotDifferences(differences)

        # MAKE ANIMATED HEATMAPS WITH UNIFORM COLORBARS
        vmax = 1.0 #np.min(reduce(np.minimum, all_values))
        vmin = -.2 #np.max(reduce(np.maximum, all_values))
        value_heatmap_animated(values, field, fps=2, filename="heatmap_%s.gif" % experiment_name, vmin=vmin, vmax=vmax)
