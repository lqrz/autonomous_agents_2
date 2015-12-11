__author__ = 'fbuettner'
from collections import defaultdict
import random
import numpy as np
from models.policies.policy import Policy
from models.prey import Prey

class Mixed_policy(Policy):
    def __init__(self, field=None, agent=None, epsilon=.0):
        value_init = 1.0 / len(Prey(None,None).get_actions())
        super(Mixed_policy, self).__init__(field=field, agent=agent, value_init=value_init)
        self.epsilon = epsilon

    def get_probability_mapping(self, state):
        prob_map = [(self.value[state, action], action) for action in self.agent.actions]
        return prob_map

    def get_probability_mapping_dict(self, state):
        return {action: self.value[state, action] for action in self.agent.actions}

    def pick_next_action(self, state):
        """
        Picks the next action in a state according to a mixed strategy:
        with probability epsilon, a random action is selected.
        else, the action is selected according to the probability mapping in that state.
        In learning, the probability mapping is changed towards the highest valued action with a specified learning rate.
        Note that here, the plearner keeps the Q-values.
        :param state: the current state
        :return: action tuple
        """
        if random.random() <= self.epsilon:
            return self.pick_random_action()
        else:
            move = random.random()
            # select the action that belongs to random move value
            probability = 0.0
            prob_map = self.get_probability_mapping(state)
            while move > probability and len(prob_map) > 0:
                prob, action = prob_map.pop()
                probability += prob
            return action