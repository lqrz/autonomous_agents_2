from models.policies.policy import Policy
import random
import numpy as np


class SoftmaxPolicy(Policy):
    """
    The SoftMax policy
    """

    def __init__(self, field, agent, value_init=None, tau=.0):
        super(SoftmaxPolicy, self).__init__(field=field, agent=agent, value_init=value_init)
        self.tau = tau

    def pick_next_action(self, state):
            sum_of_action_values = sum([np.exp(self.get_value(state, next_action) / self.tau)
                                        for next_action in self.agent.get_actions()])
            action_probabilities = [(np.exp(self.get_value(state, next_action) / self.tau) / sum_of_action_values, next_action)
                                    for next_action in self.agent.get_actions()]
            # probabilistic style for these probabilities
            move = random.random()
            probability = 0.0
            while move > probability and len(action_probabilities) > 0:
                prob, action = action_probabilities.pop()
                probability += prob
            return action
