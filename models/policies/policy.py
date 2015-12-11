import random
import collections
import numpy as np
from models.value_dict import Value_dict


class Policy(object):
    """
    The policy describes the probabilities for a player to move in any direction.
    """
    def __init__(self, field, agent, value_init):
        self.agent = agent
        self.field = field
        self.value = Value_dict(default_value=value_init)

    def pick_next_action(self, state):
        """
        selects an action according to the action probability distribution of the policy
        :param state:
        :param actions:
        :param style:
            - probabilistic: select a random action with equal probability. This is the default.
            - greedy: select thee action that yields the highest immediate reward. Optional parameter epsilon makes an
              epsilon-greedy selection (random non-greedy action with probability epsilon)
            - q-greedy: like greedy, but based on Q-Values (state-action values) instead of state-values. Optional
              parameter epsilon.
            - max_value: select the action that yields the highest value (immediate reward + gamma*value_of_next_state)
              requires additional parameter gamma.
            - softmax: varies the action probabilities as a graded function of estimated value. The greedy action is
              still given the highest selection probability, but all the others are ranked and weighted according to
              their value estimates. See section 2.3 in Sutton&Barto. Requires additional parameter tau.
        :return: action tuple according to policy
        """
        pass

    def pick_random_action(self):
        return random.choice(self.agent.get_actions())

    def set_value(self, state, action, value):
        self.value[(state, action)] = value

    def get_value(self, state, action):
        return self.value[(state, action)]