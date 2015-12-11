__author__ = 'fbuettner'
import numpy as np
import random
import copy
from collections import defaultdict
from models.plearners.plearner import Plearner
from models.policies.mixed_policy import Mixed_policy
from models.value_dict import Value_dict


class Wolf_phc(Plearner):
    def __init__(self, policy, field, agent, alpha=0.1, gamma=0.9, delta_l=3.0, delta_w=1.0):
        """
        Win-or-Learn-Fast Policy Hill Climbing
        :param alpha: learning rate
        :param gamma: discount factor
        :param delta_l: step size for losing
        :param delta_w: step size for winning (must be smaller than dl)
        :return:
        """
        if delta_w >= delta_l:
            raise ValueError("Parameter dw must be smaller than parameter dl")
        super(Wolf_phc, self).__init__(policy, field, agent)
        self.average_policy = copy.deepcopy(self.policy)
        self.state_visit_counts = defaultdict(lambda: 0)
        self.alpha = alpha
        self.gamma = gamma
        self.dl = delta_l
        self.dw = delta_w
        self.q_values = Value_dict(default_value=0.0)

    def init_plearner(self, **kwargs):
        self.state_visit_counts[self.field.state] += 1

    def pick_next_action(self, state):
        return self.policy.pick_next_action(state)

    def update(self, old_state, new_state, actions, rewards):
        """
        Update the values after receiving a new state and reward from the field.
        1. update Q-values for previously selected state-action pair
        2. update estimate of average policy
        3. update policy with step size towards the best Q-value
        :param old_state:
        :param new_state:
        :param actions:
        :param rewards:
        :return:
        """
        action = actions.get(self.agent)
        reward = rewards.get(self.agent)
        # compute Q value for old state-action pair
        self.q_values[old_state, action] = self.compute_q_value(old_state, new_state, action, reward)

        # increase visit count for old state
        self.state_visit_counts[old_state] += 1

        # update estimate of average policy for all actions b and old_state
        for b in self.agent.actions:
            self.average_policy.value[old_state, b] += (1.0 / self.state_visit_counts[old_state]) * (
                self.policy.value[old_state, b] - self.average_policy.value[old_state, b])

        # update policy(s, a) and constrain it to legal probability distribution
        if action == self.max_action_value_for_q(old_state, return_arg=True):
            # was the taken action the winning action?
            update_delta = self.choose_delta(old_state)
        else:
            update_delta = -(self.choose_delta(old_state) / (len(self.agent.actions) - 1.0))
        self.policy.value[old_state, action] += update_delta

    def compute_q_value(self, old_state, new_state, action, reward):
        """
        Compute the Q-value for a state-action pair given the new state and reward:
        Q(s,a) = (1-alpha)*Q(s,a) + alpha*(reward + gamma*max(a')Q(s',a'))
        :return: new Q-value
        """
        result = (1 - self.alpha) * self.q_values[old_state, action] + self.alpha * (
            reward + self.gamma * self.max_action_value_for_q(new_state))
        return result

    def max_action_value_for_q(self, state, return_arg=False):
        """
        search all action values for a given state and return the maximum value or maximizing action (with ties broken
        arbitrarily)
        :param state: the given state
        :param return_arg: if true, return maximizing action. Else return maximum value
        :return: value or action
        """
        max = -np.inf
        argmax = []
        for a in self.agent.get_actions():
            tmp = self.q_values[state, a]
            if tmp > max:
                max = tmp
                argmax = [a]
            elif tmp == max:
                argmax.append(a)
        if return_arg:
            return random.choice(argmax)
        else:
            return max

    def choose_delta(self, state):
        """
        chooses winning or losing step size based on state
        :param state:
        :return: step size delta
        """
        policy_sum = sum(
            [self.policy.value[state, a] * self.q_values[state, a] for a in self.agent.actions])
        average_policy_sum = sum(
            [self.average_policy.value[state, a] * self.q_values[state, a] for a in self.agent.actions])
        if policy_sum > average_policy_sum:
            return self.dw
        else:
            return self.dl


    @classmethod
    def create_greedy_plearner(cls, field, agent, epsilon=0.01, learning_rate=1,
                               discount_factor=0.9):
        """
        Wrapper method for creating WoLF-PHC plearner from parameters
        :param field:
        :param agent:
        :param epsilon: exploration probability
        :param learning_rate: Q-learning rate
        :param discount_factor: Q-learning discount factor
        :return:
        """
        return Wolf_phc(
            policy=Mixed_policy(field=field, agent=agent, epsilon=epsilon),
            field=field, agent=agent, alpha=learning_rate, gamma=discount_factor)