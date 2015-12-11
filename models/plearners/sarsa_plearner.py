import numpy as np
from models.plearners.plearner import Plearner
from models.policies.greedy_policy import GreedyPolicy
from models.policies.softmax_policy import SoftmaxPolicy


class SarsaPlearner(Plearner):

    def __init__(self, policy, field, agent, learning_rate, discount_factor):
        super(SarsaPlearner, self).__init__(policy, field, agent)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.next_action = None

    @classmethod
    def create_greedy_plearner(cls, field, agent, value_init=15, epsilon=0.1, gamma=0.0, learning_rate=0.1, discount_factor=0.9, q_value_select=True):
        """
        generator method to easily create a greedy learner
        :param field:
        :param agent:
        :param value_init:
        :param tau:
        :param learning_rate:
        :param discount_factor:
        :return:
        """
        return SarsaPlearner(policy=GreedyPolicy(field=field, agent=agent,
                                             value_init=value_init, epsilon=epsilon,
                                             gamma=gamma, q_value_select=q_value_select), field=field, agent=agent,
                         learning_rate=learning_rate, discount_factor=discount_factor)

    @classmethod
    def create_softmax_plearner(cls, field, agent, value_init=15, tau=0.1, learning_rate=0.1, discount_factor=0.7):
        """
        generator method to easily create a softmax learner
        :param field:
        :param agent:
        :param value_init:
        :param tau:
        :param learning_rate:
        :param discount_factor:
        :return:
        """
        return SarsaPlearner(policy=SoftmaxPolicy(field=field, agent=agent,
                                              value_init=value_init, tau=tau), field=field, agent=agent,
                         learning_rate=learning_rate, discount_factor=discount_factor)

    def update(self, old_state, new_state, actions, rewards):
        self.next_action = self.policy.pick_next_action(new_state)
        self.policy.set_value(old_state, actions.get(self.agent), self.compute_q_value(old_state, new_state, actions.get(self.agent), rewards.get(self.agent)))

    def compute_q_value(self, old_state, new_state, old_action, reward):
        result = self.policy.get_value(old_state, old_action) + \
                 self.learning_rate * (
                     reward
                     + self.discount_factor * self.policy.get_value(new_state,self.next_action)
                     - self.policy.get_value(old_state, old_action)
                 )
        return result

    def max_action_value_for_q(self, state):
        max = -np.inf
        for action in self.agent.get_actions():
            tmp = self.policy.get_value(state, action)
            if tmp > max:
                max = tmp
        return max

    def pick_next_action(self, state):
        if self.next_action is None:
            self.next_action = self.policy.pick_next_action(state)
        return self.next_action