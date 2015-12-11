__author__ = 'Fritjof'


class Plearner(object):
    """
    Planning/Learning object. Contains the policy and models the update step for planning/learning
    """
    def __init__(self, policy, field, agent):
        self.policy = policy
        self.field = field
        self.agent = agent
        policy.agent = agent
        policy.field = field

    def init_plearner(self, **kwargs):
        """
        propagation of player.init_player
        :param kwargs:
        :return:
        """
        pass

    def update(self, old_state, new_state, actions, rewards):
        pass

    def pick_next_action(self, state):
        return self.policy.pick_next_action(state)