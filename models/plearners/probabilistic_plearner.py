from models.policies.probabilistic_policy import ProbabilisticPolicy
from models.plearners.plearner import Plearner
__author__ = 'xtroce'


class ProbabilisticPlearner(Plearner):
    def __init__(self, field, agent):
        policy = ProbabilisticPolicy(field=field, agent=agent)
        super(ProbabilisticPlearner, self).__init__(policy, field, agent)

    def update(self, old_state, new_state, actions, rewards):
        #do nothing here since the policy does not change at all
        pass

