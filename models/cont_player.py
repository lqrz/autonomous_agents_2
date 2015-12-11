__author__ = 'fbuettner'
from player import Player
import math
import itertools
import numpy as np

class ContPlayer(Player):
    """
    superclass for predator and prey
    implements common properties like position
    """

    def __init__(self, id, location=(0, 0), actions=[(0.1, 1.5), (0, 2*math.pi)], sampling_steps=0.01,
                 plearner=None, tripping_prob = 0.0):
        self.location = location
        self.actions = actions
        self.plearner = plearner
        self.tripping_prob = tripping_prob
        self.sampling_step = sampling_steps
        self.id = str(id)

    def sample_action(self):
        lamb = np.arange(self.actions[0][0], self.actions[0][1], self.sampling_step)
        theta = np.arange(self.actions[1][0], self.actions[1][1], self.sampling_step)
        return [(math.cos(t)*l, math.sin(t)*l) for (l, t) in itertools.product(lamb, theta)] + [(0.0, 0.0)]

    def get_actions(self):
        return self.sample_action()

if __name__ == '__main__':
    cp = ContPlayer("bla", sampling_steps=0.1)
    for c in cp.sample_action():
        print c