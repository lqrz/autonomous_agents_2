from unittest import TestCase
from collections import Counter
from models.player import Player
from models.field import Field

__author__ = 'fbuettner'


class TestField(TestCase):
    def test_transition(self, n_runs=1000, rel_delta=0.05):
        """
        test if the transitions are computed correctly based on tripping probabilities.
        :param n_runs: number of runs to test, should not be too low because there is randomness involved
        :param rel_delta: relative plus/minus delta to be accepted as success
        """
        p = Player(location=(0, 0), tripping_prob=0.5)
        f = Field(11, 11)
        f.add_player(p)
        new_states = [f.transition(p, action=(-1, 0)) for i in range(n_runs)]
        c = Counter(new_states)
        print c
        self.assertAlmostEqual(c[c.keys()[0]], n_runs*p.tripping_prob, delta=rel_delta*n_runs)