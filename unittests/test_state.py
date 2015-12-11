from unittest import TestCase
from models.state import State
from models.field import Field
from models.predator import Predator
from models.prey import Prey

__author__ = 'Fritjof'


class TestState(TestCase):
    def test_predators_have_collided(self):
        f = Field(11, 11)
        predator1 = Predator(id="Plato", location=(1, 1))
        predator2 = Predator(id="Pythagoras", location=(1, 1))
        chip = Prey(id="Kant", location=(5, 5))
        f.add_player(predator1)
        f.add_player(predator2)
        f.add_player(chip)
        s = State.state_from_field(f)
        self.assertTrue(s.predators_have_collided())
        self.assertTrue(s.prey_is_caught() == False)