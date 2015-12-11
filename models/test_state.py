from unittest import TestCase

from models.field import Field
from models.predator import Predator
from models.prey import Prey
from models.state import State


__author__ = 'bas'


class TestState(TestCase):
    def setUp(self):
        self.field = Field(11, 11)
        predators = [Predator((0, 0)), Predator((0, 10)), Predator((10, 0))]
        prey = Prey((5, 5))
        for pred in predators:
            self.field.add_player(pred)
        self.field.add_player(prey)

    def test_state_from_field(self):
        state = State.state_from_field(self.field)
        self.assertEqual(state.relative_distances, [(5, 5), (5, -5), (-5, 5)])

    def test_terminal_functions(self):
        state = State.state_from_field(self.field)
        self.assertFalse(state.is_terminal())
        self.assertFalse(state.predators_have_collided())
        self.assertFalse(state.prey_is_caught())

        # Move prey to location of first predator
        self.field.get_prey().location = (0, 0)
        state = State.state_from_field(self.field)
        self.assertTrue(state.is_terminal())
        self.assertFalse(state.predators_have_collided())
        self.assertTrue(state.prey_is_caught())

        # Move predator 1 to location of last predator
        self.field.get_predators()[0].location = (10, 0)
        state = State.state_from_field(self.field)
        self.assertTrue(state.is_terminal())
        self.assertTrue(state.predators_have_collided())
        self.assertFalse(state.prey_is_caught())

    def test_all_states(self):
        all_states = State.all_states(self.field)
        self.assertEqual(len(all_states), 1771561)

    def test_all_states_without_terminal(self):
        states = State.all_states_without_terminal(self.field)
        self.assertEqual(len(states), 1685040)