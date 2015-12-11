from unittest import TestCase
from models.value_dict import Value_dict
from models.state import State

__author__ = 'Fritjof'


class TestValue_dict(TestCase):
    def test_set_value(self):
        v = Value_dict(1.0)
        s = State(relative_distances=((1, 1), (2, 3)))
        a = (1, 0)
        v[s,a] = 10
        self.assertEqual(v[s,a], 10)