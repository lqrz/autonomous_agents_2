import itertools

import numpy as np


class State(object):
    """
    Represents a state
    Responsibilities:
    - Compute terminal states
    """

    def __init__(self, relative_distances=()):
        self.relative_distances = relative_distances

    def __str__(self):
        return "State" + self.relative_distances.__str__() + ""

    def __hash__(self):
        return hash(tuple(self.relative_distances))

    def rep(self):
        return tuple(self.relative_distances)

    # def __eq__(self, other):
    #     return self.relative_distances == other.relative_distances


    @classmethod
    def state_from_field(cls, field):
        """
        Generates a state-representation given a field (which contains players that know their location)
        :param field: a field that contains predators and a prey
        :return:
        """
        predators, prey = field.get_predators(), field.get_prey()
        state = State()
        state.relative_distances = []
        for predator in predators:
            distance = tuple([int(x) for x in field.get_relative_position(predator.location, prey.location)])
            state.relative_distances.append(distance)
        state.relative_distances = tuple(state.relative_distances)
        return state

    def is_terminal(self):
        """
        Compute if this state is terminal
        :return: boolean
        """
        return self.prey_is_caught() or self.predators_have_collided()

    def predators_have_collided(self):
        """
        Compute if any pair of agent in this state have collided
        :return: boolean
        """
        for distance in self.relative_distances:
            if self.relative_distances.count(distance) > 1:
                return True
        return False

    def prey_is_caught(self):
        """
        Compute if the prey is caught in this state representation
        :return: boolean
        """
        for distance in self.relative_distances:
            if distance == (0, 0):
                return True
        return False

    @classmethod
    def all_states(cls, field):
        """
        Returns a list of all possible states
        :param field: A field with a given set of predators
        :return: list of states
        """
        x_lim = int(np.floor(field.width / 2))
        y_lim = int(np.floor(field.height / 2))
        # Make a combination of all possible distances between two agents
        all_distances = itertools.product(range(-1 * x_lim, x_lim + 1), range(-1 * y_lim, y_lim + 1))

        predator_count = len(field.get_predators())

        # Make a combination of all_distances for every predator, and convert to a list of State objects
        result = []
        for state_rep in itertools.product(all_distances, repeat=predator_count):
            result.append(State(list(state_rep)))

        return result

    @classmethod
    def all_states_without_terminal(cls, field):
        """
        Returns a list of all possible states except terminal states
        :param field: A field with a given set of predators
        :return: list of states
        """
        return [state for state in cls.all_states(field) if not state.is_terminal()]