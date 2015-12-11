from random import randint


class Player(object):
    """
    superclass for predator and prey
    implements common properties like position
    """

    def __init__(self, id, location=(0.0, 0.0), actions=[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
                 plearner=None, tripping_prob = 0.0):
        self.location = location
        self.actions = actions
        self.plearner = plearner
        self.tripping_prob = tripping_prob
        self.id = str(id)

    def init_player(self, **kwargs):
        """
        call this when the field is ready. May be needed for plearners that initialize values
        over the complete state space (which is not available until all players have been added)
        :param kwargs: depends on the plearner type
        :return:
        """
        self.plearner.init_plearner(**kwargs)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id and self.location == other.location

    def get_actions(self):
        """
        returns the possible actions for the player
        :return: the possible actions
        """
        return list(self.actions)

    def act(self, state):
        """
        update the location according to the action in the policy
        :return: nothing
        """
        if self.plearner is None:
            print "No Plearner set for Agent, ", self
            raise
        else:
            action = self.plearner.pick_next_action(state)
            return action

    def update(self, old_state, new_state, actions, rewards):
        """
        This function gets the new state and a reward from the field after picking an action.
        Then it uses these values to update the player's policy.
        :param old_state:
        :param new_state:
        :param actions:
        :param rewards:
        :return:
        """
        self.plearner.update(old_state, new_state, actions, rewards)

    def get_index(self):
        # TODO: Check if working correctly
        return self.field.players.index(self)
