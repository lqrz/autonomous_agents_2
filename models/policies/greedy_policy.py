from models.policies.policy import Policy
import random
import numpy as np


class GreedyPolicy(Policy):
    """
    The greedy policy
    """

    def __init__(self, field=None, agent=None, value_init=None, epsilon=.0, gamma=.0, q_value_select=False):
        super(GreedyPolicy, self).__init__(field=field, agent=agent, value_init=value_init)
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_value_select = q_value_select

    def select_states(self, max_val, comp_val, val_sel, selected_states):
        """
        choose the highest value between max_val and comp_val
        if the max_val is higher return the current value, do nothing else
        if the comp_val is higher, delete the content of the current selected_states and add val_sel to the list
        if the values are equal, just add val_sel to the current list
        :param max_val: the current maximum value
        :param comp_val: the value to compare the current value with
        :param selected_states:
        :param val_sel: the state, action pair that should be added to the selection
        :return: the maximum value
        """
        if max_val > comp_val:
            return max_val
        elif max_val < comp_val:
            del selected_states[:]
        selected_states.append(val_sel)
        return comp_val

    def select_best_state_action(self, selected_states):
        """
        select the best value from the list of selected state representations with teh highest value
        :param selected_states: the list of selected state representations with the highest value
        :return: the best state, action pair
        """
        if len(selected_states) > 0:
            best_state_rep, best_action = random.choice(selected_states)
        else:
            raise ValueError("no state found")
        return best_state_rep, best_action

    def calc_tmp_val(self, next_action, next_state, state, tmp_val):
        """
        calculate the temporary value for the selection criteria with a discount factor (gamma)
        :param next_action: the next action
        :param next_state: the next state
        :param state: the state
        :param tmp_val: the current temporary value
        :return: the new temporary value
        """
        tmp_prob = self.get_probability(state, next_state, next_action)
        tmp_rew = self.field.get_reward(next_state, self.agent) + self.gamma * self.get_value(next_state, None)
        tmp_val += tmp_prob * tmp_rew
        return tmp_val

    def pick_next_action(self, state):
        """
        picks the next action according to a greedy policy:

        In case self.q_value_select is set to True, it picks the highest value
        according to the state and next_action combination.

        In case of a gamma > 0 it picks the next action according to the reward and discounted value
        of the next state

        In case gamma = 0 and q_value_select = False, the highest value given the next state will be picked.

        In case a epsilon of > 0 is given, a random action is chosen with epsilon chance.
        otherwise one of the above happens

        if there are more than 1 choices with equally high values one of them will be picked at random
        :param state: the current state
        :return: the best action for the current state
        """

        if random.random() <= self.epsilon:
            return self.pick_random_action()
        else:
            max_val = -np.inf
            selected_states = []

            for next_action in self.agent.get_actions():
                if self.q_value_select:
                    max_val = self.select_states(max_val=max_val, comp_val=self.get_value(state, next_action),
                                                 val_sel=(state.rep(), next_action), selected_states=selected_states)
                else:
                    if self.gamma > 0:
                        tmp_val = 0

                    for next_state in self.field.get_next_states(state, next_action):
                        if self.gamma > 0:
                            tmp_val = self.calc_tmp_val(next_action, next_state, state, tmp_val)
                        else:
                            max_val = self.select_states(max_val=max_val, comp_val=self.get_value(next_state, None),
                                                        val_sel=(next_state.rep(), next_action), selected_states=selected_states)
                    if self.gamma > 0:
                        max_val = self.select_states(max_val=max_val, comp_val=tmp_val,
                                                    val_sel=(next_state.rep(), next_action), selected_states=selected_states)

            best_state_rep, best_action = self.select_best_state_action(selected_states)

        if best_action not in self.agent.get_actions():
            raise ValueError("action not in legal actions of agent from State: ", state.rep(), ", NextState: ",
                         best_state_rep, ", action: ", best_action)

        return best_action