from models.policies.probabilistic_policy import ProbabilisticPolicy
from models.plearners.plearner import Plearner
from collections import defaultdict
from models.policies.mixed_policy import Mixed_policy
from pulp import *
import numpy as np
class MiniMaxQPlearner(Plearner):
    def __init__(self, field, agent,gamma=0.9,epsilon=0.,end_alpha=0.01,num_episodes=10**5):
        self.policy = Mixed_policy(field,agent,epsilon=epsilon)
        super(MiniMaxQPlearner, self).__init__(self.policy, field, agent)
        self.q_value = defaultdict(lambda: 1)
        self.value = defaultdict(lambda: 1)
        self.alpha = 1.0
        self.gamma = gamma
        # Set the decay parameter such that at the end of num_steps episodes it will be end_alpha
        self.decay = decay=10**((np.log(end_alpha)/(num_episodes)))

    def update(self, old_state, new_state, actions, rewards):
        opponent = self.field.get_opponent(self.agent)
        alpha = self.alpha
        gamma = self.gamma
        reward = rewards.get(self.agent)
        action = actions.get(self.agent)
        op_act = actions.get(opponent)

        # Update Q[s,a,o]
        self.q_value[old_state.rep(),action,op_act] = \
            (1-alpha) * (self.q_value[old_state.rep(),action,op_act]) \
            + alpha * (reward + gamma*self.value[new_state])

        # Use linear programming to find pi(s,.) such that:
        # pi[s,.] = argmax{pi'[s,.], min{o',sum{a', pi[s,a'] * Q[s,a',o']}}}}
        state_policy, new_value = self.solve_minmax(opponent, old_state, new_state)
        # Update policy
        for i,action in enumerate(self.agent.get_actions()):
            self.policy.value[old_state,action] = state_policy[i]

        self.value[old_state.rep()] = new_value
        self.alpha = self.alpha * self.decay



    def solve_minmax(self,opponent, old_state, new_state):
        n_actions = len(self.agent.get_actions())
        old_p = [self.policy.value[old_state,action] for action in self.agent.get_actions()]
        q_matrix = [[self.q_value[old_state.rep(),action,op_act] for action in self.agent.get_actions()] for op_act in opponent.get_actions()]

        p = pulp.LpVariable.dicts("p", range(n_actions),lowBound=0, upBound=1)
        V = LpVariable("V")
        lp_prob = pulp.LpProblem("Minmax Problem", pulp.LpMaximize)
        lp_prob += V, "Maximize_the_minimum"

        # Add minimizing contraint for every opponent action
        for i in range(n_actions):
            label = "Min_constraint_%d" % i
            val_sum = pulp.lpSum([p[j] * q_matrix[i][j] for j in range(n_actions)])
            condition = val_sum >= V
            lp_prob += condition, label

        cond = pulp.lpSum([p[i] for i in range(n_actions)]) == 1
        lp_prob += cond, "probs_sum_up_to_1"

        # lp_prob.writeLP("MinmaxProblem.lp")  # optional
        lp_prob.solve()

        # print "Status:", pulp.LpStatus[lp_prob.status]
        # print old_state,action,op_act,new_state
        # for v in lp_prob.variables():
            # print v.name, "=", v.varValue
        # for i,action in enumerate(self.agent.get_actions()):
        #     print action, lp_prob.variables()[i+1].varValue
        # print "Total Cost =", pulp.value(lp_prob.objective)

        return [lp_prob.variables()[i+1].varValue for i,action in enumerate(self.agent.get_actions())], lp_prob.variables()[0]