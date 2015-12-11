import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from models.field import Field
from models.predator import Predator
from models.prey import Prey
from models.plearners.probabilistic_plearner import ProbabilisticPlearner
from models.plearners.q_plearner import QPlearner
from models.plearners.sarsa_plearner import SarsaPlearner
from models.plearners.minimax_q_plearner import MiniMaxQPlearner
from models.plearners.wolf_phc import Wolf_phc
from graphics.gui import GameFrame
from models.state import State
from models.policies.mixed_policy import Mixed_policy
from models.policies.greedy_policy import GreedyPolicy

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_stdev(a, n=3):
    result = np.zeros(len(a) - n + 1)
    for i in range(0, len(a)-n + 1):
        try:
            result[i] = np.std(a[i:(i+n)])
        except:
            print i

    # result[(len(a)-n):len(a)] = 5
    return result



def run_configuration(experiment_name, prey_plearner, prey_plearner_params, pred_plearners, pred_plearner_params, n_episodes,field):
    def reset_player_locations():
        prey.location = prey_loc
        for i, predator in enumerate(predators):
            predator.location = pred_locs[i]
        field.update_state()

    def setup_prey():
        prey = Prey("Chip", prey_loc)
        prey.plearner = prey_plearner(field=field, agent=prey, **prey_plearner_params)  # TODO: check of *array kan
        field.add_player(prey)
        return prey

    def setup_predator():
        predator = Predator("Predator%d" % i, pred_locs[i])
        predator.plearner = pred_plearners[i](field=field, agent=predator,
                                              **pred_plearner_params[i])  # TODO: check of *array kan
        field.add_player(predator)
        predators.append(predator)

    def run_episode():
        field.steps = 0
        while not field.is_ended():
            field.run_step()

    prey_loc = (0,1)#(int(field.width/2),int(field.height/2))
    pred_locs = [(0,0),(field.width -1 ,0),(0,field.height - 1)]

    # Build up field and players
    prey = setup_prey()
    predators = []
    for i,plearner in enumerate(pred_plearners):
        setup_predator()


    # Initialize experiment
    field.init_players()
    num_steps = []
    prey_caught = []
    prob_maps = []
    # Run experiment
    for i in range(0, n_episodes):
        reset_player_locations()
        run_episode()
        # print State.state_from_field(field), field.steps, ("Prey caught" if field.state.prey_is_caught() else "collided")
        num_steps.append(field.steps)
        prey_caught.append(1 if field.state.prey_is_caught() else 0)
        if isinstance(field.get_predators()[0].plearner.policy, Mixed_policy):
            prob_maps.append(field.get_predators()[0].plearner.policy.get_probability_mapping(State([(0,1)])))
        # Print progress
        if i % (n_episodes/20) == 0:
            print i,
            # print prob_maps[i]
    print
    return num_steps, prob_maps, prey_caught

if __name__ == '__main__':
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm']
    color_iter = enumerate(colors)

    experiment = run_configuration
    parameter_sets = [
        # One dict per experiment/plot
        # TODO: LEAVE THE FINISHED EXPERIMENTS IN THE COMMENTS (For reference to the parameters)
        # dict(
        #     experiment_name = ["3 Predator Random policy"],
        #     prey_plearner = [ProbabilisticPlearner],
        #     prey_plearner_params = [dict()],
        #     pred_plearners = [[ProbabilisticPlearner,ProbabilisticPlearner,ProbabilisticPlearner]],
        #     pred_plearner_params = [[dict(),dict(),dict()]],
        #     n_episodes = [10000],
        #     field = [Field(11, 11)],
        # ),
        dict(
            experiment_name = ["1 vs 1 with minimax-q for predator 5by5"],
            prey_plearner = [MiniMaxQPlearner],
            prey_plearner_params = [dict(end_alpha=0.5,num_episodes=500,epsilon=0.1,gamma=0.7)],
            pred_plearners = [[MiniMaxQPlearner]],
            pred_plearner_params = [[dict(end_alpha=0.5,num_episodes=500,epsilon=0.1,gamma=0.7)]],
            n_episodes = [1000],
            field = [Field(5, 5)]
        ),
        # dict(
        #     experiment_name = ["1 vs 2 Independent Q-learning Greedy"],
        #     prey_plearner = [QPlearner],
        #     prey_plearner_params = [dict(policy=GreedyPolicy(value_init=15, epsilon=0.1,
        #                                      gamma=0.9, q_value_select=True),
        #                  learning_rate=0.1, discount_factor=0.9)],
        #     pred_plearners = [[QPlearner,QPlearner]],
        #     pred_plearner_params = [[dict(policy=GreedyPolicy(value_init=15, epsilon=0.1,
        #                                      gamma=0.9, q_value_select=True),learning_rate=0.1, discount_factor=0.9),
        #                              dict(policy=GreedyPolicy(value_init=15, epsilon=0.1,
        #                                      gamma=0.9, q_value_select=True),learning_rate=0.1, discount_factor=0.9)]],
        #     n_episodes = [10000],
        #     field = [Field(11, 11)],
        # ),
        #
        # dict(
        #     experiment_name = ["1 vs 2 WoLF Greedy"],
        #     prey_plearner = [ProbabilisticPlearner],
        #     prey_plearner_params = [dict()],
        #     pred_plearners = [[Wolf_phc,Wolf_phc]],
        #     pred_plearner_params = [[dict(policy=Mixed_policy(epsilon=0.01)),
        #                              dict(policy=Mixed_policy(epsilon=0.01))]],
        #     n_episodes = [10000],
        #     field = [Field(11, 11)],
        # ),

    ]

    # Call the experiment for all the combinations of parameters and log the results
    for parameters in parameter_sets:
        # Log the results
        # log_file = open("log/results-%s.log"% ([),"w")
        experiment_name = parameters["experiment_name"][0]
        average_window = parameters["n_episodes"][0] / 20
        for args in it.product(*parameters.values()):
            args_dict = {key: args[i] for i,key in enumerate(parameters.keys())}
            print "Running experiment: %s with parameters: %s" % (experiment_name, args_dict)
            print args
            num_steps, prob_maps, prey_caught = experiment(**args_dict)
            # log_file.write("Summary results:") # TODO: Summarize results
            print prob_maps[-5:]
            # Compute plot lines
            avg_steps = moving_average(num_steps,average_window)
            stdev_steps = moving_stdev(num_steps,average_window)
            prey_caught = moving_average(prey_caught,average_window) * 30


            # Plot them
            next_color = color_iter.next()[1]
            plt.plot(avg_steps,alpha=0.5,label="Steps per episode")
            plt.fill_between(range(0,len(stdev_steps)),avg_steps+0.2*stdev_steps,avg_steps-0.2*stdev_steps,alpha=0.1,facecolor=next_color,interpolate=True)
            color_iter.next()
            plt.plot(prey_caught,alpha=0.5,label="Times prey caught (30=100%)")

        # Close log file
        # log_file.close()

        # Finish up plot
        # arg_list = arg_list[0:7]
        plt.ylabel("Plot y")
        plt.xlabel("episode # (moving average and 0.2*stdev(area plot) window of %d)" % (average_window))
        plt.legend(fontsize=9, loc=4)
        axes = plt.gca()
        axes.set_ylim([0,30])
        plt.grid(True)
        # learning_algo = "-".join(learning_algo)
        plt.savefig("../reports/plot-%s.png" % experiment_name, bbox_inches='tight',dpi=200)
        plt.close()
        print "Experiment Done."