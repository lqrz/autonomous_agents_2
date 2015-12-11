__author__ = 'fbuettner'
from models.field import Field
from models.predator import Predator
from models.prey import Prey
from models.plearners.wolf_phc import Wolf_phc
from models.plearners.minimax_q_plearner import MiniMaxQPlearner
from models.state import State
from matplotlib import pyplot as plt
from graphics.plot import get_output_path
import timeit


def run_wolf(n_episodes=1000):
    # initialize the environment
    field = Field(3, 3)

    """
    initial state:
    | | | |
    |X|O|X|
    | | | |
    """
    pred1loc = (0, 1)
    pred2loc = (2, 1)
    preyloc = (1, 1)

    predator1 = Predator(id="Plato", location=pred1loc)
    predator2 = Predator(id="Pythagoras", location=pred2loc)

    # WoLF
    predator1.plearner = Wolf_phc.create_greedy_plearner(field=field, agent=predator1)
    predator2.plearner = Wolf_phc.create_greedy_plearner(field=field, agent=predator2)
    field.add_player(predator1)
    field.add_player(predator2)

    chip = Prey(id="Kant", location=preyloc)
    chip.plearner = Wolf_phc.create_greedy_plearner(field=field, agent=chip, epsilon=0.01)
    field.add_player(chip)
    field.init_players()

    plot_state = State.state_from_field(field)

    num_steps = []
    pred_win = []
    value_of_pred1 = []
    value_of_pred2 = []
    value_of_prey = []

    for i in range(0, n_episodes):
        predator1.location = pred1loc
        predator2.location = pred2loc
        chip.location = preyloc
        field.update_state()
        field.steps = 0
        # run the simulation
        while not field.is_ended():
            field.run_step()

        num_steps.append(field.steps)
        pred_win.append(field.state.prey_is_caught())
        value_of_pred1.append(predator1.plearner.policy.get_probability_mapping(plot_state))
        value_of_pred2.append(predator2.plearner.policy.get_probability_mapping(plot_state))
        value_of_prey.append(chip.plearner.policy.get_probability_mapping(plot_state))

        # print progress every 10%
        if n_episodes > 10 and i % (n_episodes / 10) == 0:
            print int(1.0 * i / n_episodes * 100), "%"

    # some list wrangling to get a list of 5 action lists with values for each predator
    vp1 = [[val[0] for val in sublist] for sublist in zip(*value_of_pred1)]
    vp2 = [[val[0] for val in sublist] for sublist in zip(*value_of_pred2)]
    vpc = [[val[0] for val in sublist] for sublist in zip(*value_of_prey)]


    # create plots
    colors = ["r", "b", "g", "k", "m"]
    actions = {
        (0, 0): "stay",
        (-1, 0): "left",
        (1, 0): "right",
        (0, -1): "up",
        (0, 1): "down"
    }
    plt.figure(figsize=(15, 15))

    s = plt.subplot(3, 1, 1)
    s.set_yscale("log")
    for index, action in enumerate(predator1.actions):
        plt.plot(vp1[index], c=colors[index], label=actions[action])
    plt.title("action probabilities for predator 1")
    plt.legend(loc="upper right")

    s = plt.subplot(3, 1, 2)
    s.set_yscale("log")
    for index, action in enumerate(predator2.actions):
        plt.plot(vp2[index], c=colors[index], label=actions[action])
    plt.title("action probabilities for predator 2")
    # plt.legend(loc="upper left")

    s = plt.subplot(3, 1, 3)
    s.set_yscale("log")
    for index, action in enumerate(chip.actions):
        plt.plot(vpc[index], c=colors[index], label=actions[action])
    plt.title("action probabilities for prey")

    plt.suptitle(str(n_episodes) + " episodes")
    plt.savefig(get_output_path() + "policychange-wolf-" + str(n_episodes) + ".pdf")

########################################################################################################################
def run_minimax(n_episodes=1000):
    # initialize the environment
    field = Field(5, 5)

    """
    initial state:
    | | | |
    |X|O| |
    | | | |
    """
    pred1loc = (0, 0)
    preyloc = (2, 2)

    predator1 = Predator(id="Plato", location=pred1loc)

    # WoLF
    predator1.plearner = MiniMaxQPlearner(field=field, agent=predator1, end_alpha=0.1, num_episodes=n_episodes, epsilon=0.1)
    field.add_player(predator1)

    chip = Prey(id="Kant", location=preyloc, tripping_prob=0.2)
    chip.plearner = MiniMaxQPlearner(field=field, agent=chip, end_alpha=0.1, num_episodes=n_episodes, epsilon=0.1)
    field.add_player(chip)
    field.init_players()

    plot_state = State([(1, 0)])

    num_steps = []
    pred_win = []
    value_of_pred1 = []
    value_of_prey = []

    for i in range(0, n_episodes):
        predator1.location = pred1loc
        chip.location = preyloc
        field.update_state()
        field.steps = 0
        # run the simulation
        while not field.is_ended():
            field.run_step()
            # print field.state

        num_steps.append(field.steps)
        pred_win.append(field.state.prey_is_caught())
        value_of_pred1.append(predator1.plearner.policy.get_probability_mapping(plot_state))
        # print predator1.plearner.policy.get_probability_mapping(plot_state)
        value_of_prey.append(chip.plearner.policy.get_probability_mapping(plot_state))

        # print progress every 10%
        if n_episodes >= 10 and i % (n_episodes / 10) == 0:
            print int(1.0 * i / n_episodes * 100), "%:", field.steps, "steps"

    # some list wrangling to get a list of 5 action lists with values for each predator
    vp1 = [[val[0] for val in sublist] for sublist in zip(*value_of_pred1)]
    vpc = [[val[0] for val in sublist] for sublist in zip(*value_of_prey)]


    # create plots
    colors = ["r", "b", "g", "k", "m"]
    actions = {
        (0, 0): "stay",
        (-1, 0): "left",
        (1, 0): "right",
        (0, -1): "up",
        (0, 1): "down"
    }
    plt.figure(figsize=(15, 15))

    s = plt.subplot(2, 1, 1)
    # s.set_yscale("log")
    plt.ylim([-0.1, 1.1])
    for index, action in enumerate(predator1.actions):
        plt.plot(vp1[index], c=colors[index], label=actions[action])
    plt.title("action probabilities for predator 1")
    plt.legend(loc="upper right")

    s = plt.subplot(2, 1, 2)
    #s.set_yscale("log")
    plt.ylim([-0.1, 1.1])
    for index, action in enumerate(chip.actions):
        plt.plot(vpc[index], c=colors[index], label=actions[action])
    plt.title("action probabilities for prey")

    plt.suptitle(str(n_episodes) + " episodes")
    plt.savefig(get_output_path() + "policychange-minimax-" + str(n_episodes) + ".pdf")


if __name__ == "__main__":
    start = timeit.default_timer()
    # run_wolf(n_episodes=10000)
    run_minimax(n_episodes=1000)
    print "finished after", round(timeit.default_timer() - start, 3), "seconds."

