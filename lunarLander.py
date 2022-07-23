import gym
from deepQAgent import *
import sys

def check_command_line():
    """
    Verify that the correct command-line arguments have been provided.
    If not, print a usage message and exit the program.
    """
    args = sys.argv
    valid = True
    if len(args) != 3:
        valid = False
    elif not (args[1] == "train" or args[1] == "test"):
        valid = False
    if not valid:
        print("Invalid command-line arguments")
        print("Usage: train numEpisodes or test wtsFile")
        exit()

def main():
    check_command_line()
    env = gym.make("LunarLander-v2")
    #env = gym.make("CarRacing-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    if sys.argv[1] == "train":
        agent.train(env, int(sys.argv[2]), 200, 128, 8, "LunarLander")
        agent.plot_history()
        agent.test(env, 3, 200)
    elif sys.argv[1] == "test":
        try:
            agent.load_weights(sys.argv[2])
        except:
            print("Unable to load weights from file:", sys.argv[2])
        agent.test(env, 3, 200)
    env.close()

main()
