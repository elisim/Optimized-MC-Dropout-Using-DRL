import gym
from gym import error, spaces, utils
from gym.utils import seeding


MAX_MC_DROPOUT_ITERATIONS = 10000

BATCH_SIZE = 32
NUM_CLASSES = 10 # TODO: mnist


class EliEnv(gym.Env):
    """
    A MC-Dropout environment to learn how much forward passes is needed per example, given confidence C.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, net, confidence_rate, mc_dropout_iterations=50, mc_dropout_rate=0.5):
        """
        :param net: keras network with 'set_mc_dropout_rate' function
        :param confidence_rate: confidence rate (uncertainty)
        :param mc_dropout_iterations: number of forward iterations used to estimate the uncertainty
        :param mc_dropout_rate: dropout rate
        """
        self.net = net
        self.confidence_rate = confidence_rate
        self.mc_dropout_iterations = mc_dropout_iterations
        self.mc_dropout_rate = mc_dropout_rate
        self.action_space = spaces.Discrete(mc_dropout_iterations)   ###TODO actions: agent can run 2 till mc_dropout_iterations iters.
        self.observation_space = spaces.Box(low=0, high=1, shape=(BATCH_SIZE, NUM_CLASSES))  ###TODO obs: numbers, accuracies. run T mc_iters, get accuracy of that ran for your batch.
        self.reward_range = (-MAX_MC_DROPOUT_ITERATIONS, +MAX_MC_DROPOUT_ITERATIONS) ###TODO minus num_iters for wrong, and 0 for right.

    def step(self, action):
        """
        :param action: number of mc_dropout_iterations
        :return: agent's observation of the current environment
        """
        #### needed: mc_dropout_iterations, net, batch_size, true_label
        # run mc_dropout on the network and collect predictions on your batch. output should be: batch x num_classes (use evaluate mc_dropout).
        # get mean of runs and compare it to true value
        # you got an error. the reward will be:
        #               (correct/batch) * mc_dropout_iters
        #
        mean_y = None
        reward = None
        done = None
        info = {}
        return mean_y, reward, done, info

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


    # TODO: set fixed seed
    # def seed(self, seed=None):
    #     pass

