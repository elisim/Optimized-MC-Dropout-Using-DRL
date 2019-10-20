import gym
from gym import error, spaces
from gym.utils import seeding
from .mc_dropout_utils import mc_dropout
import numpy as np

MAX_MC_DROPOUT_ITERATIONS = 10000


class EliEnv(gym.Env):
    """
    A MC-Dropout environment to learn how many forward passes is needed per example, given confidence C.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, net, confidence_rate, X_train, y_train, mc_dropout_iterations=50, mc_dropout_rate=0.5):
        """
        :param net: keras network with 'set_mc_dropout_rate' function
        :param confidence_rate: confidence rate (uncertainty)
        :param mc_dropout_iterations: number of forward iterations used to estimate the uncertainty
        :param mc_dropout_rate: dropout rate
        """
        self.net = net
        self.confidence_rate = confidence_rate
        self.X_train = X_train
        self.y_train = y_train
        self.mc_dropout_iterations = mc_dropout_iterations
        self.mc_dropout_rate = mc_dropout_rate
        ###TODO actions: agent can run 2 till mc_dropout_iterations iters.
        self.action_space = spaces.Discrete(mc_dropout_iterations)
        ###TODO obs: numbers, probs. run T mc_iters, get accuracy of that ran for your batch.
        self.observation_space = spaces.Box(low=0, high=1, shape=self.y_train.shape)
        ###TODO minus num_iters for wrong, and 0 for right.
        self.reward_range = (-MAX_MC_DROPOUT_ITERATIONS, +MAX_MC_DROPOUT_ITERATIONS)

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
        ### One episode = one epoch (one pass over all data)

        y_mc_dropout, err_mc_dropout = self._take_action(action)
        reward = None  # TODO reward: think about it. something with the error.
        done = True
        info = {}
        return y_mc_dropout, reward, done, info

    def reset(self):
        pass
        # return an initial observation of the env. But what obs I need to return after episode end?
        # ??????

    def render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        """
        :param action: dict with 'batch_size' and 'mc_dropout_iterations'
        :return: y_mc_dropout, error
        """
        batch_size = action['batch_size']
        mc_dropout_iterations = action['mc_dropout_iterations']
        y_mc_dropout, mc_uncertainty = mc_dropout(self.net, self.X_train,
                                                  batch_size,
                                                  self.mc_dropout_rate,
                                                  mc_dropout_iterations)
        err = self._compute_error(self.y_train, y_mc_dropout)
        return y_mc_dropout, err

    def _compute_error(self, y_true, y_pred):
        # count the number of wrong classifications (label is 1 and i said 2 after mc)
        ## 0-1 % error
        err = np.count_nonzero(np.not_equal(y_pred.argmax(axis=1), y_true.argmax(axis=1)))
        err = err / y_true.shape[0]
        return err

    # TODO: set fixed seed
    # def seed(self, seed=None):
    #     pass
