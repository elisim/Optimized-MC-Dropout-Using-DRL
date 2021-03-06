import gym
from gym import error, spaces
from gym.utils import seeding
from .mc_dropout_utils import mc_dropout
import numpy as np
import tensorflow as tf


class TensorFlowLogger:
    def __init__(self, logdir):
        self._log_dir = logdir
        self._file_writer = tf.summary.FileWriter(self._log_dir + "/metrics")

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._file_writer.add_summary(summary, step)


class EliEnv(gym.Env):
    """
    A MC-Dropout environment to learn how many forward passes is needed per example, given confidence C.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, net, confidence_rate, X_train, y_train, db, log_dir,
                 mc_dropout_rate=0.5,
                 max_mc_dropout_iterations=1000,
                 basic_option=True,
                 right_reward=1,
                 new_val_map=1):
        """
        :param net: keras network with 'set_mc_dropout_rate' function
        :param confidence_rate: confidence rate (uncertainty)
        :param mc_dropout_iterations: number of forward iterations used to estimate the uncertainty
        :param mc_dropout_rate: dropout rate
        """
        print("in inittt")
        self.net = net
        self.confidence_rate = confidence_rate
        self.X_train = X_train
        self.y_train = y_train
        self.data_shape = y_train.shape
        self.mc_dropout_rate = mc_dropout_rate
        ###TODO actions: agent can run 2 till mc_dropout_iterations iters.
        ###Calculation: want: 2 till max. with Discrete(max): 0-max-1. So Discrete(max-1)+2!
        self.action_space = spaces.Discrete(max_mc_dropout_iterations-1)
        ###TODO obs: numbers, probs. run T mc_iters, get accuracy of that ran for your data.
        self.observation_space = spaces.Box(low=0, high=1, shape=self.data_shape)
        ###TODO minus num_iters for wrong, and 0 for right.
        self.reward_range = (-max_mc_dropout_iterations, +max_mc_dropout_iterations)
        self.num_episodes = 0
        self.curr_observation = np.zeros(self.data_shape)
        self.curr_mc_iters = 2
        self.basic_option = basic_option
        self.right_reward = right_reward
        self.db = db  # dict of n_mc_iters -> (mean, var) on X_train
        self.tf_logger = TensorFlowLogger(log_dir)
        self.new_val_map = new_val_map  # see next comment
        self.min_error = 0.98  # min error rate of the ds. used for transform (min_error, 1) to (0, new_val_map)
        

    def step(self, action):
        """
        :param action: number of mc_dropout_iterations
        :return: agent's observation of the current environment
        """
        # run mc_dropout on the network and collect predictions on your batch.
        # output should be: batch x num_classes
        # get mean of runs and compare it to true value
        # you got an error. the reward will be:
        #               (correct/batch) * mc_dropout_iters

#         print("in step")

        # err is float representing the part of the mistake.
        self.curr_mc_iters = action+2
        y_mc_dropout, err, mc_uncertainty = self._take_action(self.curr_mc_iters)
        
        if self.basic_option:  # working option
            # # see https://math.stackexchange.com/questions/377169/calculating-a-value-inside-one-range-to-a-value-of-another-range/377174
            # one_minus_err_after_range = np.interp(1-err, [self.min_error, 1], [0, self.new_val_map])
            # reward = one_minus_err_after_range*self.right_reward - self.curr_mc_iters
            reward = err*(-self.right_reward) - self.curr_mc_iters
        else:  # not working
            reward = (1-err)*self.right_reward - err*self.curr_mc_iters
        
        self.tf_logger.log_scalar(tag="err", value=err, step=self.num_episodes)
        # self.tf_logger.log_scalar(tag="one_minus_err_after_range", value=one_minus_err_after_range, step=self.num_episodes)
        self.tf_logger.log_scalar(tag="mc_iters", value=action, step=self.num_episodes)
        print(f"episode = {self.num_episodes}, action = {self.curr_mc_iters}, err = {err}, reward = {reward}")
    
        done = True  # One episode = one epoch (one pass over all data)
        self.curr_observation = y_mc_dropout
        self.num_episodes += 1
        info = {'err': err, 'num_episodes': self.num_episodes}
        return y_mc_dropout, reward, done, info

    def reset(self):
        return self.curr_observation

    def render(self, mode='human', close=False):
        print(f"num_episodes = {self.num_episodes}")
        print(f"mc_dropout_iterations = {self.curr_mc_iters}")

    def _take_action(self, action):
        """
        :param action: number (mc_dropout_iterations)
        :return: y_mc_dropout, error
        """
#         print("in take action")
#         y_mc_dropout, mc_uncertainty = mc_dropout(net=self.net,
#                                                   X_train=self.X_train,
#                                                   dropout=self.mc_dropout_rate,
#                                                   T=action)
        ans = self.db.get(action)
        if ans:
            # y_mc_dropout, mc_uncertainty = ans # mnist
            standrad_rmse, err, test_ll, y_mc_dropout = ans # conc
        else:
            print(f"ERROR! action {action} is not in the db")
        # err = self._sum_of_true_class_prob(self.y_train, y_mc_dropout) # mnist
        # TODO: do something with mc_uncertainty
        mc_uncertainty = 0
        return y_mc_dropout.flatten(), err, mc_uncertainty

    def _compute_error(self, y_true, y_pred):
        # count the number of wrong classifications (label is 1 and i said 2 after mc)
        ## 0-1 % error
        err = np.count_nonzero(np.not_equal(y_pred.argmax(axis=1), y_true.argmax(axis=1)))
        err = err / y_true.shape[0]
        return err

    def _sum_of_true_class_prob(self, y_true, y_pred):
        # scaled sum of true class probability in y_pred. In this way, we capture if our proba for the true class 
        # increased or decreased.
        # return the part of the mistake (error)
        true_class_indices = y_true.argmax(axis=1)
        data_size = len(true_class_indices)
        tcp_sum = sum(y_pred[range(data_size), true_class_indices])
        return 1-tcp_sum/data_size
    
    # TODO: set fixed seed
    # def seed(self, seed=None):
    #     pass

