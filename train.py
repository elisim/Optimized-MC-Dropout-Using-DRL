import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
np.random.seed(42)

import time
import joblib
import argparse

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from datetime import timedelta

from src.models import LeNet
from src.data_utils import *
from gym_eli.envs import EliEnv
from src.yarin_gal_net import net as yg_net


parser = argparse.ArgumentParser()

parser.add_argument('--data', '-d', required=True, help='Dataset: mnist or conc')
parser.add_argument('--right_reward', '-r', default=1, type=int, help='right reward')
parser.add_argument('--new_val_map', '-n', default=1, type=int, help='new val map')
parser.add_argument('--episodes', '-e', default=500, type=int, help='episodes')


def train(net, X_train, y_train, db, episodes=500, lr=0.00025, right_reward=1, new_val_map=1):
    logdir = f"{net.__class__.__name__}/reward={right_reward}"
    env = DummyVecEnv([lambda: EliEnv(net=net,
                                      confidence_rate=0.5,
                                      X_train=X_train,
                                      y_train=y_train,
                                      max_mc_dropout_iterations=1000,
                                      basic_option=True,
                                      right_reward=right_reward,
                                      new_val_map=new_val_map,
                                      db=db,
                                      log_dir=logdir
                                      )])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=logdir, learning_rate=lr)
    model.learn(total_timesteps=episodes)


def main():
    args = parser.parse_args()
    right_reward = args.right_reward
    new_val_map = args.new_val_map
    episodes = args.episodes

    if args.data == 'mnist':
        (X_train, y_train), (X_test, y_test) = get_mnist()
        X_train, X_train_db, y_train, y_train_db = split_to_create_db(X_train, y_train, fold_size=0.2)
        # prepare the model
        net = LeNet(
            input_shape=X_train.shape[1:],
            num_classes=10,
        )
        net.load_model("Assets/lenet-0.5-20-4folds.h5")
        db = joblib.load("Assets/db_1000_iters.jblib")

    elif args.data == 'conc':
        _DATA_FILE = "Data/Concrete_Strength.txt"
        X_train, y_train = get_concrete(_DATA_FILE)
        X_train, X_train_db, y_train, y_train_db = split_to_create_db(X_train, y_train, fold_size=0.2)
        net = yg_net(X_train, y_train,
                     n_hidden=([40] * 1),
                     n_epochs=4000,
                     normalize=True,
                     tau=0.05,
                     dropout=0.005,
                     load_model=True
                     )
        net.load_model("Assets/concrete_net_4000_epochs.h5")
        db = joblib.load("Assets/db_concrete_10000_iters.jblib")

    tic = time.time()
    train(net, X_train_db, y_train_db, db, episodes=episodes, lr=0.00025, right_reward=right_reward, new_val_map=new_val_map)
    toc = time.time()
    elapsed = str(timedelta(seconds=toc-tic))
    print(elapsed)
    # with open("time.txt", "w") as text_file:
    #     text_file.write(elapsed)


if __name__ == '__main__':
    main()
