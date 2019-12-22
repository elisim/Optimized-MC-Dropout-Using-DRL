import sys
import time
import warnings
import tensorflow as tf
import numpy as np
import joblib

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from datetime import timedelta

from src.models import LeNet
from src.data_utils import *
from gym_eli.envs import EliEnv

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
np.random.seed(42)



def train(net, X_train, y_train, episodes=500, lr=0.00025, right_reward=1, new_val_map=1):
    db_1000_iters = joblib.load("Assets/db_1000_iters.jblib")
    logdir = f"with_db_and_new_reward/reward={right_reward}"
    env = DummyVecEnv([lambda: EliEnv(net=net,
                                      confidence_rate=0.5,
                                      X_train=X_train,
                                      y_train=y_train,
                                      max_mc_dropout_iterations=1000,
                                      basic_option=True,
                                      right_reward=right_reward,
                                      new_val_map=new_val_map,
                                      db=db_1000_iters,
                                      log_dir=logdir
                                      )])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=logdir, learning_rate=lr)
    model.learn(total_timesteps=episodes)


def main():
    if len(sys.argv) > 1:
        right_reward = int(sys.argv[1])
        new_val_map = int(sys.argv[2])
    else:
        right_reward = 1
        new_val_map = 1
        
    (X_train, y_train), (X_test, y_test) = get_mnist()
    X_train, X_train_db, y_train, y_train_db = split_to_create_db(X_train, y_train, fold_size=0.2)
    
    # prepare the model
    lenet = LeNet(
        input_shape=X_train.shape[1:],
        num_classes=10,
    )

    epochs = 20
    mc_rate = 0.5
    lenet.set_mc_dropout_rate(mc_rate)
#     lenet.train(X_train, y_train, X_test, y_test, epochs=epochs, verbose=1)
#     lenet.save_model(f'lenet-{mc_rate}-{epochs}')
    lenet.load_model("Assets/lenet-0.5-20-4folds.h5")

    ### mask
    # mask = 1000
    # X_train = X_train[:mask]
    # y_train = y_train[:mask]

    tic = time.time()
    train(lenet, X_train_db, y_train_db, episodes=500, lr=0.00025, right_reward=right_reward, new_val_map=new_val_map)
    toc = time.time()
    elapsed = str(timedelta(seconds=toc-tic))
    print(elapsed)
    with open("time.txt", "w") as text_file:
        text_file.write(elapsed)


if __name__ == '__main__':
    main()
