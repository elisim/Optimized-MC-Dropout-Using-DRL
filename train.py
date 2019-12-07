import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel(tf.logging.ERROR)
import numpy as np
np.random.seed(42)


from src.models import LeNet
from src.data_utils import get_mnist
from gym_eli.envs import EliEnv
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
from datetime import timedelta



def train(net, X_train, y_train, episodes=500, lr=0.00025, right_reward=1):
    env = DummyVecEnv([lambda: EliEnv(net=net,
                                      confidence_rate=0.5, 
                                      X_train=X_train, 
                                      y_train=y_train,
                                      max_mc_dropout_iterations=500,
                                      basic_option=True,
                                      right_reward=right_reward
                                      )])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="yalla", learning_rate=lr)
    model.learn(total_timesteps=episodes)
    

def main():
    (X_train, y_train), (X_test, y_test) = get_mnist()

    # prepare the model
    lenet = LeNet(
        input_shape=X_train.shape[1:],
        num_classes=10,
    )
    
    epochs=20
    mc_rate=0.5
    lenet.set_mc_dropout_rate(mc_rate)
    # lenet.train(X_train, y_train, X_test, y_test, epochs=epochs, verbose=1)
    # lenet.save_model(f'lenet-{mc_rate}-{epochs}')
    lenet.load_model("lenet-0.5-20.h5")

    ### mask
    # mask = 1000
    # X_train = X_train[:mask]
    # y_train = y_train[:mask]

    tic = time.time()
    train(lenet, X_train, y_train, episodes=500, lr=0.00025, right_reward=1)
    toc = time.time()
    print(timedelta(seconds=toc-tic))
    with open("time.txt", "w") as text_file:
    	text_file.write(str(timedelta(seconds=toc-tic)))
    
    
if __name__ == '__main__':
    main()