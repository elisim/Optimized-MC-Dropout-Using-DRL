from gym.envs.registration import register

register(
    id='eli-v0',  # will pass into gym.make() to call our environment.
    entry_point='gym_eli.envs:EliEnv',  # gym_eli.envs is the folders, and 'EliEnv' is the class name inside eli_env.py
)
