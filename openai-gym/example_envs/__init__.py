'''
When we import example_envs we want to register the gridworld env and 
make it available to be imported
'''

from example_envs.grid_world import GridWorldEnv
from gymnasium.envs.registration import register

register(
    id='example_envs/GridWorld-v0',
    entry_point='example_envs:GridWorldEnv',
    max_episode_steps=300
)