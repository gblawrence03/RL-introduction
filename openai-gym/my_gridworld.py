'''
Here I'm testing the gridworld environment that I've created.
'''

import gymnasium as gym
import example_envs
import time
import numpy as np

'''
env = gym.make('example_envs/GridWorld-v0', render_mode='human')
env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
    time.sleep(0.3)

env.close()
'''

env = gym.make('example_envs/GridWorld-v0')

q_table = np.zeros([env.observation_space.n, env.action_space.n])

def epsilon_greedy(state, epsilon):
    if np.random.uniform() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
    
alpha = 0.1
gamma = 1
epsilon = 0.1

step = 0
steps = []
episodes = []

print("here")

for episode in range(1, 5000):
    state, info = env.reset()
    done = False
    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)

        # Update the q-value
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = alpha * old_value + (1 - alpha) * (reward + gamma * next_max)

        state = next_state
        step += 1
    print(f"Episode {episode}", end='\r')
    steps.append(step)
    episodes.append(episode)

env.close()

env = gym.make('example_envs/GridWorld-v0', render_mode='human')
obs, info = env.reset()

for _ in range(1000):
    action = np.argmax(q_table[obs])
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
    time.sleep(0.3)