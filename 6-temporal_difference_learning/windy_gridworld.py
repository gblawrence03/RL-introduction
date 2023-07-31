import bresenham as br
import time
import random
import matplotlib.pyplot as plt

# A state is a 2-tuple of the current position: (x_pos, y_pos)

class Environment:
    '''
    The "ex" argument signifies which exercise we want to solve. This can be left as None to solve 
    the windy gridworld task as stated in the original example (with deterministic wind and 
    #vertical / horizontal movement) or set to 9 to add King's moves, or set to 10 to add stochastic wind. 
    '''
    def __init__(self, ex=None):
        self.ex = ex
        self.world_width = 10
        self.world_height = 7
        self.actions = [
            (-1, 0), # left
            (0, -1), # down
            (1, 0), # right
            (0, 1) # up
        ]
        # Add diagonal moves
        if ex == 9 or ex == 10:
            self.actions += [
                (-1, -1), 
                (-1, 1), 
                (1, -1),
                (1, 1),
                (0, 0)
            ]
        self.start_state = (1, 4)
        self.goal_state = (8, 4)
    
    '''
    Generate reward and new state given state and action
    '''
    def new_reward_state(self, state, action):
        xpos, ypos = state

        # add action to position
        xpos += action[0]
        ypos += action[1]

        # account for wind (stochastic wind if ex 10)
        if xpos in (4, 5, 6, 9):
            if self.ex == 10:
                ypos += random.choice((0, 1, 2))
            else:
                ypos += 1
        if xpos in (7, 8):
            if self.ex == 10:
                ypos += random.choice((1, 2, 3))
            else:
                ypos += 2

        # clamp to world limits
        xpos = min(max(xpos, 1), self.world_width)
        ypos = min(max(ypos, 1), self.world_height)

        if (xpos, ypos) == self.goal_state:
            return 0, self.goal_state

        return -1, (xpos, ypos)
    
    def visualise_state(self, state):
        br.visualise((1, 1), (self.world_width, self.world_height), { 'o': [state], '+': [self.goal_state]})
        print()

class Agent:
    def __init__(self, env):
        self.env = env
        self._generate_tables()

    '''
    Set up the value table
    '''
    def _generate_tables(self):
        states = [(x + 1, y + 1) for x in range(self.env.world_width)
                                 for y in range(self.env.world_height)]
        entries = [(state, action) for state in states for action in self.env.actions]
        self.q = {s_a : -50 for s_a in entries}
        # Set the goal state to 0
        for a in self.env.actions:
            self.q[(self.env.goal_state, a)] = 0

    '''
    Choose optimal action with probability 1 - epsilon, randon action otherwise
    '''
    def epsilon_greedy_policy(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(env.actions)
        else:
            max_q = max([self.q[(state, a)] for a in self.env.actions])
            return random.choice([a for a in self.env.actions if self.q[(state, a)] == max_q])

    '''    
    Given a sarsa and learning rate, update the value table
    '''
    def sarsa_learn(self, lr, s1, a1, r, s2, a2):
        self.q[(s1, a1)] += lr * (r + self.q[(s2, a2)] - self.q[(s1, a1)])

env = Environment(ex=10)
agent = Agent(env)

steps = 0
episodes = 0

step_values = [0]
episode_values = [0]

plt.xlabel("Time steps")
plt.ylabel("Episodes")

epsilon = 0.1
lr = 0.5

# The Sarsa algorithm as described in section 6.4
while steps <= 8000:
    state = env.start_state
    action = agent.epsilon_greedy_policy(state, epsilon)
    while state != env.goal_state:
        reward, new_state = env.new_reward_state(state, action)
        new_action = agent.epsilon_greedy_policy(new_state, epsilon)
        agent.sarsa_learn(lr, state, action, reward, new_state, new_action)
        state, action = new_state, new_action

        steps += 1
    episodes += 1
    step_values.append(steps)
    episode_values.append(episodes)

'''
# Show off our optimal policy with a single run
state = env.start_state
while state != env.goal_state:
    reward, new_state = env.new_reward_state(state, action)
    new_action = agent.epsilon_greedy_policy(new_state, epsilon)
    state, action = new_state, new_action
    env.visualise_state(state)
    time.sleep(0.1)
'''
plt.plot(step_values, episode_values, color='b')
plt.show()
