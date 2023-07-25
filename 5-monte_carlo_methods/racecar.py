import bresenham as br
import random
import time
import signal

'''
Python implementation of off-policy Monte Carlo control learning 
for Exercise 5.12 of Reinforcement Learning - An Introduction (Sutton & Barto)
The task is to train a racecar to reach the finish of a track as quickly as possible. 
by George B. Lawrence
'''

# A state is a tuple containing a position and velocity: ((x_pos, y_pos), (x_vel, y_vel))
# Positions and velocities are discrete. 

class Environment:
    def __init__(self):
        self.track_rows = 32
        self.track_cols = 17
        self.track_bounds = self._generate_track()
        self.track_finish = self._rect_coords((17, 32), (17, 27))
        self.track_start = self._rect_coords((4, 1), (9, 1))
        
    '''
    This generates a list of all the out-of-bounds coordinates
    '''
    def _generate_track(self):
        track = []
        track += self._rect_coords((1, 32), (1, 29)) + self._rect_coords((1, 18), (1, 1))
        track += self._rect_coords((2, 32), (2, 30)) + self._rect_coords((2, 10), (2, 1))
        track += self._rect_coords((3, 32), (3, 32)) + self._rect_coords((3, 3), (3, 1))
        track += self._rect_coords((10, 25), (10, 1))
        track += self._rect_coords((11, 26), (17, 1))
        return track
    
    '''
    Filters out actions as specified, i.e. those that increase a velocity component to 5, 
    make a component negative, or results in both components equalling zero
    '''
    def valid_actions(self, state):
        _, vel = state
        actions = [(-1, -1),
                   (-1, 0), 
                   (-1, 1),
                   (0, -1), 
                   (0, 0), 
                   (0, 1), 
                   (1, -1), 
                   (1, 0), 
                   (1, 1)]
        return list(filter(lambda a:    a[0] + vel[0] in range(-4, 5)
                                    and a[1] + vel[1] in range(5)
                                    and not(a[1] + vel[1] == 0 and a[0] + vel[0] == 0), 
                                    actions))

    '''
    Given a state and an action, this will output the new state, taking into account
    the track boundaries and finish line. 
    '''
    def new_state(self, state, action):
        pos, vel = state
        newvel = (vel[0] + action[0], vel[1] + action[1])
        newpos = (pos[0] + vel[0], pos[1] + vel[1]) # Using the old velocity is done on purpose here
        path = br.bresenham(pos, newpos)
        # Check for intersection between path and bounds
        if list(set(path) & set(self.track_bounds)) or not (1 <= newpos[0] <= self.track_cols and 1 <= newpos[1] <= self.track_rows):
            return random.choice(self.track_start), (0, 0)
        # Check for intersection between path and finish
        if list(set(path) & set(self.track_finish)):
            return True
        return newpos, newvel
    
    '''
    For debugging / viewing purposes: 
    Will print the track (the bounds are represented by the # character, 
    the start and finish lines with _ and | respectively), 
    and the car and its current velocity with o and + respectively. 
    '''
    def visualise_state(self, state):
        pos, vel = state
        br.visualise((1, 1), (self.track_cols, self.track_rows), {  '#': self.track_bounds, 
                                                                    '|': self.track_finish,
                                                                    '_': self.track_start,
                                                                    '+': br.bresenham(pos, (pos[0] + vel[0], pos[1] + vel[1])), 
                                                                    'o': [pos]})

    
    '''
    Helper function to generate a rectangular set of coordinates for track generation
    Takes two points and outputs a set of coordinates
    '''
    def _rect_coords(self, p1, p2):
        return [(x, y)  for x in (range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1)) 
                        for y in (range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1))]

class Agent:
    def __init__(self, env):
        self.env = env
        self._generate_tables()

    '''
    Generate the tables required for MC prediction including the values and the target policy. 
    '''
    def _generate_tables(self):
        positions = [(x, y) for x in range(1, self.env.track_cols + 1) 
                            for y in range(1, self.env.track_rows + 1) 
                            if (x, y) not in self.env.track_bounds]
        states = [(pos, (x_vel, y_vel)) for pos in positions for x_vel in range(-4, 5) for y_vel in range(5)]
        entries = [(state, action) for state in states for action in self.env.valid_actions(state)]
        self.q = {s_a : -1000 for s_a in entries} # Value table
        self.c = {s_a : 0 for s_a in entries}
        self.policy = {s : self.env.valid_actions(s)[0] for s in states}

    '''
    Runs a single episode in the environment, and returns the sequence of states and actions encountered.
    visualise: Set to true if you want the run to be visualised in stdout. 
    delay: Set a delay, in seconds, between each step for visualisation purposes
    deterministic: Set to false to use a random policy, and true to use a deterministic learned policy
    max_steps: Force the episode to end after a certain number of steps. Only use with deterministic 
    '''
    def generate_trajectory(self, visualise = False, delay = 0, deterministic = False, max_steps = 0):
        sequence = []
        state = (random.choice(self.env.track_start), (0, 0))
        steps = 0
        while True:
            if visualise:
                self.env.visualise_state(state)
            if deterministic:
                action = self.policy[state]
            else:
                action = random.choice(self.env.valid_actions(state))
            sequence.append((state, action))
            new_state = self.env.new_state(state, action)
            if new_state == True or (max_steps > 0 and steps > max_steps):
                return sequence
            state = new_state

            steps += 1
            if delay > 0:
                time.sleep(delay)

    '''
    Learn using off-policy Monte Carlo prediction. Behaviour policy is random.
    '''
    def learn(self, episodes):
        for _ in range(episodes):
            traj = self.generate_trajectory()
            w = 1
            g = 0
            for t in range(len(traj) - 1, -1, -1):
                g -= 1
                state, action = traj[t]
                self.c[(state, action)] += w
                self.q[(state, action)] += (w / self.c[(state, action)]) * (g - self.q[(state, action)])
                best_v = None
                best_a = self.policy[state]
                for a in self.env.valid_actions(state):
                    v = self.q[(state, a)]
                    if best_v == None or v > best_v:
                        best_v = v
                        best_a = a
                self.policy[state] = best_a
                if action != self.policy[state]:
                    break
                w = w / (1 / len(self.env.valid_actions(state)))

env = Environment()
state = (random.choice(env.track_start), (0, 0))
agent = Agent(env)

'''
Show us a deterministic run on ctrl-c
'''
def onquit(s, f):
    agent.generate_trajectory(visualise=True, delay=0.1, deterministic=True, max_steps=100)
signal.signal(signal.SIGINT, onquit)

episodes = 0

'''
Note: For visualisation, you probably want your command line window quite large. 
For best viewing it should be 32 lines tall (or if you've changed the track, 
however tall you made it.)
'''

while True:
    # Demonstrate a deterministic run - comment this out if you just want it to train as fast as possible
    traj = agent.generate_trajectory(visualise=True, delay=0.1, deterministic=True, max_steps=100) 
    print(f"{episodes } episodes trained")
    agent.learn(100)
    episodes += 100