import numpy as np 
import bresenham as br
import random
import time

# need a path which is a set of coordinates which the racecar passes through between the current state and the next state 
# if the path intersects with the set of coordinates which makes up the finish line, episode terminates
# if the path intersects with the set of coordinates which makes up the boundary, reset to start

# A state is a tuple containing a position and velocity: ((x_pos, y_pos), (x_vel, y_vel))

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
        return list(filter(lambda a:    a[0] + vel[0] in range(0, 5)
                                    and a[1] + vel[1] in range(0, 5)
                                    and not(a[1] + vel[1] == 0 and a[0] + vel[0] == 0), 
                                    actions))

    '''
    Given a state and an action, this will output the new state, taking into account
    the track boundaries and finish line. 
    '''
    def new_state(self, state, action):
        # For redundancy's sake we'll make sure we're not picking an invalid action
        if action not in self.valid_actions(state):
            print(f"Action {action} not allowed in state {state}")
            exit()
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

env = Environment()
state = random.choice(env.track_start), (0, 0)

# At the moment we're simply running one episode on a random policy
while True:
    env.visualise_state(state)
    action = random.choice(env.valid_actions(state)) # Choose a new action
    new_state = env.new_state(state, action) # Get the new state
    if new_state == True: # If the racecar reached the finish, exit
        break
    state = new_state
    time.sleep(0.5)
    
env.visualise_state(state)



    


