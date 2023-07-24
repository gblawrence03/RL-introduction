import numpy as np 

# need a path which is a set of coordinates which the racecar passes through between the current state and the next state 
# if the path intersects with the set of coordinates which makes up the finish line, episode terminates
# if the path intersects with the set of coordinates which makes up the boundary, reset to start

class Environment:
    def __init__(self):
        self.track_rows = 17
        self.track_cols = 32
        return

