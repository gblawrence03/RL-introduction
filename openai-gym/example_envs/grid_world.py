'''
My first custom environment, following the tutorial at https://www.gymlibrary.dev/content/environment_creation/.
Environment is a simple gridworld with a target cell. Actions are "up", "down", "left", and "right". Done signal
issued when the agent reaches the target cell. Rewards is 1 when target reached and 0 at all other times. 
There are 25 * 24 = 600 possible states (agent position and target position).  
'''

import gymnasium as gym
from gymnasium import spaces
import pygame 
import numpy as np

class GridWorldEnv(gym.Env):
    '''
    The env class must inherit from the abstract class gym.Env. 
    This requires that it defines four functions: step(), reset(), render(), and close(). 
    '''

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps" : 4}
    def __init__(self, render_mode=None, size=5):
        self.size = size # The size of the grid
        self.window_size = 512 # The size of the pygame window

        # Observations are a single integer from 0 to 599. 
        self.observation_space = spaces.Discrete(size ** 4)

        # Actions are one of 4 options. 
        self.action_space = spaces.Discrete(4)

        # We need a mapping from self.action_space to movement directions.
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Unsupported render mode {render_mode}"

        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def _get_obs(self):
        location_agent = self._agent_location[0] * self.size + self._agent_location[1]
        location_target = self._target_location[0] * self.size + self._target_location[1]

        return location_agent * (self.size ** 2) + location_target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        obs = self._get_obs()
        info = {}

        if self.render_mode == 'human':
            self._render_frame()

        return obs, info
    
    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = -1
        obs = self._get_obs()
        info = {}

        if self.render_mode == 'human':
            self._render_frame()
        
        return obs, reward, terminated, False, info
    

    def render(self):
        if self.render_mode == 'rgb_array':
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        pygame.draw.rect(
            canvas,
            (255, 0, 0), 
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size)
            )
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3
            )
            pygame.draw.line(
                canvas, 
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3
            )

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()