"""
Originally generated with
copier copy https://github.com/Farama-Foundation/gymnasium-env-template.git "gridworld"
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
Linted and cleaned up

Migrated from pygame to SDL3
allows manual game when this file is excuted
can be executed from grid_world.dqn.py (which calls reset and step,  
but can't quit SDL3 window

Assuming this file is -
C:\\Users\\shane\\dev\\gymnasium\\gridworld\\gymnasium_env\\envs\\grid_world.py

Set up dunder-init files (__init__.py)
======================================
$ cat gridworld/gymnasium_env/__init__.py
from gymnasium.envs.registration import register
register(
    id="GridWorld-v0",
    entry_point="gridworld.gymnasium_env.envs:GridWorldEnv",
)
$ cat gridworld/gymnasium_env/envs/__init__.py
from gridworld.gymnasium_env.envs.grid_world import GridWorldEnv

Register while making
=====================
import gymnasium as gym
env = gym.make('gridworld.gymnasium_env:GridWorld-v0')
"""

from enum import Enum
import ctypes
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
os.environ["SDL_MAIN_USE_CALLBACKS"] = "1"
os.environ["SDL_RENDER_DRIVER"] = "opengl"
import sdl3 # pylint: disable=wrong-import-order, wrong-import-position

RENDERER = ctypes.POINTER(sdl3.SDL_Renderer)()
WINDOW = ctypes.POINTER(sdl3.SDL_Window)()
WINDOW_SIZE = 512
SIZE = 5
PIX_SQUARE_SIZE = WINDOW_SIZE / SIZE  # The size of a single grid square in pixels

SLOW = 4

GLOBAL_DATA = {
    "font" : None,
    "render_modes": ["human", "rgb_array"],
    "render_fps": SLOW,
    "step": 0,
    "distance": -1,
    "terminated": False,
    "direct": True,
    "SDL3_GridWorldEnv": None
}

class Actions(Enum):
    """Actions"""
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    """Grid world environment"""
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": SLOW,
                "step": 0,
                "distance": -1,
                "terminated": False,
                "direct": True
                }

    def __init__(self, render_mode=None):

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, SIZE - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, SIZE - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.metadata['action_to_direction'] = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, -1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.metadata['render_mode'] = render_mode

        """
        If human-rendering is used, `self.screen` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.screen = None
        self.clock = None
        self._agent_location = None
        self._target_location = None
        self.fps_font = None

        GLOBAL_DATA['sdl_appinit'] = self.sdl_appinit()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        new_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        if self.metadata["distance"] >= 0 and self.metadata[
        "direct"] and new_distance >= self.metadata["distance"]:
            self.metadata["direct"] = False
        self.metadata["distance"] = new_distance
        return {
            "distance": new_distance
        }

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, SIZE, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, SIZE, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        # if self.metadata['render_mode'] == "human":
            # self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.metadata['action_to_direction'][action]

   # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, SIZE - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            self.metadata["terminated"] = True

        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.metadata['render_mode'] == "human":
            # self._render_frame()
            self.sdl3_render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.metadata['render_mode'] == "rgb_array":
            return self.sdl3_render_frame()
        return None

    def sdl3_render_frame(self):
        """render frame based on event"""
        sdl3.SDL_SetRenderDrawColor(RENDERER, 255, 255, 255, sdl3.SDL_ALPHA_OPAQUE)
        sdl3.SDL_RenderClear(RENDERER)
        # First we draw the target=
        sdl3.SDL_SetRenderDrawColor(RENDERER, 255, 0, 0, sdl3.SDL_ALPHA_OPAQUE)
        rect = sdl3.SDL_FRect(PIX_SQUARE_SIZE * self._target_location[0],
        PIX_SQUARE_SIZE * self._target_location[1], PIX_SQUARE_SIZE, PIX_SQUARE_SIZE)
        sdl3.SDL_RenderFillRect(RENDERER, rect)
        # Now we draw the agent
        sdl3.SDL_SetRenderDrawColor(RENDERER, 0, 0, 255, sdl3.SDL_ALPHA_OPAQUE)
        rect = sdl3.SDL_FRect(PIX_SQUARE_SIZE * self._agent_location[0],
        PIX_SQUARE_SIZE * self._agent_location[1], PIX_SQUARE_SIZE, PIX_SQUARE_SIZE)
        sdl3.SDL_RenderFillRect(RENDERER, rect)
        sdl3.SDL_RenderPresent(RENDERER)
        return sdl3.SDL_APP_CONTINUE

    def sdl_appinit(self):
        """SDL_AppInit"""
        if not sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO):
            sdl3.SDL_Log("Couldn't initialize SDL: %s".encode() % sdl3.SDL_GetError())
            return sdl3.SDL_APP_FAILURE

        # Initialize the TTF library
        if not sdl3.TTF_Init():
            sdl3.SDL_Log("Couldn't initialize TTF: %s".encode() % sdl3.SDL_GetError())
            return sdl3.SDL_APP_FAILURE

        if not sdl3.SDL_CreateWindowAndRenderer(
        "Grid World".encode(), WINDOW_SIZE,
        WINDOW_SIZE, 0, WINDOW, RENDERER):
            sdl3.SDL_Log("Couldn't create window/renderer: %s".encode() % sdl3.SDL_GetError())
            return sdl3.SDL_APP_FAILURE

        sdl3.SDL_SetRenderVSync(RENDERER, 1) # Turn on vertical sync
        GLOBAL_DATA["font"] = sdl3.TTF_OpenFont("C:/Windows/Fonts/arial.ttf".encode(), 26)
        if not GLOBAL_DATA["font"]:
            sdl3.SDL_Log("Error: %s".encode() % sdl3.SDL_GetError())
            return sdl3.SDL_APP_FAILURE
        return sdl3.SDL_APP_CONTINUE


@sdl3.SDL_AppInit_func
def SDL_AppInit(appstate, argc, argv):# pylint: disable=invalid-name, unused-argument
    """SDL_AppInit"""
    GLOBAL_DATA["SDL3_GridWorldEnv"] = GridWorldEnv()
    # rc = GLOBAL_DATA["SDL3_GridWorldEnv"].sdl_appinit()
    if GLOBAL_DATA['sdl_appinit'] == sdl3.SDL_APP_CONTINUE:
        GLOBAL_DATA["SDL3_GridWorldEnv"].reset()
    return GLOBAL_DATA['sdl_appinit']


@sdl3.SDL_AppEvent_func
def SDL_AppEvent(appstate, event):# pylint: disable=invalid-name, unused-argument
    """SDL_AppEvent"""
    if sdl3.SDL_DEREFERENCE(event).type == sdl3.SDL_EVENT_QUIT:
        return sdl3.SDL_APP_SUCCESS
    if sdl3.SDL_DEREFERENCE(event).type == sdl3.SDL_EVENT_KEY_DOWN:
        if sdl3.SDL_DEREFERENCE(event).key.scancode == sdl3.SDL_SCANCODE_RIGHT:
            GLOBAL_DATA["SDL3_GridWorldEnv"].step(Actions.RIGHT.value)
        if sdl3.SDL_DEREFERENCE(event).key.scancode == sdl3.SDL_SCANCODE_UP:
            GLOBAL_DATA["SDL3_GridWorldEnv"].step(Actions.UP.value)
        if sdl3.SDL_DEREFERENCE(event).key.scancode == sdl3.SDL_SCANCODE_LEFT:
            GLOBAL_DATA["SDL3_GridWorldEnv"].step(Actions.LEFT.value)
        if sdl3.SDL_DEREFERENCE(event).key.scancode == sdl3.SDL_SCANCODE_DOWN:
            GLOBAL_DATA["SDL3_GridWorldEnv"].step(Actions.DOWN.value)
    if GLOBAL_DATA["SDL3_GridWorldEnv"].metadata["terminated"]:
        GLOBAL_DATA["SDL3_GridWorldEnv"].reset()
        GLOBAL_DATA["SDL3_GridWorldEnv"].metadata["terminated"] = False

    return sdl3.SDL_APP_CONTINUE

@sdl3.SDL_AppIterate_func
def SDL_AppIterate(appstate):# pylint: disable=invalid-name, unused-argument
    """SDL_AppIterate"""
    return GLOBAL_DATA["SDL3_GridWorldEnv"].sdl3_render_frame()

@sdl3.SDL_AppQuit_func
def SDL_AppQuit(appstate, result):# pylint: disable=invalid-name, unused-argument
    """SDL_AppQuit"""
