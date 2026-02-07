import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)


# --- 1. CONFIGURATION CONSTANTS ---
# (Tweaking these changes how the "game" feels)

FPS = 50
SCALE = 30.0  # affects how large the screen elements are
VIEWPORT_W = 600
VIEWPORT_H = 400

# Rocket Physics
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_FUEL = 1000.0   # Total units of fuel available
FUEL_CONSUMPTION_RATE= 5.0  # Fuel used per second of full power

# Terrain / World
GRAVITY_X = 0.0
GRAVITY_Y = -10.0  # Earth-like gravity (pulls down)
WIND_POWER_MAX = 1.0 # Max random wind force

# Render Colors
LANDER_COLOR = (0.5, 0.4, 0.9)
FUEL_BAR_COLOR = (1.0, 0.0, 0.0)

class RocketLander(gym.Env):
    """
    Custom Rocket Landing Environment
    Action Space: Continuous (Main Engine, Side Thrusters)
    Observation Space: [x, y, vx, vy, angle, angular_vel, left_leg_touch, right_leg_touch, fuel_left, wind_x, wind_y]
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.world=None
        self.lander = None
        self.legs = []
        
        # 'high' defines the upper bounds of the observation space for the rocket landing RL environment. 
        #  Each value corresponds to the maximum expected range for one state variable: 
        # [X position, Y position, X velocity, Y velocity, angle (±π), angular velocity, # left leg contact, right leg contact, fuel, wind X, wind Y]. 
        # These limits normalize the agent’s inputs and keep training stable by capping values.
        high = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Action Space: # This defines the agent’s possible control inputs. 
        # # Index 0 → Main Engine thrust, ranging from 0 (off) to 1 (full power). 
        # # Index 1 → Steering control, ranging from -1 (full left tilt) to +1 (full right tilt).
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )