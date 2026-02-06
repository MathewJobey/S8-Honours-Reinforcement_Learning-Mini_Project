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
