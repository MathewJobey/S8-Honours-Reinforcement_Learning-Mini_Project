import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)
from .settings import *
from .physics import ContactDetector
from .visualizer import RocketVisualizer

class Phase2Descent(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.world = None
        self.lander = None
        self.main_engine_power = 0.0
        self.landing_status = "IN_PROGRESS"
        self.visualizer = RocketVisualizer(self)
        
        # Contains 7 variables: [X Pos, Y Pos, X Vel, Y Vel, Angle, Angular Vel, Fuel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Action space to 3 thrusters variables: [Main, Center, Nose]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
    """THE RESET FUNCTION TO RESTART THE ENVIRONMENT. THIS IS CALLED AT THE START OF EVERY EPISODE."""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy() 
        
        self.world = Box2D.b2World(gravity=(GRAVITY_X, GRAVITY_Y))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.game_over = False
        self.fuel_left = INITIAL_FUEL
        self.landing_status = "IN_PROGRESS"
        
        self.current_drag = 0.0
        self.current_torque = 0.0
        
        self._create_terrain()
        self._create_rocket()

        start_y = 500.0 
        start_x = self.np_random.uniform(-30.0, 30.0)
        self.lander.position = (start_x, start_y)
        self.lander.angle = self.np_random.uniform(-math.pi, math.pi)
        #4. Pick a random falling speed between a hover (0.0) and terminal velocity (-140.0)
        # We use negative numbers because the rocket is moving down the Y-axis
        start_vy = self.np_random.uniform(-140.0, 0.0)
        self.lander.linearVelocity = (0, start_vy)

        return self.step(np.array([0, 0, 0]))[0], {}     
    
    def _create_rocket(self):
        initial_x = 0
        initial_y = 500.0
        
        # 1. Compound Body: Main Hull (90kg) + Nose Cone (10kg) = 100kg Total
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=[
                
                # FIXTURE 1: The Heavy Main Hull
                fixtureDef(
                    shape=polygonShape(vertices=[
                        (-ROCKET_H_WIDTH, 0), 
                        (ROCKET_H_WIDTH, 0), 
                        (ROCKET_H_WIDTH, ROCKET_HEIGHT), 
                        (-ROCKET_H_WIDTH, ROCKET_HEIGHT)
                    ]),
                    density=5.0, friction=0.5, categoryBits=0x0010, maskBits=0x001, restitution=0.0
                ),
                
                # FIXTURE 2: The Lightweight Nose Cone
                fixtureDef(
                    shape=polygonShape(vertices=[
                        (-ROCKET_H_WIDTH, ROCKET_HEIGHT),    # Bottom Left of nose
                        (ROCKET_H_WIDTH, ROCKET_HEIGHT),     # Bottom Right of nose
                        (0, ROCKET_HEIGHT + NOSE_HEIGHT)     # Top Center Tip
                    ]),
                    density=(10.0 / 2.25), friction=0.5, categoryBits=0x0010, maskBits=0x001, restitution=0.0
                )
            ]
        )
        self.lander.color1 = LANDER_COLOR
        
        # --- ADD A NAME TAG ---
        self.lander.userData = "rocket"
        
        # Set drawlist to only include the main hull
        self.drawlist = [self.lander]