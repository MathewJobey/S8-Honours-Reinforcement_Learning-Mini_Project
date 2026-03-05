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
    
    def _apply_forces(self, main_power, center_power, nose_power):
        angle = self.lander.angle
        vel = self.lander.linearVelocity
        
        # 1. Main Engine Force pushed on COM
        main_force_x = float(-math.sin(angle) * main_power * MAIN_ENGINE_POWER)
        main_force_y = float(math.cos(angle) * main_power * MAIN_ENGINE_POWER)
        self.lander.ApplyForceToCenter((main_force_x, main_force_y), wake=True)
        
        # 2. Center Thrusters (Pure Translation/Sliding)
        center_force = float(center_power * SIDE_ENGINE_POWER)
        c_impulse_x = float(-center_force * math.cos(angle))
        c_impulse_y = float(-center_force * math.sin(angle))
        self.lander.ApplyLinearImpulse(
            (c_impulse_x, c_impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, (ROCKET_HEIGHT + NOSE_HEIGHT) / 2.0)),#because height is 12.5
            wake=True
        )
        # 3. Nose Thrusters (Torque/Tilting)
        nose_force = float(nose_power * NOSE_ENGINE_POWER)
        n_impulse_x = float(-nose_force * math.cos(angle))
        n_impulse_y = float(-nose_force * math.sin(angle))
        self.lander.ApplyLinearImpulse(
            (n_impulse_x, n_impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, ROCKET_HEIGHT)), #nose thrusters are at the top of the rocket, so we use the full height as the local point
            wake=True
        )
        
        # 4. AERODYNAMIC DRAG
        self.current_drag = 0.0
        self.current_torque = 0.0
        exposed_area = abs(math.sin(angle)) * 0.9 + 0.1 
        velocity_squared = vel.x**2 + vel.y**2
        if velocity_squared > 0:
            drag_magnitude = 0.5 * velocity_squared * exposed_area
            self.current_drag = drag_magnitude
            
            speed = math.sqrt(velocity_squared)
            drag_x = -(vel.x / speed) * drag_magnitude
            drag_y = -(vel.y / speed) * drag_magnitude
            
            center_y = (ROCKET_HEIGHT + NOSE_HEIGHT) / 2.0
            offset_y = center_y+0.1 # Slightly above the center of mass to create torque
            drag_point = self.lander.GetWorldPoint(localPoint=(0, offset_y))
            
            self.lander.ApplyForce((drag_x, drag_y), drag_point, wake=True)
    
    
    
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