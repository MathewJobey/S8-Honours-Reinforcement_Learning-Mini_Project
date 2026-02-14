import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)

# Import our new separated modules
from .settings import *
from .physics import ContactDetector
from .visualizer import RocketVisualizer

class RocketLander(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.world = None
        self.lander = None
        self.legs = []
        self.main_engine_power = 0.0
        self.landing_status = "IN_PROGRESS" # <--- ADD THIS LINE
        
        # Initialize Visualizer
        self.visualizer = RocketVisualizer(self)

        # Observation Space (11 inputs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # Action Space (Main Engine, Steering)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        
        # 1. Create Physics World
        self.world = Box2D.b2World(gravity=(GRAVITY_X, GRAVITY_Y))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.game_over = False
        self.prev_shaping = None
        self.fuel_left = INITIAL_FUEL
        self.landing_status = "IN_PROGRESS" # <--- ADD THIS LINE

        # 2. Create Objects
        self._generate_wind()
        self._create_terrain()
        self._create_rocket()

        return self.step(np.array([0, 0]))[0], {}

    def step(self, action):
        # 1. Prepare Action
        action = np.clip(action, -1, 1) 
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        side_engine_power = np.clip(action[1], -1.0, 1.0)
        
        if self.fuel_left <= 0:
            main_engine_power = 0
            side_engine_power = 0
            
        # --- SAVE FOR VISUALIZER ---
        self.main_engine_power = main_engine_power
        
        # 2. Apply Physics Forces
        angle = self.lander.angle
        
        # Main Engine
        main_force_x = float(-math.sin(angle) * main_engine_power * MAIN_ENGINE_POWER)
        main_force_y = float(math.cos(angle) * main_engine_power * MAIN_ENGINE_POWER)
        self.lander.ApplyForceToCenter((main_force_x, main_force_y), wake=True)
        
        # Side Thrusters
        side_force = float(side_engine_power * SIDE_ENGINE_POWER)
        impulse_x = float(-side_force * math.cos(angle))
        impulse_y = float(-side_force * math.sin(angle))
        
        self.lander.ApplyLinearImpulse(
            (impulse_x, impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, 0)), 
            wake=True
        )

        # Wind
        self.lander.ApplyForceToCenter((float(self.wind_x), float(self.wind_y)), wake=True)

        # 3. Step Simulation
        self.world.Step(1.0 / FPS, 6, 2)

        # 4. Update State & Fuel
        if main_engine_power > 0:
            self.fuel_left -= FUEL_CONSUMPTION_RATE / FPS * main_engine_power

        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        world_width_half = VIEWPORT_W / SCALE / 2
        world_height = VIEWPORT_H / SCALE
        
        state = [
            pos.x / world_width_half,       
            (pos.y / world_height) - 0.5,   
            vel.x * (world_width_half) / FPS,
            vel.y * (world_height) / FPS,
            self.lander.angle,
            self.lander.angularVelocity,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.fuel_left / INITIAL_FUEL,
            self.wind_x / WIND_POWER_MAX,
            self.wind_y / WIND_POWER_MAX
        ]
        
        # --- 5. CALCULATE REWARD (Corrected) ---
        reward = 0
        terminated = False 
        truncated = False
        
        # A. Metric Calculations
        dist_x_meters = abs(pos.x)          
        dist_y_meters = abs(pos.y)          
        velocity_total = math.sqrt(vel.x**2 + vel.y**2) 
        tilt_rad = abs(angle)               

        # B. Shaping Reward
        # We use small multipliers (-10) so it doesn't overpower the landing reward
        shaping = \
            - 10.0 * math.sqrt(dist_x_meters**2 + dist_y_meters**2) \
            - 10.0 * velocity_total \
            - 20.0 * tilt_rad 

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # --- KEY FIX 1: Make Fuel Cheap ---
        # Was 0.15 -> Now 0.03
        # This encourages the AI to use the engine instead of falling.
        reward -= main_engine_power * 0.03
        
        # --- KEY FIX 2: Survival Bonus ---
        # We give it +0.05 points for every frame it stays alive.
        # This forces it to fight gravity to farm points.
        reward += 0.05

        # C. Terminal Rewards
        
        # 1. CRASH Conditions
        if self.game_over or abs(state[0]) >= 1.0 or tilt_rad > 0.5:
            terminated = True
            # We use -100 (not -1000) so it's not too afraid to explore actions.
            reward = -100 
            self.landing_status = "CRASH"
            
        # 2. LANDING Conditions
        elif self.legs[0].ground_contact and self.legs[1].ground_contact:
            if velocity_total < 0.5: 
                terminated = True
                reward = 100 # Safe Landing
                self.landing_status = "LANDED"
                
                # Bonus for accuracy
                if dist_x_meters < 0.5:
                    reward += 50
            
            elif velocity_total > 1.5:
                terminated = True
                reward = -50 # Hard Landing
                self.landing_status = "CRASH"

        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        # Delegate rendering to the Visualizer class
        return self.visualizer.render(self.render_mode)

    def close(self):
        self.visualizer.close()
        self._destroy()

    def _destroy(self):
        if not self.world: return
        self.world = None

    def _generate_wind(self):
        self.wind_x = self.np_random.uniform(-WIND_POWER_MAX, WIND_POWER_MAX)
        self.wind_y = self.np_random.uniform(-WIND_POWER_MAX, WIND_POWER_MAX)

    def _create_terrain(self):
        # 1. The flat ground line (Infinite floor at Y=0)
        self.ground = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(-VIEWPORT_W / SCALE, 0), (VIEWPORT_W / SCALE, 0)])
        )
        self.ground.fixtures[0].friction = 0.8 # Apply friction to the created edge fixture
        
        # 2. The Landing Pad Block (New Physical Object)
        # FIX: We use 'fixtures=fixtureDef(...)' to correctly apply friction
        self.pad_body = self.world.CreateStaticBody(
            position=(0, PAD_HEIGHT_METERS / 2),
            fixtures=fixtureDef(
                shape=polygonShape(box=(PAD_WIDTH_METERS / 2, PAD_HEIGHT_METERS / 2)),
                friction=1.0,  # High friction so the rocket sticks to it
                density=0.0    # 0 density for static objects
            )
        )

    def _create_rocket(self):
        # Random start x (Centered around 0)
        # We spawn between -3m and 3m from the center
        initial_x = self.np_random.uniform(-3.0, 3.0) 
        initial_y = VIEWPORT_H / SCALE
        
        # 1. Main Body (Dynamic)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                # USE SETTINGS CONSTANTS HERE:
                shape=polygonShape(vertices=[
                    (-ROCKET_H_WIDTH, 0), 
                    (ROCKET_H_WIDTH, 0), 
                    (ROCKET_H_WIDTH, ROCKET_HEIGHT), 
                    (-ROCKET_H_WIDTH, ROCKET_HEIGHT)
                ]),
                density=5.0, friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0
            )
        )
        self.lander.color1 = LANDER_COLOR
        
        # 2. Legs
        self.legs = []
        # Leg dimensions relative to rocket size
        leg_w = ROCKET_H_WIDTH * 0.2
        leg_h = ROCKET_HEIGHT * 0.15
        
        for i in [-1, 1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * (ROCKET_H_WIDTH * 0.8), initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(leg_w, leg_h)),
                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001
                )
            )
            leg.ground_contact = False
            self.legs.append(leg)
            
            # Joints
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(0, 0), localAnchorB=(i * (ROCKET_H_WIDTH * 0.8), leg_h),
                enableMotor=True, enableLimit=True, maxMotorTorque=2000.0, motorSpeed=0.3 * i
            )
            if i == -1: rjd.lowerAngle, rjd.upperAngle = 0.4, 0.9
            else: rjd.lowerAngle, rjd.upperAngle = -0.9, -0.4
            self.world.CreateJoint(rjd)
            
        self.drawlist = [self.lander] + self.legs