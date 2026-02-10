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
        # We don't strictly need side_engine_power for the current HUD, 
        # but good to keep if you add it back later.
        
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
        
        # FIX: Apply side thrust at Center (0,0) to push sideways without wild spinning
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
        
        # FIX: Coordinate Normalization
        # Previous code subtracted view width, confusing Center with Left Edge.
        # Now: 0 is Center, -1 is Left Edge, +1 is Right Edge.
        world_width_half = VIEWPORT_W / SCALE / 2
        world_height = VIEWPORT_H / SCALE
        
        state = [
            pos.x / world_width_half,       # X Position (-1 to 1)
            (pos.y / world_height) - 0.5,   # Y Position
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
        
        # --- 5. CALCULATE REWARD ---
        reward = 0
        terminated = False 
        truncated = False
        
        # A. Metric Calculations (The Physics Data)
        # We use absolute values because "distance" is always positive
        dist_x_meters = abs(pos.x)          # Horizontal distance from center (0.0)
        dist_y_meters = abs(pos.y)          # Vertical altitude
        velocity_total = math.sqrt(vel.x**2 + vel.y**2) # Total speed (m/s)
        tilt_rad = abs(angle)               # Tilt in radians (0 is upright)

        # B. Shaping Reward (Hot/Cold Game)
        # This gives small hints every frame to guide the rocket towards the goal.
        # We penalize being far away, moving fast, or spinning.
        shaping = \
            - 100 * math.sqrt(dist_x_meters**2 + dist_y_meters**2) \
            - 100 * velocity_total \
            - 100 * tilt_rad

        # Calculate the "Delta" (Improvement) since last frame
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Fuel Penalty (Small cost per frame to prevent hovering forever)
        reward -= main_engine_power * 0.10
        
        # C. Terminal Rewards (The "Report Card" from your Table)
        # This only triggers when the episode ends (Crash or Land)
        
        # 1. CRASH Conditions
        # - Body hit ground (game_over)
        # - Left screen bounds (abs(x) > 1.0)
        # - Tilted too far (> 45 degrees / 0.78 rad)
        if self.game_over or abs(state[0]) >= 1.0 or tilt_rad > 0.78:
            terminated = True
            reward = -100 # "Crash: Hit ground too fast / Tipped"
            
        # 2. LANDING Conditions
        # - Legs touching ground AND Rocket is sleeping (stopped moving)
        elif not self.lander.awake:
            terminated = True
            reward = 100 # "Survival: Safe Landing"
            
            # --- ACCURACY (Bullseye vs Pad vs Grass) ---
            # Pad Radius is half the width (approx 1.33 meters)
            pad_radius = PAD_WIDTH_METERS / 2
            
            if dist_x_meters < 0.2:
                reward += 50  # "Accuracy: Distance < 0.2m (Bullseye)"
            elif dist_x_meters < pad_radius:
                reward += 20  # "Accuracy: Distance < Pad Width"
            else:
                reward -= 20  # "Accuracy: Missed Pad (Landed on grass)"
                
            # --- SOFTNESS (Butter vs Hard) ---
            if velocity_total < 0.5:
                reward += 30  # "Softness: Speed < 0.5 m/s"
            elif velocity_total > 2.0:
                reward -= 30  # "Softness: Speed > 2.0 m/s"
                
            # --- STYLE (Upright) ---
            if tilt_rad < 0.05:
                reward += 20  # "Style: Tilt < 0.05 rad"

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
        self.ground = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(-VIEWPORT_W / SCALE, 0), (VIEWPORT_W / SCALE, 0)])
        )
        self.ground.friction = 0.8

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