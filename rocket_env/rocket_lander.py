import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)

from .settings import *
from .physics import ContactDetector
from .visualizer import RocketVisualizer

class RocketLander(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.world = None
        self.lander = None
        # self.legs = []  <-- Deleted
        self.main_engine_power = 0.0
        self.landing_status = "IN_PROGRESS"
        
        self.visualizer = RocketVisualizer(self)

        # --- UPDATE THIS ---
        # Old: shape=(11,)
        # New: shape=(7,)  [x, y, vx, vy, angle, angular_vel, fuel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Action Space stays the same
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        
        self.world = Box2D.b2World(gravity=(GRAVITY_X, GRAVITY_Y))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.game_over = False
        self.prev_shaping = None
        self.fuel_left = INITIAL_FUEL
        self.landing_status = "IN_PROGRESS"

        #self._generate_wind()
        self._create_terrain()
        self._create_rocket()

        # --- BELLY FLOP INITIALIZATION ---
        # 1. Start High (80% of screen height)
        # 2. Start Sideways (90 degrees / 1.57 radians)
        # 3. Start with random velocity
        
        start_y = (VIEWPORT_H / SCALE) * 0.9 
        self.lander.position = (0, start_y)
        
        # Randomize orientation (+/- 90 deg) to force it to learn both sides
        if self.np_random.random() > 0.5:
            self.lander.angle = 1.57  # 90 degrees (Horizontal)
        else:
            self.lander.angle = -1.57 # -90 degrees
            
        # Give it a nudge so it drifts
        self.lander.linearVelocity = (self.np_random.uniform(-10, 10), -10)

        return self.step(np.array([0, 0]))[0], {}

    def step(self, action):
        action = np.clip(action, -1, 1) 
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        side_engine_power = np.clip(action[1], -1.0, 1.0)
        
        if self.fuel_left <= 0:
            main_engine_power = 0
            side_engine_power = 0
            
        self.main_engine_power = main_engine_power
        
        # Apply Physics (Now includes Drag!)
        self._apply_forces(main_engine_power, side_engine_power)

        self.world.Step(1.0 / FPS, 6, 2)

        if main_engine_power > 0:
            self.fuel_left -= FUEL_CONSUMPTION_RATE / FPS * main_engine_power

        state = self._get_state()
        reward, terminated, truncated = self._compute_reward(state, main_engine_power)

        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}

    def _apply_forces(self, main_power, side_power):
        angle = self.lander.angle
        vel = self.lander.linearVelocity
        
        # 1. Main Engine Force
        main_force_x = float(-math.sin(angle) * main_power * MAIN_ENGINE_POWER)
        main_force_y = float(math.cos(angle) * main_power * MAIN_ENGINE_POWER)
        self.lander.ApplyForceToCenter((main_force_x, main_force_y), wake=True)
        
        # 2. Side Thrusters (Steering)
        side_force = float(side_power * SIDE_ENGINE_POWER)
        impulse_x = float(-side_force * math.cos(angle))
        impulse_y = float(-side_force * math.sin(angle))
        self.lander.ApplyLinearImpulse(
            (impulse_x, impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, ROCKET_HEIGHT/2)), 
            wake=True
        )

        # 3. AERODYNAMIC DRAG (KEEP THIS!)
        # This creates the belly-flop physics (slows down when sideways)
        exposed_area = abs(math.sin(angle)) * 0.9 + 0.1 
        velocity_squared = vel.x**2 + vel.y**2
        if velocity_squared > 0:
            drag_magnitude = 0.5 * velocity_squared * exposed_area
            
            speed = math.sqrt(velocity_squared)
            drag_x = -(vel.x / speed) * drag_magnitude
            drag_y = -(vel.y / speed) * drag_magnitude
            
            self.lander.ApplyForceToCenter((drag_x, drag_y), wake=True)

        # --- IMPORTANT CHECK ---
        # Ensure there is NO line here like:
        # self.lander.ApplyForceToCenter((self.wind_x, self.wind_y))
        # If there is, DELETE IT.
        
    def _get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # 1. SCALE CONSTANTS
        # Viewport width in meters: 800 / 10 = 80m. Half = 40m.
        world_width_half = VIEWPORT_W / SCALE / 2 
        
        # Max Altitude: We spawn at 350m, so let's normalize against 400m.
        # This gives a value like 0.875, which is perfect for Neural Networks (0 to 1).
        MAX_ALTITUDE = 400.0
        
        return [
            pos.x / world_width_half,       # 1. Horizontal Position (-1 to 1)
            pos.y / MAX_ALTITUDE,           # 2. Vertical Position (0 to 1)
            
            vel.x / 50.0,                   # 3. Horizontal Velocity (Normalized by approx max speed)
            vel.y / 50.0,                   # 4. Vertical Velocity
            
            self.lander.angle,              # 5. Angle
            self.lander.angularVelocity,    # 6. Angular Velocity
            
            self.fuel_left / INITIAL_FUEL   # 7. Fuel
            
            # REMOVED: Legs (No longer exist)
            # REMOVED: Wind (Disabled)
        ]

    def _compute_reward(self, state, main_engine_power):
        reward = 0
        terminated = False 
        truncated = False
        
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # Use target repo's normalization logic for better signals
        x_range = VIEWPORT_W / SCALE / 2
        y_range = VIEWPORT_H / SCALE
        
        dist_x = abs(pos.x)
        dist_y = abs(pos.y - PAD_HEIGHT_METERS)
        
        # 1. Distance Reward (Normalized 0 to 1)
        dist_norm = (dist_x / x_range) + (dist_y / y_range)
        dist_reward = 1.0 - min(1.0, dist_norm)
        
        # 2. Orientation Reward
        # We relax this slightly at high altitudes to allow the flop,
        # but penalties ramp up as it gets closer.
        tilt_rad = abs(self.lander.angle)
        pose_reward = 1.0 - min(1.0, tilt_rad / 1.5)
        
        # Continuous Reward
        reward += 0.1 * dist_reward
        reward += 0.1 * pose_reward
        
        # Fuel Cost (Encourage Efficiency)
        reward -= main_engine_power * 0.05

        # --- TERMINAL STATES ---
        velocity_total = math.sqrt(vel.x**2 + vel.y**2)
        
        # Crash conditions
        if self.game_over or abs(pos.x) >= x_range or tilt_rad > 1.2:
            terminated = True
            reward = -100
            self.landing_status = "CRASH"
            
        # Landing conditions
        elif self.legs[0].ground_contact and self.legs[1].ground_contact:
            # Must be SLOW and UPRIGHT
            if velocity_total < 1.5 and tilt_rad < 0.2: 
                terminated = True
                reward = 100 
                self.landing_status = "LANDED"
                if dist_x < 2.0: # Precision Bonus
                    reward += 50
            else:
                # Hard landing (too fast)
                terminated = True
                reward = -50
                self.landing_status = "CRASH"
        
        return reward, terminated, truncated

    def render(self):
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
        self.ground.fixtures[0].friction = 1.0 
        
        self.pad_body = self.world.CreateStaticBody(
            position=(0, PAD_HEIGHT_METERS / 2),
            fixtures=fixtureDef(
                shape=polygonShape(box=(PAD_WIDTH_METERS / 2, PAD_HEIGHT_METERS / 2)),
                friction=1.0,  
                density=0.0    
            )
        )

    def _create_rocket(self):
        # Initial position handled in reset()
        initial_x = 0
        initial_y = 0 
        
        # 1. Main Body
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[
                    (-ROCKET_H_WIDTH, 0), 
                    (ROCKET_H_WIDTH, 0), 
                    (ROCKET_H_WIDTH, ROCKET_HEIGHT), 
                    (-ROCKET_H_WIDTH, ROCKET_HEIGHT)
                ]),
                density=5.0, friction=0.5, categoryBits=0x0010, maskBits=0x001, restitution=0.0
            )
        )
        self.lander.color1 = LANDER_COLOR
        
        # 2. Legs
        self.legs = []
        leg_w = 0.2
        leg_h = 1.0
        
        for i in [-1, 1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * (ROCKET_H_WIDTH * 0.8), initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(leg_w, leg_h)),
                    density=1.0, restitution=0.0, friction=1.0, categoryBits=0x0020, maskBits=0x001
                )
            )
            leg.ground_contact = False
            self.legs.append(leg)
            
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(0, 0), localAnchorB=(i * (ROCKET_H_WIDTH * 0.8), leg_h),
                enableMotor=True, enableLimit=True, maxMotorTorque=2500.0, motorSpeed=0.3 * i
            )
            if i == -1: rjd.lowerAngle, rjd.upperAngle = 0.4, 0.9
            else: rjd.lowerAngle, rjd.upperAngle = -0.9, -0.4
            self.world.CreateJoint(rjd)
            
        self.drawlist = [self.lander] + self.legs