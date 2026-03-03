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

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Expand action space to 3 variables: [Main, Center, Nose]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
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
        # --- FIX 1: THE TILT TEST ---
        # We spawn the rocket slightly tilted (between -15 and +15 degrees).
        # This FORCES the AI to instantly fire the nose thrusters to straighten out!
        self.lander.angle = self.np_random.uniform(-0.25, 0.25)
        
        start_vy = self.np_random.uniform(-90.0, -60.0)
        self.lander.linearVelocity = (0, start_vy)

        return self.step(np.array([0, 0, 0]))[0], {}
    
    def _apply_forces(self, main_power, center_power, nose_power):
        angle = self.lander.angle
        vel = self.lander.linearVelocity
        
        # 1. Main Engine Force
        main_force_x = float(-math.sin(angle) * main_power * MAIN_ENGINE_POWER)
        main_force_y = float(math.cos(angle) * main_power * MAIN_ENGINE_POWER)
        self.lander.ApplyForceToCenter((main_force_x, main_force_y), wake=True)
        
        # 2. Center Thrusters (Pure Translation/Sliding)
        center_force = float(center_power * SIDE_ENGINE_POWER)
        c_impulse_x = float(-center_force * math.cos(angle))
        c_impulse_y = float(-center_force * math.sin(angle))
        self.lander.ApplyLinearImpulse(
            (c_impulse_x, c_impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, ROCKET_HEIGHT / 2.0)), 
            wake=True
        )

        # 3. Nose Thrusters (Torque/Tilting)
        # Removed the 0.5 handicap. Full power so it can snap the rocket upright!
        nose_force = float(nose_power * SIDE_ENGINE_POWER)
        n_impulse_x = float(-nose_force * math.cos(angle))
        n_impulse_y = float(-nose_force * math.sin(angle))
        self.lander.ApplyLinearImpulse(
            (n_impulse_x, n_impulse_y), 
            self.lander.GetWorldPoint(localPoint=(0, ROCKET_HEIGHT * 0.85)), 
            wake=True
        )
        
        self.current_drag = 0.0
        self.current_torque = 0.0

        # 4. AERODYNAMIC DRAG
        exposed_area = abs(math.sin(angle)) * 0.9 + 0.1 
        velocity_squared = vel.x**2 + vel.y**2
        if velocity_squared > 0:
            drag_magnitude = 0.5 * velocity_squared * exposed_area
            self.current_drag = drag_magnitude
            
            speed = math.sqrt(velocity_squared)
            drag_x = -(vel.x / speed) * drag_magnitude
            drag_y = -(vel.y / speed) * drag_magnitude
            
            center_y = ROCKET_HEIGHT / 2.0
            offset_y = center_y + 0.05
            drag_point = self.lander.GetWorldPoint(localPoint=(0, offset_y))
            
            self.lander.ApplyForce((drag_x, drag_y), drag_point, wake=True)
            
    def step(self, action):
        action = np.clip(action, -1, 1) 
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        center_side_power = np.clip(action[1], -1.0, 1.0)
        nose_side_power = np.clip(action[2], -1.0, 1.0)
        
        if self.fuel_left <= 0:
            main_engine_power = 0
            center_side_power = 0
            nose_side_power = 0
            
        self.main_engine_power = main_engine_power
        self.center_side_power = center_side_power
        self.nose_side_power = nose_side_power
        
        vel = self.lander.linearVelocity
        self.pre_step_velocity = math.sqrt(vel.x**2 + vel.y**2)
        
        self._apply_forces(main_engine_power, center_side_power, nose_side_power)
        self.world.Step(1.0 / FPS, 6, 2)

        # --- FIX 2: STEERING FUEL COST ---
        # If steering is completely free, the AI will jitter. 
        # Adding a tiny fuel cost forces it to fly cleanly and efficiently.
        if main_engine_power > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * main_engine_power
        if abs(center_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(center_side_power) * 0.1
        if abs(nose_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(nose_side_power) * 0.05

        state = self._get_state()
        reward, terminated, truncated = self._compute_reward(state, main_engine_power)

        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}
    
    def _get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # 1. SCALE CONSTANTS
        # Viewport width in meters: 800 / 10 = 80m. Half = 40m.
        world_width_half = VIEWPORT_W / SCALE / 2 
        
        # Max Altitude: We spawn at 350m, so let's normalize against 400m.
        # This gives a value like 0.875, which is perfect for Neural Networks (0 to 1).
        MAX_ALTITUDE = 600.0
        
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
        
        x_range = VIEWPORT_W / SCALE / 2
        y_range = 500.0 
        
        dist_x = abs(pos.x)
        
        # --- FIX 3: THE PERFECT HOVER RADAR ---
        # Instead of targeting the pad (1.0m), it targets 0.3m ABOVE the pad (1.3m).
        hover_target_y = PAD_HEIGHT_METERS + 0.3
        dist_y = abs(pos.y - hover_target_y)
        
        dist_norm = (dist_x / x_range) + (dist_y / y_range)
        dist_reward = 1.0 - min(1.0, dist_norm)
        
        tilt_rad = abs(self.lander.angle)
        pose_reward = 1.0 - min(1.0, tilt_rad / 1.5)
        
        reward += 0.1 * dist_reward
        reward += 0.1 * pose_reward
        reward -= 0.15
        reward -= main_engine_power * 0.05
        # --- THE FIX: STRICT ALIGNMENT PENALTY ---
        # 1. Bleed points for every degree it leans away from perfect 0.
        reward -= (tilt_rad * 2.0) 
        # 2. Bleed points if it is spinning or wobbling.
        reward -= (abs(self.lander.angularVelocity) * 0.5)

        # Directional Radar (Anti-Reversing)
        if vel.y > 0.5:
            reward -= vel.y * 1.0 
        else:
            allowed_speed = max(2.0, pos.y * 0.5)
            current_fall_speed = abs(vel.y)
            if current_fall_speed > allowed_speed:
                speed_violation = current_fall_speed - allowed_speed
                reward -= (speed_violation / 10.0)

        # --- FIX 4: STRICT HOVER BONUS ---
        # It ONLY gets rich if it is between 0.1m and 0.5m above the pad, 
        # nearly centered, incredibly slow, and perfectly straight.
        if (PAD_HEIGHT_METERS + 0.1) < pos.y < (PAD_HEIGHT_METERS + 0.5):
            if abs(pos.x) < 0.5 and abs(vel.y) < 0.5 and tilt_rad < 0.1:
                reward += 5.0 # Massive reward for hovering here!

        velocity_total = math.sqrt(vel.x**2 + vel.y**2)
        
        if self.game_over:
            terminated = True
            if self.pre_step_velocity < 3.0 and tilt_rad < 0.2: 
                distance_from_center = abs(pos.x)
                reward = 100.0 * math.exp(-distance_from_center)
                self.landing_status = "LANDED"
            else:
                reward = -100 - (self.pre_step_velocity * 2.0)
                self.landing_status = "CRASH"
                
        elif abs(pos.x) >= x_range or pos.y > 1200.0:
            terminated = True
            reward = -100
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
        initial_x = 0
        initial_y = 500.0
        
        # 1. Main Body ONLY (No Legs)
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
        
        # --- THE FIX: ADD A NAME TAG ---
        self.lander.userData = "rocket"
        
        # Set drawlist to only include the main hull
        self.drawlist = [self.lander]