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
            self.lander.GetWorldPoint(localPoint=(0, 1)), 
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
        
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (VIEWPORT_H / SCALE / 2)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            self.lander.angularVelocity,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.fuel_left / INITIAL_FUEL,
            self.wind_x / WIND_POWER_MAX,
            self.wind_y / WIND_POWER_MAX
        ]
        
        # 5. Calculate Reward
        reward = 0
        terminated = False 
        truncated = False
        
        dist = math.sqrt(state[0]**2 + state[1]**2)
        shaping = -100 * dist - 100 * math.sqrt(state[2]**2 + state[3]**2) - 100 * abs(state[4])
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        reward -= main_engine_power * 0.10
        
        if self.game_over or abs(state[0]) >= 1.0: 
            terminated = True
            reward = -100
        elif not self.lander.awake: 
            terminated = True
            reward = +100
            
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
        # Random start x (-10 to 10 meters roughly)
        initial_x = self.np_random.uniform(-0.3, 0.3) * (VIEWPORT_W / SCALE / 2)
        initial_y = VIEWPORT_H / SCALE
        
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(-14/SCALE, 0), (14/SCALE, 0), (14/SCALE, 40/SCALE), (-14/SCALE, 40/SCALE)]),
                density=5.0, friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0
            )
        )
        self.lander.color1 = LANDER_COLOR
        
        self.legs = []
        for i in [-1, 1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i*10/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(3/SCALE, 15/SCALE)),
                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001
                )
            )
            leg.ground_contact = False
            self.legs.append(leg)
            
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(0, 0), localAnchorB=(i * 10/SCALE, 15/SCALE),
                enableMotor=True, enableLimit=True, maxMotorTorque=2000.0, motorSpeed=0.3 * i
            )
            if i == -1: rjd.lowerAngle, rjd.upperAngle = 0.4, 0.9
            else: rjd.lowerAngle, rjd.upperAngle = -0.9, -0.4
            self.world.CreateJoint(rjd)
            
        self.drawlist = [self.lander] + self.legs