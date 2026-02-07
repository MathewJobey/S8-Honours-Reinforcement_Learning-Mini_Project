import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import pygame


# --- 1. CONFIGURATION CONSTANTS ---
# (Tweaking these changes how the "game" feels)

FPS = 50
SCALE = 30.0  # affects how large the screen elements are
VIEWPORT_W = 600
VIEWPORT_H = 400

# Rocket Physics
MAIN_ENGINE_POWER = 13
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
        self.screen = None
        self.clock = None
        
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
        
        # Observation Space:
        # We define the range of values the agent can see (Infinity means no limit)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), # 11 inputs total (including fuel and wind)
            dtype=np.float32
        )
        
        '''WORLD SETUP'''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        
        # 1. Create the Physics World
        # Gravity is -10 (Earth-like). The rocket must fight this to fly.
        self.world = Box2D.b2World(gravity=(GRAVITY_X, GRAVITY_Y))
        
        # 2. Setup Collision Detection
        # We need to know when the legs touch the ground to stop the game.
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.game_over = False
        self.prev_shaping = None
        self.fuel_left = INITIAL_FUEL

        # 3. Randomize the Environment
        self._generate_wind()

        # 4. Build the Objects
        self._create_terrain()
        self._create_rocket()

        # 5. Return the first observation
        # We run one small step with (0,0) action to get the initial numbers
        return self.step(np.array([0, 0]))[0], {}
    
    def step(self, action):
        # --- 1. PREPARE ACTION ---
        # Action is an array of 2 numbers: [Main Engine, Side Thrusters]
        # We clip them to make sure the AI doesn't try to use 1000% power.
        action = np.clip(action, -1, 1) 
        
        # Map raw action to engine power
        # Main Engine: 0.0 (off) to 1.0 (full)
        # Side Engine: -1.0 (left) to 1.0 (right)
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        side_engine_power = np.clip(action[1], -1.0, 1.0)
        
        # If out of fuel, engines die
        if self.fuel_left <= 0:
            main_engine_power = 0
            side_engine_power = 0
            
        # --- 2. APPLY FORCES (PHYSICS) ---
        
        # A. Main Engine Force
        # We calculate the force vector. Since the rocket rotates, "Up" changes direction!
        # math.sin and math.cos help us find the X and Y components of the force.
        angle = self.lander.angle
        main_force_x = -math.sin(angle) * main_engine_power * MAIN_ENGINE_POWER
        main_force_y = math.cos(angle) * main_engine_power * MAIN_ENGINE_POWER
        
        # Apply force to the center of the rocket
        self.lander.ApplyForceToCenter(
            (main_force_x, main_force_y), 
            wake=True # Ensure physics engine wakes up
        )
        
        # B. Side Thruster Force (Steering)
        # We apply an "Impulse" (sudden push) to the TOP of the rocket to tilt it.
        side_force = side_engine_power * SIDE_ENGINE_POWER
        self.lander.ApplyLinearImpulse(
            (-side_force * math.cos(angle), -side_force * math.sin(angle)), # Direction
            self.lander.GetWorldPoint(localPoint=(0, 1)), # Apply at Top of rocket
            wake=True
        )

        # C. Wind Force
        # Wind pushes the rocket constantly in one direction
        self.lander.ApplyForceToCenter(
            (self.wind_x, self.wind_y), 
            wake=True
        )

        # --- 3. RUN SIMULATION ---
        # Run 1/50th of a second of physics
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # --- 4. UPDATE STATE & FUEL ---
        if main_engine_power > 0:
            self.fuel_left -= FUEL_CONSUMPTION_RATE / FPS * main_engine_power

        # Get the new position and velocity
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # Create the observation vector (what the AI sees)
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2), # X (Normalized)
            (pos.y - (VIEWPORT_H / SCALE / 2)) / (VIEWPORT_H / SCALE / 2), # Y (Normalized)
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS, # Velocity X
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS, # Velocity Y
            self.lander.angle,
            self.lander.angularVelocity,
            1.0 if self.legs[0].ground_contact else 0.0, # Left Leg Contact
            1.0 if self.legs[1].ground_contact else 0.0, # Right Leg Contact
            self.fuel_left / INITIAL_FUEL,               # Fuel Remaining
            self.wind_x / WIND_POWER_MAX,                # Wind X
            self.wind_y / WIND_POWER_MAX                 # Wind Y
        ]
        
        # --- 5. CALCULATE REWARD ---
        reward = 0
        terminated = False 
        truncated = False
        
        # Shaping Reward: Help the AI find the center
        # "dist" is distance from the landing pad (0,0)
        dist = math.sqrt(state[0]**2 + state[1]**2)
        
        # We penalize distance, speed, and tilting.
        # The closer and slower it is, the less negative the score.
        shaping = -100 * dist - 100 * math.sqrt(state[2]**2 + state[3]**2) - 100 * abs(state[4])
        
        # Reward is the *improvement* from last frame
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Fuel Penalty: Wasting gas costs points
        reward -= main_engine_power * 0.10
        
        # --- 6. CHECK GAME OVER ---
        
        # Crash Condition: Body hit ground OR flew too far away
        if self.game_over or abs(state[0]) >= 1.0: 
            terminated = True
            reward = -100 # Big penalty for crashing
            
        # Landing Condition: Sleeping (stopped moving) and upright
        elif not self.lander.awake: 
            terminated = True
            reward = +100 # Big reward for safe landing!
            
        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}
    

    def _generate_wind(self):
        # Generates random wind force for this specific episode
        self.wind_x = self.np_random.uniform(-WIND_POWER_MAX, WIND_POWER_MAX)
        self.wind_y = self.np_random.uniform(-WIND_POWER_MAX, WIND_POWER_MAX)

    def _create_terrain(self):
        # Create the ground at y=0
        # Friction=0.8 means the ground is rough (rocket won't slide forever)
        self.ground = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(-VIEWPORT_W / SCALE, 0), (VIEWPORT_W / SCALE, 0)])
        )
        self.ground.friction = 0.8

    def _create_rocket(self):
        # Start at the top center
        initial_x = 0  
        initial_y = VIEWPORT_H / SCALE 
        
        # --- A. The Main Hull ---
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(-14/SCALE, 0), (14/SCALE, 0), (14/SCALE, 40/SCALE), (-14/SCALE, 40/SCALE)]),
                density=5.0,  # Heavy enough to fall fast
                friction=0.1,
                categoryBits=0x0010, # Category: Rocket
                maskBits=0x001,      # Collides with: Ground
                restitution=0.0      # No bounce
            )
        )
        self.lander.color1 = LANDER_COLOR
        
        # --- B. The Landing Legs ---
        self.legs = []
        for i in [-1, 1]: # -1 (Left), 1 (Right)
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i*10/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(3/SCALE, 15/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001
                )
            )
            leg.ground_contact = False
            self.legs.append(leg)
            
            # --- C. The Leg Joints (Suspension) ---
            # Connect leg to hull with a spring-like motor
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * 10/SCALE, 15/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=2000.0,
                motorSpeed=0.3 * i  # Push legs outward
            )
            
            # Limit how much the legs can wiggle
            if i == -1:
                rjd.lowerAngle = 0.4 
                rjd.upperAngle = 0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.4
                
            self.world.CreateJoint(rjd)
            
        self.drawlist = [self.lander] + self.legs

    def _destroy(self):
        if not self.world: return
        self.world = None
        
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        # This function is called automatically by Box2D when two shapes hit each other.
        # We check "user data" to see *what* hit *what*.
        
        # In Box2D, A and B are the two objects colliding.
        # We don't know which is which, so we check both.
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True # The main hull hit the ground! CRASH!
            
        # Check if the LEGS hit the ground
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    
    def EndContact(self, contact):
        # This is called when objects stop touching (e.g., rocket bounces up)
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False