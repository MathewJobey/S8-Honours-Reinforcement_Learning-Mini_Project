import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)
from .settings import *
from .physics import ContactDetector
from .visualizer import RocketVisualizer

class Phase3Final(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.world = None
        self.lander = None
        self.main_engine_power = 0.0
        self.landing_status = "IN_PROGRESS"
        self.visualizer = RocketVisualizer(self)
        self.start_y = 500.0 # Set the default drop height, and create a ceiling that is 20% higher
        self.max_altitude = self.start_y * 1.2
        
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
        
        self._create_terrain()
        self._create_rocket()

        start_y = self.start_y
        
        """start_x = self.np_random.uniform(-30.0, 30.0)
        self.lander.position = (start_x, start_y)
        self.lander.angle = self.np_random.uniform(-math.pi, math.pi)
        #4. Pick a random falling speed between a hover (0.0) and terminal velocity (-140.0)
        # We use negative numbers because the rocket is moving down the Y-axis
        start_vy = self.np_random.uniform(-50.0, 0.0)
        self.lander.linearVelocity = (0, start_vy)"""
        # THE FIX: Spawn perfectly in the center, perfectly upright!
        self.lander.position = (0.0, start_y)
        self.lander.angle = 0.0
        
        # Pick a random falling speed
        start_vy = self.np_random.uniform(-50.0, 0.0)
        self.lander.linearVelocity = (0, start_vy)
        
        # Ensure it is not spinning when it spawns
        self.lander.angularVelocity = 0.0

        return self.step(np.array([0, 0, 0]))[0], {}     
    
    """APPLY FORCES BASED ON THE ACTIONS TAKEN BY THE AGENT. THIS INCLUDES MAIN ENGINE THRUST, SIDE THRUSTERS, AND AERODYNAMIC DRAG."""
    def _apply_forces(self, main_power, center_power, nose_power):
        angle = self.lander.angle
        vel = self.lander.linearVelocity
        
        # 1. Main Engine Force pushed on COM
        main_force_x = float(-math.sin(angle) * main_power * MAIN_ENGINE_POWER)
        main_force_y = float(math.cos(angle) * main_power * MAIN_ENGINE_POWER)
        self.lander.ApplyForceToCenter((main_force_x, main_force_y), wake=True)
        
        # 2. Center Thrusters (Pure Translation/Sliding)
        center_force_mag = float(center_power * SIDE_ENGINE_POWER)
        c_force_x = float(-center_force_mag * math.cos(angle))
        c_force_y = float(-center_force_mag * math.sin(angle))
        
        # We changed this from ApplyLinearImpulse to ApplyForce!
        self.lander.ApplyForce(
            (c_force_x, c_force_y), 
            self.lander.worldCenter, 
            wake=True
        )
        
        # 3. Nose Thrusters (Pure Torque/Tilting)
        # We calculate the exact middle of the nose cone!
        thruster_y = ROCKET_HEIGHT + (NOSE_HEIGHT / 2.0)
        
        # Calculate how far the nose thrusters are from the true balancing point
        lever_arm = thruster_y - self.lander.localCenter.y
        
        # Calculate the pure twisting force (Torque)
        torque = float(nose_power * NOSE_ENGINE_POWER * lever_arm)
        
        # Apply pure rotation without any horizontal pushing!
        self.lander.ApplyTorque(torque, wake=True)
        
        # 4. AERODYNAMIC DRAG
        self.current_drag = 0.0
        
        # Calculates belly-flop drag! (Upright = 0.1, Sideways = 1.0)
        exposed_area = abs(math.sin(angle)) * 0.9 + 0.1 
        velocity_squared = vel.x**2 + vel.y**2
        
        if velocity_squared > 0:
            drag_magnitude = 0.5 * velocity_squared * exposed_area
            self.current_drag = drag_magnitude
            
            speed = math.sqrt(velocity_squared)
            drag_x = -(vel.x / speed) * drag_magnitude
            drag_y = -(vel.y / speed) * drag_magnitude
            
            # --- THE FIX: Apply drag perfectly to the middle! ---
            # This slows the rocket down without causing any artificial spinning
            self.lander.ApplyForceToCenter((drag_x, drag_y), wake=True)
    
    """THE STEP FUNCTION TO ADVANCE THE ENVIRONMENT BY ONE TIME STEP. THIS IS CALLED AT EVERY TIME STEP OF THE EPISODE."""
    def step(self, action):
        action = np.clip(action, -1, 1) 
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        #center_side_power = np.clip(action[1], -1.0, 1.0)
        #nose_side_power = np.clip(action[2], -1.0, 1.0)
        
        center_side_power = 0.0
        nose_side_power = 0.0
        
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
        if main_engine_power > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * main_engine_power
            
        # Added fuel consumption for center side thrusters, at 50% the rate of the main engine since they are smaller and less powerful
        if abs(center_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(center_side_power) * 0.5
            
        # Added fuel consumption for nose thrusters, also at 50% the rate of the main engine
        if abs(nose_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(nose_side_power) * 0.3

        state = self._get_state()
        reward, terminated, truncated = self._compute_reward(state)
        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}
    
    """NORMALIZATION OF THE OBSERVATION SPACE. THIS FUNCTION TAKES THE RAW PHYSICS DATA AND SCALES IT TO A RANGE THAT IS EASIER FOR THE AI TO LEARN FROM. THIS INCLUDES NORMALIZING POSITION, VELOCITY, ANGLE, AND FUEL LEVEL."""
    def _get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        world_width_half = VIEWPORT_W / SCALE / 2 
        
        # --- THE FIX 1: WRAP THE ANGLE ---
        # This math forces the angle to always stay between -3.14 and +3.14
        norm_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        
        return [
            pos.x / world_width_half,       # 1. Horizontal Position (-1 to 1)
            pos.y / self.max_altitude,      # 2. Vertical Position (0 to 1)
            vel.x / 150.0,                  # 3. Horizontal Velocity 
            vel.y / 150.0,                  # 4. Vertical Velocity
            norm_angle,                     # 5. Angle
            self.lander.angularVelocity/6.0,# 6. Angular Velocity
            self.fuel_left / INITIAL_FUEL   # 7. Fuel
        ]
        
    """REWARD FUNCTION TO CALCULATE THE REWARD FOR THE CURRENT STATE. THIS FUNCTION TAKES INTO ACCOUNT THE DISTANCE TO THE LANDING PAD, THE VELOCITY, THE ANGLE, AND THE FUEL LEFT TO COMPUTE A COMPREHENSIVE REWARD SIGNAL THAT ENCOURAGES SAFE AND EFFICIENT LANDINGS."""   
    def _compute_reward(self, state):
        reward = 0.0
        terminated = False 
        truncated = False
        
        # THE FIX: The reward magnets must also track the Center of Mass!
        pos = self.lander.worldCenter
        vel = self.lander.linearVelocity
        
        # Standardize the angle
        norm_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        tilt_rad = abs(norm_angle)
        x_range = VIEWPORT_W / SCALE / 2

        # ==========================================
        # STEP 1: THE SANDBOX
        # ==========================================
        if abs(pos.x) >= x_range or pos.y > self.max_altitude:
            self.landing_status = "CRASH"
            return -200.0, True, False

        # ==========================================
        # STEP 2: POSTURE & RADAR (The Magnets)
        # ==========================================
        # THE FIX: Changed from * 10.0 to * 5.0 to soften the panic!
        posture_reward = math.exp(-tilt_rad * 5.0)
        reward += posture_reward * 10.0
        
        # THE FIX: Doubled the spin penalty to kill the wiggling!
        reward -= abs(self.lander.angularVelocity) * 20.0

        # Centering Magnet (Trains Side Thrusters)
        center_reward = math.exp(-abs(pos.x))
        reward += center_reward * 10.0
        reward -= abs(vel.x) * 5.0

        # ==========================================
        # STEP 3: GLIDE SLOPE (The Main Engine Trainer)
        # ==========================================
        # THE FIX: Because the Center of Mass is 5.5 meters above the engine, 
        # we subtract 5.5 from pos.y so the speed limit properly reaches 0 at the ground!
        true_altitude = max(0.0, pos.y - 5.5)
        target_speed = -max(1.0, math.sqrt(true_altitude))
        
        # If it falls faster than the target speed, apply a heavy penalty!
        if vel.y < target_speed:
            speed_violation = abs(vel.y - target_speed)
            reward -= speed_violation * 2.0
            
        # ... Keep your Fuel Taxes and Terminal State exactly the same below this! ...
        # ==========================================
        # STEP 4: THE BRAKES & EFFICIENCY
        # ==========================================
        # Heavy penalty for spinning (This kills the wiggling!)
        reward -= abs(self.lander.angularVelocity) * 10.0
        
        # Penalty for sliding sideways
        reward -= abs(vel.x) * 2.0
        
        # Tiny fuel taxes to prevent infinite hovering
        reward -= 0.05  # Time tax
        reward -= (self.main_engine_power * 0.1)  

        # ==========================================
        # STEP 5: TERMINAL STATE
        # ==========================================
        if self.game_over:
            terminated = True
            
            # Did it land perfectly?
            is_slow = abs(vel.y) < 1.0       
            is_straight = tilt_rad < 0.15    
            is_on_pad = abs(pos.x) < (PAD_WIDTH_METERS / 2.0)
            
            if is_slow and is_straight and is_on_pad:
                reward += 1000.0 + (self.fuel_left * 2.0)
                self.landing_status = "LANDED"
            else:
                reward -= 500.0
                self.landing_status = "CRASH"

        return reward, terminated, truncated
    
    """RENDERING FUNCTION TO VISUALIZE THE ENVIRONMENT. THIS USES THE RocketVisualizer CLASS TO DRAW THE ROCKET, TERRAIN, AND OTHER ELEMENTS ON THE SCREEN."""
    def render(self):
        return self.visualizer.render(self.render_mode)

    """CLOSE FUNCTION TO CLEAN UP RESOURCES WHEN THE ENVIRONMENT IS DONE. THIS ENSURES THAT THE VISUALIZER IS PROPERLY CLOSED AND THAT THE PHYSICS WORLD IS DESTROYED TO FREE UP MEMORY."""
    def close(self):
        self.visualizer.close()
        self._destroy()
        
    """DESTROY FUNCTION TO CLEAN UP THE PHYSICS WORLD. THIS IS CALLED WHEN THE ENVIRONMENT IS RESET OR CLOSED TO ENSURE THAT ALL PHYSICS OBJECTS ARE PROPERLY REMOVED AND THAT THERE ARE NO MEMORY LEAKS."""
    def _destroy(self):
        if not self.world: return
        self.world = None
        
    """FUNCTION TO CREATE THE TERRAIN, INCLUDING THE FLAT DIRT AND THE LANDING PAD. THIS SETS UP THE PHYSICS BODIES FOR THE GROUND AND THE PAD, AND TAGS THEM WITH USER DATA SO THAT THE CONTACT DETECTOR CAN IDENTIFY THEM DURING COLLISIONS."""   
    def _create_terrain(self):
        # We calculate how wide the physical world is in meters
        world_width = VIEWPORT_W / SCALE
        
        # 1. THE FLAT DIRT
        # Create an unbreakable static body (it has no weight and never moves)
        self.ground = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=edgeShape(vertices=[(-world_width, 0), (world_width, 0)])
        )
        # Tag it so the ContactDetector knows this is a crash zone!
        self.ground.userData = "ground"
        
        # 2. THE LANDING PAD
        # Create a raised concrete block exactly in the center
        self.pad = self.world.CreateStaticBody(
            position=(0, PAD_HEIGHT_METERS / 2.0),
            fixtures=fixtureDef(
                # box() takes half-width and half-height to draw a perfect rectangle
                shape=polygonShape(box=(PAD_WIDTH_METERS / 2.0, PAD_HEIGHT_METERS / 2.0)),
                friction=0.8,     # Very grippy so the rocket doesn't slide off easily!
                restitution=0.0   # Solid concrete, zero bounce
            )
        )
        # Tag it so the ContactDetector knows this is the safe zone!
        self.pad.userData = "pad"
        
    """FUNCTION TO CREATE THE ROCKET BODY. THIS COMPOUNDS TWO FIXTURES TOGETHER TO FORM THE ROCKET: A HEAVY MAIN HULL AND A LIGHTWEIGHT NOSE CONE. THIS ALLOWS US TO SIMULATE MORE REALISTIC PHYSICS, SUCH AS THE ROCKET TIPPING OVER IF THE NOSE CONE HITS THE GROUND FIRST. WE ALSO SET A USER DATA TAG ON THE ROCKET SO THAT THE CONTACT DETECTOR CAN IDENTIFY IT DURING COLLISIONS."""
    def _create_rocket(self):
        initial_x = 0
        initial_y = self.start_y
        
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
                    # Density is lighter because the cone is mostly empty space!
                    density=(10.0 / 2.25), friction=0.5, categoryBits=0x0010, maskBits=0x001, restitution=0.0
                )
            ]
        )
        self.lander.color1 = LANDER_COLOR
        
        # --- ADD A NAME TAG ---
        self.lander.userData = "rocket"
        
        # Set drawlist to only include the main hull
        self.drawlist = [self.lander]