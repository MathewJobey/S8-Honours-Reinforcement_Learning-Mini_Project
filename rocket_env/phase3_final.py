import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef)
from .utils import get_true_altitude, calculate_distance_reward, calculate_posture_reward, calculate_x_reward
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
        
        # Action space to 3 continuous sliders: [Main, Center, Nose]
        # We give the AI a perfectly symmetric box from -1.0 to 1.0 to keep the math happy
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
        
        self.step_id = 0
        self.max_steps = 2000
        
        self.current_drag = 0.0
        
        self._create_terrain()
        self._create_rocket()

        start_y = self.start_y
        
        # --- THE FIX: DOMAIN RANDOMIZATION ---
        # 1. Pick a random X position between -30 and 30
        start_x = self.np_random.uniform(-30.0, 30.0)
        self.lander.position = (start_x, start_y)
        
        # 2. Force a Belly-Flop! Pick an angle between -90 and +90 degrees (-1.57 to 1.57 rads)
        self.lander.angle = self.np_random.uniform(-math.pi, math.pi)
        
        # 3. Force it to already be falling fast! (-80 to -40 m/s)
        # Now if it fires the engine, it just slows down instead of reversing gravity instantly.
        start_vy = self.np_random.uniform(-100.0, 0.0)
        self.lander.linearVelocity = (0, start_vy)
        
        # Ensure it is not spinning when it spawns
        self.lander.angularVelocity = 0.0
        
        self.prev_y = self.lander.worldCenter.y
        self.impact_speed = None  # <--- THE CRASH MEMORY RESET!

        state = self._get_state()
        return np.array(state, dtype=np.float32), {}
    
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
        self.step_id += 1
        # 1. Safety First: Ensure the AI's raw slider values never exceed -1.0 to 1.0
        action = np.clip(action, -1.0, 1.0) 
        
        # 2. Main Engine: The AI might output a negative number, so we crush it to 0.0
        main_engine_power = np.clip(action[0], 0.0, 1.0)
        
        # 3. Side Thrusters: We can use the AI's exact decimal values!
        # THE FIX: Removed the extra -1.0, 1.0 arguments!
        center_side_power = float(action[1])
        nose_side_power = float(action[2])
        
        if self.fuel_left <= 0:
            main_engine_power = 0.0
            center_side_power = 0.0
            nose_side_power = 0.0   
            
        self.main_engine_power = main_engine_power
        self.center_side_power = center_side_power
        self.nose_side_power = nose_side_power
        
        vel = self.lander.linearVelocity
        self.pre_step_velocity = math.sqrt(vel.x**2 + vel.y**2)
        
        pre_physics_vy = vel.y
        self.pre_step_vy = vel.y
        self.prev_y = self.lander.worldCenter.y
        
        self._apply_forces(main_engine_power, center_side_power, nose_side_power)
        self.world.Step(1.0 / FPS, 6, 2)

        post_physics_vy = self.lander.linearVelocity.y
        
        # The G-Force Check
        if abs(post_physics_vy - pre_physics_vy) > 3.0:
            self.game_over = True
            if getattr(self, 'impact_speed', None) is None:
                self.impact_speed = pre_physics_vy
                
        if self.game_over and getattr(self, 'impact_speed', None) is None:
            self.impact_speed = pre_physics_vy

        # Fuel Costs
        if main_engine_power > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * main_engine_power
        self.fuel_left = max(0.0, self.fuel_left)
            
        if abs(center_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(center_side_power) * 0.2
            
        if abs(nose_side_power) > 0:
            self.fuel_left -= (FUEL_CONSUMPTION_RATE / FPS) * abs(nose_side_power) * 0.1

        state = self._get_state()
        
        # --- THE FIX 2: Compute reward without potentials ---
        reward, terminated, truncated = self._compute_reward(state)
        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}
    
    """NORMALIZATION OF THE OBSERVATION SPACE. THIS FUNCTION TAKES THE RAW PHYSICS DATA AND SCALES IT TO A RANGE THAT IS EASIER FOR THE AI TO LEARN FROM. THIS INCLUDES NORMALIZING POSITION, VELOCITY, ANGLE, AND FUEL LEVEL."""
    def _get_state(self):
        pos = self.lander.worldCenter
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
            
            # --- THE TINY TWEAK ---
            norm_angle / math.pi,           # 5. Angle (-1 to 1) 
            
            self.lander.angularVelocity/6.0,# 6. Angular Velocity
            self.fuel_left / INITIAL_FUEL   # 7. Fuel
        ]
    
    """REWARD FUNCTION TO CALCULATE THE REWARD FOR THE CURRENT STATE. THIS FUNCTION TAKES INTO ACCOUNT THE DISTANCE TO THE LANDING PAD, THE VELOCITY, THE ANGLE, AND THE FUEL LEFT TO COMPUTE A COMPREHENSIVE REWARD SIGNAL THAT ENCOURAGES SAFE AND EFFICIENT LANDINGS."""   
    def _compute_reward(self, state):
        pos = self.lander.worldCenter
        vel = self.lander.linearVelocity
        true_altitude = get_true_altitude(self.lander, ROCKET_H_WIDTH, ROCKET_HEIGHT)

        reward = 0.0
        terminated = False
        truncated = False

        # Calculate basic physics values
        theta = abs((self.lander.angle + math.pi) % (2 * math.pi) - math.pi)
        v = math.sqrt(vel.x**2 + vel.y**2)
        
        # ==========================================
        # 0. THE EXTREME SPAWN KILL SWITCH
        # ==========================================
        max_x = VIEWPORT_W / SCALE / 2
        
        is_off_sides = abs(pos.x) >= max_x
        is_above_ceiling = pos.y > self.max_altitude
        is_below_floor = true_altitude < -2.0 
        
        # If the 100m/s spawn flings it off the map, instantly kill it and penalize!
        if is_off_sides or is_above_ceiling or is_below_floor:
            self.landing_status = "OUT_OF_BOUNDS"
            return -100.0, True, False
        
        # ==========================================
        # 1. The Distance "Pull" (Dense Reward)
        # ==========================================
        max_x = VIEWPORT_W / SCALE / 2
        max_dist = math.sqrt(max_x**2 + self.max_altitude**2)
        
        dist_x = abs(pos.x)
        dist_y = max(0.0, true_altitude)
        
        dist_norm = min(1.0, math.sqrt(dist_x**2 + dist_y**2) / max_dist)
        dist_reward = 0.1 * (1.0 - dist_norm)

        # ==========================================
        # 2. The Posture "Rule" (Dense Reward)
        # ==========================================
        posture_reward = 0.0
        
        # math.pi / 36 is exactly 5 degrees. The ultimate strict zone!
        tolerance = math.pi / 36
        
        if theta <= tolerance:
            # If it leans 5 degrees or less, it gets a perfect score
            posture_reward = 0.1
        else:
            # Step A: Figure out how much it is leaning beyond the 5 allowed degrees
            extra_lean = theta - tolerance
            
            # Step B: Figure out the maximum possible "wrong" lean (180 degrees minus the 5 allowed)
            max_wrong_lean = math.pi - tolerance
            
            # Step C: Turn that into a percentage (0.0 to 1.0)
            proportion = extra_lean / max_wrong_lean
            
            # Step D: Clamp it to safely stay between 0% and 100%
            proportion = min(1.0, max(0.0, proportion)) 
            
            # Step E: Deduct that percentage from the maximum 0.1 reward
            posture_reward = 0.1 * (1.0 - proportion)

        # Total Dense Reward for this frame
        dense_reward = dist_reward + posture_reward
        reward += dense_reward

        # ==========================================
        # 3 & 4. The Terminal Rewards (Crash & Jackpot)
        # ==========================================
        
        # The author's trigger for ending the game
        if getattr(self, 'game_over', False) or true_altitude <= 0.1:
            terminated = True
            
            # The Author's Speed Bonus & Time Variables
            time_remaining = max(0, self.max_steps - self.step_id)
            speed_bonus = 5.0 * math.exp(-v / 10.0)
            
            # The Author's strict landing conditions
            is_straight = theta <= (math.pi / 36)             # Under 5 degrees
            is_on_pad = dist_x < (PAD_WIDTH_METERS / 2.0)    # Inside pad bounds
            is_safe_speed = v < 5.0                         # Under 5 m/s
            
            if is_straight and is_on_pad and is_safe_speed:
                self.landing_status = "LANDED"
                # THE JACKPOT
                reward = (1.0 + speed_bonus) * time_remaining
            else:
                self.landing_status = "CRASH"
                # THE CRASH PENALTY
                reward = (dense_reward + speed_bonus) * time_remaining

        # Check if we ran out of time safely
        elif self.step_id >= self.max_steps:
            truncated = True

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
        
        # 1. Single Body: Perfectly stable 100kg rectangular box!
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=[
                # FIXTURE 1: The Main Hull (No nose cone physics!)
                fixtureDef(
                    shape=polygonShape(vertices=[
                        (-ROCKET_H_WIDTH, 0), 
                        (ROCKET_H_WIDTH, 0), 
                        (ROCKET_H_WIDTH, ROCKET_HEIGHT), 
                        (-ROCKET_H_WIDTH, ROCKET_HEIGHT)
                    ]),
                    # Density of 5.55 across this box equals exactly 100kg!
                    density=5.55, friction=0.5, categoryBits=0x0010, maskBits=0x001, restitution=0.0
                )
            ]
        )
        self.lander.color1 = LANDER_COLOR
        self.lander.userData = "rocket"
        self.drawlist = [self.lander]