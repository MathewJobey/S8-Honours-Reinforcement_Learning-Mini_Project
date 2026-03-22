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
        
        # Contains 12 variables: [X, Y, vX, vY, Angle, vAngle, Fuel, Hover, Speedometer, Prev Main, Prev Center, Prev Nose]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
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
        
        self.current_drag = 0.0
        
        self._create_terrain()
        self._create_rocket()

        start_y = self.start_y
        
        # --- THE FIX: DISCRETE DOMAIN RANDOMIZATION ---
        
        # 1. Pick a specific training X position! (Center, halfway, or edges)
        possible_x_positions = [0.0, -15.0, 15.0, -30.0, 30.0]
        start_x = self.np_random.choice(possible_x_positions)
        
        self.lander.position = (float(start_x), start_y)
        
        # 2. Force a specific training angle! (0, 45, 90, 135, or 180 degrees)
        possible_angles = [0, 45, -45, 90, -90, 135, -135, 180, -180]
        chosen_angle_deg = self.np_random.choice(possible_angles)
        
        # Convert degrees to radians because the physics engine requires it
        self.lander.angle = math.radians(chosen_angle_deg)
        
        # 3. Force a specific training speed! (0, 25, 50, 75, or 100 m/s downwards)
        # Note: We use negative numbers because falling is mathematically negative
        possible_speeds = [0.0, -25.0, -50.0, -75.0, -100.0]
        start_vy = self.np_random.choice(possible_speeds)
        self.lander.linearVelocity = (0, float(start_vy))
        
        # Ensure it is not spinning when it spawns
        self.lander.angularVelocity = 0.0
        
        self.prev_y = self.lander.worldCenter.y
        self.impact_speed = None  
        
        # ---> THE NEW ASSIST: Blank Memory for Frame 1 <---
        self.prev_actions = [0.0, 0.0, 0.0]

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
        
        # ---> THE NEW ASSIST: Save the moves for the next frame! <---
        self.prev_actions = [main_engine_power, center_side_power, nose_side_power]
        
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
    
    def _get_state(self):
        pos = self.lander.worldCenter
        vel = self.lander.linearVelocity
        
        world_width_half = VIEWPORT_W / SCALE / 2 
        norm_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        
        # 1. The Hover Cheat Sheet
        gravity_force = self.lander.mass * abs(GRAVITY_Y)
        hover_power = float(gravity_force / MAIN_ENGINE_POWER)
        
        # ---> THE NEW ASSIST: The Glide Slope Speedometer <---
        # Step A: Find out exactly how high the physical bottom of the rocket is
        true_altitude = get_true_altitude(self.lander, ROCKET_H_WIDTH, ROCKET_HEIGHT)
        
        # Step B: Calculate the exact speed the mathematical curve wants us to go
        ideal_vy = -(math.sqrt(max(0.0, true_altitude)) * 4.0) - 1.0
        
        # Step C: Calculate the difference! 
        # (If this is 0.0, the AI is flying perfectly. If it is positive, it is falling too fast!)
        speed_error = ideal_vy - vel.y
        
        return [
            pos.x / world_width_half,       # 1. Horizontal Position (-1 to 1)
            pos.y / self.max_altitude,      # 2. Vertical Position (0 to 1)
            vel.x / 150.0,                  # 3. Horizontal Velocity 
            vel.y / 150.0,                  # 4. Vertical Velocity
            norm_angle / math.pi,           # 5. Angle (-1 to 1) 
            self.lander.angularVelocity/6.0,# 6. Angular Velocity
            self.fuel_left / INITIAL_FUEL,  # 7. Fuel
            hover_power,                    # 8. Exact T:W Ratio
            speed_error / 10.0,             # 9. The Speedometer! (Scaled by 10)
            
            # ---> THE NEW ASSIST: Action Memory <---
            self.prev_actions[0],           # 10. Previous Main Engine
            self.prev_actions[1],           # 11. Previous Center Thruster
            self.prev_actions[2]            # 12. Previous Nose Thruster
        ]
    
    """REWARD FUNCTION TO CALCULATE THE REWARD FOR THE CURRENT STATE. THIS FUNCTION TAKES INTO ACCOUNT THE DISTANCE TO THE LANDING PAD, THE VELOCITY, THE ANGLE, AND THE FUEL LEFT TO COMPUTE A COMPREHENSIVE REWARD SIGNAL THAT ENCOURAGES SAFE AND EFFICIENT LANDINGS."""   
    def _compute_reward(self, state):
        pos = self.lander.worldCenter
        vel = self.lander.linearVelocity
        true_altitude = get_true_altitude(self.lander, ROCKET_H_WIDTH, ROCKET_HEIGHT)
    
        reward = 0.0
        terminated = False
        truncated = False
        
        norm_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        tilt_rad = abs(norm_angle)
        x_range = VIEWPORT_W / SCALE / 2

        # ==========================================
        # STEP 1: THE KILL SWITCHES (The Four Walls)
        # ==========================================
        
        # 1. Did it fly off the left or right side of the screen?
        is_off_sides = abs(pos.x) >= x_range
        
        # 2. Did it fly too high into space?
        is_above_ceiling = pos.y > self.max_altitude
        
        # 3. Did it glitch through the dirt floor? 
        # (If true_altitude is less than -2.0, it is deep underground)
        is_below_floor = true_altitude < -2.0 

        # 4. Combine them all: If ANY of these are true, it is out of bounds!
        is_out_of_bounds = is_off_sides or is_above_ceiling or is_below_floor

        if is_out_of_bounds:
            reward -= 100.0
            terminated = True 
            self.landing_status = "OUT_OF_BOUNDS"
            return reward, terminated, truncated
        else:
            self.landing_status = "IN_PROGRESS"
            
        # ------------------------------------------
        # STEP 2: THE CONTINUOUS REWARD SHAPING
        # ------------------------------------------    
        
        # 1. The Time Tax (The ticking clock)
        reward -= 0.3
        
        # 2. The Main Engine Tax
        reward -= self.main_engine_power * 0.2
        
        # 3. The Side Thruster Tax
        reward -= abs(self.center_side_power) * 0.02
        reward -= abs(self.nose_side_power) * 0.01
        
        # ------------------------------------------
        #  2. Distance Points (Using utils.py)
        # ------------------------------------------
        old_distance = getattr(self, 'prev_distance', None)
        dist_points, new_distance = calculate_distance_reward(pos.x, true_altitude, old_distance)
        reward += dist_points
        self.prev_distance = new_distance
        
        # ------------------------------------------
        # 3. Posture Points (Using utils.py)
        # ------------------------------------------
        old_tilt = getattr(self, 'prev_tilt', None)
        posture_points, new_tilt = calculate_posture_reward(tilt_rad, old_tilt)
        reward += posture_points
        self.prev_tilt = new_tilt

        # If the rocket is almost perfectly straight, give it a small safe bonus!
        if new_tilt < 0.05:
            reward += 0.05
            
        # Squaring the tilt makes small wobbles cheap, but big leans incredibly expensive
        # Squaring the tilt makes small wobbles cheap, but big leans incredibly expensive
        raw_tilt_penalty = (new_tilt ** 2) * 1.0
        safe_tilt_penalty = min(1.0, raw_tilt_penalty)
        reward -= safe_tilt_penalty
        
        # ---> THE NEW RULE: Final Approach Corridor <---
        # If we are 5 meters or lower to the ground...
        if true_altitude <= 5.0:
            
            # ...and we are practically dead-straight (under 1 degree / 0.017 rads)...
            if new_tilt < 0.017:
                # ...feed it a steady stream of positive points!
                reward += 0.1
            
        # ------------------------------------------
        # 4. Horizontal Centering Points (Using utils.py)
        # ------------------------------------------
        old_x_dist = getattr(self, 'prev_x_dist', None)
        x_points, new_x_dist = calculate_x_reward(pos.x, old_x_dist)
        reward += x_points
        self.prev_x_dist = new_x_dist
        
        # ---> THE NEW RULE: The Dead-Center Bonus <---
        # If the rocket's center of mass is within 0.1 meters of the exact middle, give a small cookie!
        if new_x_dist < 0.1:
            reward += 0.05
        
        raw_x_penalty = (new_x_dist ** 2) * 0.02
        
        # Step 2: Use min() to put a hard ceiling on the punishment! 
        safe_x_penalty = min(1.0, raw_x_penalty)
        reward -= safe_x_penalty
        
        if true_altitude <= 5.0:
            
            # ...and we are practically dead-straight (under 1 degree / 0.017 rads)...
            if new_x_dist < 0.1:
                reward += 0.1
                    
        # ------------------------------------------
        # 5. The Glide Slope (Curve Match + Bonus)
        # ------------------------------------------
        
        # Step 1: Calculate the ideal falling speed for any altitude
        ideal_vy = -(math.sqrt(max(0.0, true_altitude)) * 4.0) - 1.0
        
        # Step 2: Calculate the exact difference between current speed and ideal speed
        # abs() makes sure the difference is a clean positive number
        speed_difference = abs(ideal_vy - vel.y)
        
        # ---> YOUR NEW RULE: The Perfect Speed Bonus <---
        # Step 3: If the rocket is within 1.0 m/s of the perfect curve, give a cookie!
        if speed_difference < 1.0:
            reward += 0.05
            
        if vel.y < ideal_vy:
            speed_error = ideal_vy - vel.y 
            
            # Step 5: Apply the capped exponential speed penalty for meteor-drops
            raw_speed_penalty = (speed_error ** 2) * 0.02
            safe_speed_penalty = min(1.0, raw_speed_penalty)
            reward -= safe_speed_penalty
            
        # ---> THE NEW RULE: The Anti-Climb Penalty <---
        # In physics, falling is negative speed. If vel.y is positive, the rocket is going UP!
        if vel.y > 0.0:
            raw_climb_penalty = vel.y * 0.5
            safe_climb_penalty = min(1.0, raw_climb_penalty)
            reward -= safe_climb_penalty
            
        # ---> THE NEW RULE: Final Approach Speed Corridor <---
        # If we are 5 meters or lower to the ground...
        if true_altitude <= 5.0:
            
            # ...and we are falling safely (slower than -5.0 m/s), 
            # BUT we are actively moving down (faster than -0.5 m/s)...
            if vel.y > -5.0 and vel.y < -0.5:
                # ...feed it a steady stream of positive points!
                reward += 0.1
        # ==========================================
        # STEP 3: THE EXPONENTIAL TOUCHDOWN BONUS
        # ==========================================
        
        # Step 1: Check if the episode is over (collision or touching the pad)
        if getattr(self, 'game_over', False) or true_altitude <= 0.0:
            terminated = True
            
            # ---> THE FIX: The Ghost Brake Defense <---
            # Step 2: Grab the real impact speed from the collision detector if it exists,
            # otherwise fall back to the current velocity.
            raw_impact = getattr(self, 'impact_speed', None)
            if raw_impact is None:
                raw_impact = vel.y
                
            impact_v = abs(raw_impact)
            
            # Step 3: The Strict Survival Checks
            is_straight = new_tilt < 0.087      
            is_on_pad = new_x_dist < (PAD_WIDTH_METERS / 2.0)
            is_safe_speed = impact_v < 5.0
            
            # Step 4: Calculate the Winning Bonuses
            if is_straight and is_on_pad and is_safe_speed:
                self.landing_status = "LANDED" 
                
                speed_multiplier = math.exp(-impact_v / 2.0)
                touchdown_bonus = 100.0 * speed_multiplier
                reward += touchdown_bonus
                
                accuracy_multiplier = 1.0 - (new_x_dist / (PAD_WIDTH_METERS / 2.0))
                bullseye_bonus = 30.0 * accuracy_multiplier
                reward += bullseye_bonus
                                                
                fuel_bonus = getattr(self, 'fuel_left', 0.0) * 0.2 * speed_multiplier
                reward += fuel_bonus
                
            # Step 5: The Crash Penalty
            else:
                self.landing_status = "CRASH"
                reward -= 50.0

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