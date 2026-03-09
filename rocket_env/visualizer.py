import pygame
import math
import numpy as np
import random 
import os
from .settings import *

class RocketVisualizer:
    def __init__(self, env):
        self.env = env
        self.screen = None
        self.clock = None   
        self.font = None
        self.current_flame_power = 0.0
        self.stars = []
        self.camera_y = 0.0
        self.side_particles = []
    
    def init_window(self):
        if self.screen is None:
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            
            # --- Initialize Pygame and Create Window ---
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

            # --- GENERATE STARS 150 CODE ---
            self.stars = []
            for _ in range(150):
                x = random.randint(0, VIEWPORT_W)
                y = random.randint(0, VIEWPORT_H) 
                radius = random.randint(1, 2)
                self.stars.append((x, y, radius))
    
    """RENDER FUNCTION: This is where all the magic happens. We use a "Painter's Algorithm" approach to draw the scene in layers, starting with the background and stars, then the ground, landing pad, and finally the rocket and HUD on top. The camera logic ensures that the rocket stays centered vertically, creating a smooth scrolling effect as it ascends or descends. We also added a dynamic fuel bar in the HUD that visually represents how much fuel is left, making it easier for both humans and AI to gauge their remaining resources at a glance."""       
    def render(self, mode="human"):
        """Renders the world with stars, ground, and hazard pad PAINTERS ALGORITHM."""
        self.init_window()
        
        # --- 1. CAMERA LOGIC ---
        # Keep rocket at ~40% of screen height
        rocket_y = self.env.lander.position.y
        target_cam_y = rocket_y - (VIEWPORT_H / SCALE * 0.4)                                                             
        self.camera_y = max(0.0, target_cam_y) # Clamp: Never let the camera go below 0 (underground).
        
        # Helper to convert World Y -> Screen Y
        def world_to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * (y - self.camera_y)) - 20
            return int(sx), int(sy)
        
        # --- 2. DRAWING BACKGROUND ---
        self.screen.fill(SKY_COLOR)
        
        # Stars (Static Wallpaper)
        for x, y, radius in self.stars:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius)

        # Ground (Moves with camera)
        ground_screen_y = world_to_screen(0, 0)[1]
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, ground_screen_y, VIEWPORT_W, VIEWPORT_H))
        
        # --- 3. LANDING PAD ---
        pad_w_px = int(PAD_WIDTH_METERS * SCALE)
        pad_h_px = int(PAD_HEIGHT_METERS * SCALE)
        pad_screen_x, pad_screen_y = world_to_screen(0, PAD_HEIGHT_METERS)
        pad_rect = (pad_screen_x - pad_w_px//2, pad_screen_y, pad_w_px, pad_h_px)
        self.screen.set_clip(pad_rect)

        stripe_width = 10
        for x in range(pad_screen_x - pad_w_px//2, pad_screen_x + pad_w_px//2, stripe_width * 2):
            pygame.draw.rect(self.screen, (255, 255, 0), (x, pad_screen_y, stripe_width, pad_h_px))
            pygame.draw.rect(self.screen, (0, 0, 0), (x + stripe_width, pad_screen_y, stripe_width, pad_h_px))
            
        # THE FIX: Turn off the clipping mask so we can draw the rest of the game!
        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, (0, 0, 0), pad_rect, 2)  # Draw the black Outline

        # Status Lights
        status = getattr(self.env, 'landing_status', "IN_PROGRESS")
        light_color = (255, 165, 0) # Orange
        if status == "CRASH": light_color = (255, 0, 0)
        elif status == "LANDED": light_color = (0, 255, 100)
        elif status == "OUT_OF_BOUNDS": light_color = (255, 255, 0)
        
        pygame.draw.circle(self.screen, light_color, (pad_screen_x - pad_w_px//2, pad_screen_y), 6)
        pygame.draw.circle(self.screen, light_color, (pad_screen_x + pad_w_px//2, pad_screen_y), 6)

        # --- 4. DRAW ROCKET & HUD ---
        # Draw exhaust FIRST so it sits behind the rocket body
        self._draw_exhaust()
        self._draw_rocket_details(self.env.lander)
        self._draw_hud()

        # Display
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    """DRAWING FUNCTIONS BELOW: We use local coordinate math to ensure the rocket is always perfectly centered and oriented, regardless of its position or angle. This allows us to add detailed features like fins and a nose cone without worrying about them looking distorted as the rocket rotates."""
    def _draw_rocket_details(self, lander):
        """Draws the rocket with clean, sharp outlines."""
        pos = lander.position
        angle = lander.angle
        
        # Helper: Rotate and Scale to screen (Uses Camera!)
        def transform_point(local_x, local_y):
            rot_x = local_x * math.cos(angle) - local_y * math.sin(angle)
            rot_y = local_x * math.sin(angle) + local_y * math.cos(angle)
            
            screen_x = (SCALE * (pos.x + rot_x)) + (VIEWPORT_W / 2)
            screen_y = VIEWPORT_H - (SCALE * (pos.y + rot_y - self.camera_y)) - 20
            return (int(screen_x), int(screen_y))

        # Colors & Dimensions
        WHITE = (245, 245, 245)
        BLACK = (25, 25, 25)
        RED = (200, 50, 50)
        half_w = ROCKET_H_WIDTH
        body_h = ROCKET_HEIGHT
        nose_h = NOSE_HEIGHT

        # Body
        body_points = [
            transform_point(-half_w, 0),
            transform_point( half_w, 0),
            transform_point( half_w, body_h),
            transform_point(-half_w, body_h),
        ]
        pygame.draw.polygon(self.screen, WHITE, body_points)
        pygame.draw.polygon(self.screen, BLACK, body_points, 2)

        # Nose Cone
        nose_points = [
            transform_point(-half_w, body_h),
            transform_point( half_w, body_h),
            transform_point(0, body_h + nose_h)
        ]
        pygame.draw.polygon(self.screen, RED, nose_points)
        pygame.draw.polygon(self.screen, BLACK, nose_points, 2)

        # REAR FINS 
        fin_h = body_h * 0.3
        fin_w = half_w * 1.5
        fin_bottom = body_h * 0.1
        
        # Left Fin
        l_fin = [
            transform_point(-half_w, fin_bottom),
            transform_point(-half_w, fin_bottom + fin_h),
            transform_point(-half_w - fin_w, fin_bottom)
        ]
        pygame.draw.polygon(self.screen, (100,100,100), l_fin)
        pygame.draw.polygon(self.screen, BLACK, l_fin, 2)

        # Right Fin
        r_fin = [
            transform_point(half_w, fin_bottom),
            transform_point(half_w, fin_bottom + fin_h),
            transform_point(half_w + fin_w, fin_bottom)
        ]
        pygame.draw.polygon(self.screen, (100,100,100), r_fin)
        pygame.draw.polygon(self.screen, BLACK, r_fin, 2)

        # FORWARD FLAPS
        front_fin_h = body_h * 0.2  
        front_fin_w = half_w * 1.2
        front_fin_bottom = body_h * 0.75  
        
        # Left Forward Flap
        l_front_fin = [
            transform_point(-half_w, front_fin_bottom),
            transform_point(-half_w, front_fin_bottom + front_fin_h),
            transform_point(-half_w - front_fin_w, front_fin_bottom)
        ]
        pygame.draw.polygon(self.screen, (100, 100, 100), l_front_fin)
        pygame.draw.polygon(self.screen, BLACK, l_front_fin, 2)

        # Right Forward Flap
        r_front_fin = [
            transform_point(half_w, front_fin_bottom),
            transform_point(half_w, front_fin_bottom + front_fin_h),
            transform_point(half_w + front_fin_w, front_fin_bottom)
        ]
        pygame.draw.polygon(self.screen, (100, 100, 100), r_front_fin)
        pygame.draw.polygon(self.screen, BLACK, r_front_fin, 2)

    """EXHAUST & PARTICLE EFFECTS: We create a dynamic exhaust flame that changes length based on the main engine power, with a flickering effect for realism. The side thrusters emit blue and purple bubbles that represent the AI's control inputs, giving us visual feedback on how the AI is maneuvering the rocket. These effects not only enhance the visual appeal but also provide valuable information about the rocket's current state and the AI's actions."""
    def _draw_exhaust(self):
        power = getattr(self.env, 'main_engine_power', 0.0)
        self.current_flame_power += (power - self.current_flame_power) * 0.2
        
        pos = self.env.lander.position
        angle = self.env.lander.angle
        
        # Helper: Local coordinate math for perfect centering
        def transform_point(local_x, local_y):
            rot_x = local_x * math.cos(angle) - local_y * math.sin(angle)
            rot_y = local_x * math.sin(angle) + local_y * math.cos(angle)
            screen_x = (SCALE * (pos.x + rot_x)) + (VIEWPORT_W / 2)
            screen_y = VIEWPORT_H - (SCALE * (pos.y + rot_y - self.camera_y)) - 20
            return (int(screen_x), int(screen_y))

        def to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * (y - self.camera_y)) - 20
            return (int(sx), int(sy))

        # ==========================================
        # 1. MAIN ENGINE EXHAUST (Orange Flames)
        # ==========================================
        if self.current_flame_power >= 0.05:
            t = pygame.time.get_ticks() * 0.05
            # Make the flicker slightly larger to match a bigger flame
            flicker = (math.sin(t) * 0.5) + (math.cos(t * 3) * 0.2)
            
            # THE FIX 1: Make the flame length responsive to the rocket's height!
            # At 100% power, the flame is 1.5x longer than the entire rocket.
            base_len = self.current_flame_power * (ROCKET_HEIGHT * 1.5)
            
            # THE FIX 2: Safety net! The max() guarantees the flame never goes inverted
            flame_len = max(0.1, base_len + flicker)
            
            # THE FIX 3: Make the flame width match the rocket's width
            flame_w = ROCKET_H_WIDTH * self.current_flame_power

            # Using Local Coordinates perfectly centers the flame base
            points = [
                transform_point(-flame_w, 0),  
                transform_point(flame_w, 0),   
                transform_point(0, -flame_len) 
            ]
            pygame.draw.polygon(self.screen, (255, 100, 0), points)

        # ==========================================
        # 2. SIDE THRUSTERS (Blue & Purple Bubbles)
        # ==========================================
        center_power = getattr(self.env, 'center_side_power', 0.0)
        nose_power = getattr(self.env, 'nose_side_power', 0.0)

        # Helper to spawn bubbles. 
        def spawn_bubbles(power_val, local_y, color):
            # LOWERED THRESHOLD: Show bubbles even if AI uses just 1% power!
            if abs(power_val) > 0.01:
                direction = 1 if power_val > 0 else -1
                start_x = pos.x - math.sin(angle) * local_y + (math.cos(angle) * ROCKET_H_WIDTH * direction)
                start_y = pos.y + math.cos(angle) * local_y + (math.sin(angle) * ROCKET_H_WIDTH * direction)
                
                # THE FIX 1: Scale the number of bubbles (Max 5 at full power)
                # math.ceil ensures even at 1% power, it spawns at least 1 bubble
                num_bubbles = math.ceil(5 * abs(power_val))
                
                for _ in range(num_bubbles): 
                    # THE FIX 2: Scale the speed. 
                    # We multiply the random speed by the power, and add a 1.5x boost so it looks punchy
                    base_speed = random.uniform(2.0, 6.0)
                    speed = base_speed * abs(power_val) * 1.5 
                    
                    self.side_particles.append({
                        "x": start_x, "y": start_y,
                        "vx": math.cos(angle) * speed * direction,
                        "vy": math.sin(angle) * speed * direction,
                        "life": 1.0,
                        "color": color 
                    })

        # Spawn Center bubbles (Cyan) and Nose bubbles (Purple)
        spawn_bubbles(center_power, ROCKET_HEIGHT / 2.0, (0,255,236))
        nose_y_visual = ROCKET_HEIGHT + (NOSE_HEIGHT / 2.0)
        spawn_bubbles(nose_power, nose_y_visual, (255,0,206))

        # Update and draw all bubbles
        living_particles = []
        for p in self.side_particles:
            p["x"] += p["vx"] 
            p["y"] += p["vy"] 
            p["life"] -= 0.05 
            
            if p["life"] > 0:
                living_particles.append(p) 
                radius = int(4 * p["life"])
                if radius > 0:
                    # Draw the circle using its assigned color!
                    pygame.draw.circle(self.screen, p["color"], to_screen(p["x"], p["y"]), radius)
                    
        self.side_particles = living_particles

    """HUD & FUEL BAR: The HUD displays critical flight information such as altitude, velocity, angle, and aerodynamic drag in a clear and concise manner. We also added a dynamic fuel bar in the top right corner that visually represents how much fuel is left, making it easier for both humans and AI to gauge their remaining resources at a glance. The fuel bar changes color from green to red as fuel depletes, providing an immediate visual cue about the urgency of refueling or landing safely before running out of fuel."""
    def _draw_hud(self):
        vel = self.env.lander.linearVelocity
        pos = self.env.lander.position
        
        drag_val = getattr(self.env, 'current_drag', 0.0)
        nose_val = getattr(self.env, 'nose_side_power', 0.0)
        center_val = getattr(self.env, 'center_side_power', 0.0)
        wrapped_angle = (self.env.lander.angle + math.pi) % (2 * math.pi) - math.pi
        
        # White Text for visibility against Sky
        texts = [
            f"Altitude: {max(0.0, pos.y - 1.0):.1f} m",
            # THE FIX 2: Removed abs() so we can see negative speeds (Left/Down)
            f"X Vel: {vel.x:.1f} m/s",
            f"Y Vel: {vel.y:.1f} m/s",
            # Use the wrapped angle so it stays between -180 and +180 degrees
            f"Angle: {math.degrees(wrapped_angle):.1f}",
            f"Aero Drag: {drag_val:.1f} N",               
            f"Nose Pwr: {nose_val:.2f}",               
            f"Center Pwr: {center_val:.2f}"   
        ]
        
        for i, t in enumerate(texts):
            label = self.font.render(t, True, (255, 255, 255)) 
            self.screen.blit(label, (10, 10 + (i * 20)))

        # --- MOVED FUEL BAR LOGIC (TOP RIGHT) ---
        fuel_left = getattr(self.env, 'fuel_left', 0.0)
        fuel_ratio = max(0.0, min(1.0, fuel_left / INITIAL_FUEL))
        
        bar_w = 150
        bar_h = 15
        
        bar_x = VIEWPORT_W - bar_w - 20 
        bar_y = 35 

        fuel_label = self.font.render("FUEL", True, (255, 255, 255))
        self.screen.blit(fuel_label, (bar_x, 10))

        # Draw the background (Empty / Dark Red)
        pygame.draw.rect(self.screen, (100, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        
        # THE FIX 3: Dynamic Color Logic!
        if fuel_ratio > 0.5:
            current_bar_color = (0, 255, 0)       # Green if above 50%
        elif fuel_ratio > 0.2:
            current_bar_color = (255, 255, 0)     # Yellow if above 20%
        else:
            current_bar_color = (255, 0, 0)       # Bright Red if almost empty!

        fill_w = int(bar_w * fuel_ratio)
        if fill_w > 0:
            pygame.draw.rect(self.screen, current_bar_color, (bar_x, bar_y, fill_w, bar_h))
            
        # Draw the white outline border
        pygame.draw.rect(self.screen, FUEL_BAR_BORDER, (bar_x, bar_y, bar_w, bar_h), 2)
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None