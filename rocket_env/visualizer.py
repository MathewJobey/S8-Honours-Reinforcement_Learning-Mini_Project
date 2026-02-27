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
        self.camera_y = 0.0  # Tracks camera altitude
        # --- NEW: BUBBLE MEMORY BANK ---
        self.side_particles = []
    
    def init_window(self):
        if self.screen is None:
            # --- FIX 1: CENTER WINDOW ---
            # This forces the window to open in the middle of your monitor,
            # keeping it clear of the taskbar.
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

            # --- GENERATE STARS ONCE ---
            self.stars = []
            for _ in range(150):
                x = random.randint(0, VIEWPORT_W)
                y = random.randint(0, VIEWPORT_H) 
                radius = random.randint(1, 2)
                self.stars.append((x, y, radius))
            
    def render(self, mode="human"):
        """Renders the world with stars, ground, and hazard pad."""
        self.init_window()

        # --- 1. CAMERA LOGIC ---
        # Keep rocket at ~40% of screen height
        rocket_y = self.env.lander.position.y
        
        # Calculate Target Camera Position
        # We shift the camera up so the rocket is centered
        target_cam_y = rocket_y - (VIEWPORT_H / SCALE * 0.4)
        
        # Clamp: Never let the camera go below 0 (underground).
        self.camera_y = max(0.0, target_cam_y)
        
        # Helper to convert World Y -> Screen Y
        def world_to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            # CRITICAL: Subtract self.camera_y to move the world
            sy = VIEWPORT_H - (SCALE * (y - self.camera_y)) - 20
            return int(sx), int(sy)
        
        # --- 2. DRAWING BACKGROUND ---
        self.screen.fill(SKY_COLOR)

        # Stars (Static parallax)
        for x, y, radius in self.stars:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius)

        # Ground (Moves with camera)
        ground_screen_y = world_to_screen(0, 0)[1]
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, ground_screen_y, VIEWPORT_W, VIEWPORT_H))
        
        # --- 3. LANDING PAD ---
        pad_w_px = int(PAD_WIDTH_METERS * SCALE)
        pad_h_px = int(PAD_HEIGHT_METERS * SCALE)
        pad_screen_x, pad_screen_y = world_to_screen(0, PAD_HEIGHT_METERS)
        
        # Pad Rect with yellow and black stripes
        pad_rect = (pad_screen_x - pad_w_px//2, pad_screen_y, pad_w_px, pad_h_px)
        stripe_width = 10
        for x in range(pad_screen_x - pad_w_px//2, pad_screen_x + pad_w_px//2, stripe_width * 2):
            pygame.draw.rect(self.screen, (255, 255, 0), (x, pad_screen_y, stripe_width, pad_h_px))
            pygame.draw.rect(self.screen, (0, 0, 0), (x + stripe_width, pad_screen_y, stripe_width, pad_h_px))
        pygame.draw.rect(self.screen, (0, 0, 0), pad_rect, 2)  # Outline

        # Status Lights
        status = getattr(self.env, 'landing_status', "IN_PROGRESS")
        light_color = (255, 165, 0) # Orange
        if status == "CRASH": light_color = (255, 0, 0)
        elif status == "LANDED": light_color = (0, 255, 100)
        
        pygame.draw.circle(self.screen, light_color, (pad_screen_x - pad_w_px//2, pad_screen_y), 6)
        pygame.draw.circle(self.screen, light_color, (pad_screen_x + pad_w_px//2, pad_screen_y), 6)

        # --- 4. DRAW ROCKET & HUD ---
        self._draw_rocket_details(self.env.lander)
        self._draw_exhaust()
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

    def _draw_rocket_details(self, lander):
        """Draws the rocket with clean, sharp outlines."""
        pos = lander.position
        angle = lander.angle
        
        # Helper: Rotate and Scale to screen (Uses Camera!)
        def transform_point(local_x, local_y):
            rot_x = local_x * math.cos(angle) - local_y * math.sin(angle)
            rot_y = local_x * math.sin(angle) + local_y * math.cos(angle)
            
            screen_x = (SCALE * (pos.x + rot_x)) + (VIEWPORT_W / 2)
            # Apply Camera Offset
            screen_y = VIEWPORT_H - (SCALE * (pos.y + rot_y - self.camera_y)) - 20
            return (screen_x, screen_y)

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

        # ==========================================
        # --- REAR FINS (BOTTOM FLAPS) ---
        # ==========================================
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

        # ==========================================
        # --- NEW: FORWARD FLAPS (TOP FINS) ---
        # ==========================================
        # Make them slightly smaller than the rear flaps
        front_fin_h = body_h * 0.2  
        front_fin_w = half_w * 1.2
        # Position them high up on the body, just before the nose
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

    def _draw_exhaust(self):
        # ==========================================
        # 1. MAIN ENGINE EXHAUST (Orange Flames)
        # ==========================================
        power = getattr(self.env, 'main_engine_power', 0.0)
        self.current_flame_power += (power - self.current_flame_power) * 0.2
        
        pos = self.env.lander.position
        angle = self.env.lander.angle
        
        def to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * (y - self.camera_y)) - 20
            return (int(sx), int(sy))

        # Only draw main flame if power is high enough
        if self.current_flame_power >= 0.05:
            nozzle_x = pos.x 
            nozzle_y = pos.y 

            t = pygame.time.get_ticks() * 0.05
            flicker = (math.sin(t) * 0.1) + (math.cos(t * 3) * 0.05)
            
            flame_len = (self.current_flame_power * 3.0) + flicker
            flame_w = 0.5 * self.current_flame_power

            tip_x = nozzle_x - math.sin(angle) * flame_len
            tip_y = nozzle_y - math.cos(angle) * flame_len
            
            base_l_x = nozzle_x - math.cos(angle) * flame_w
            base_l_y = nozzle_y + math.sin(angle) * flame_w
            base_r_x = nozzle_x + math.cos(angle) * flame_w
            base_r_y = nozzle_y - math.sin(angle) * flame_w

            points = [
                to_screen(base_l_x, base_l_y),
                to_screen(base_r_x, base_r_y),
                to_screen(tip_x, tip_y)
            ]
            pygame.draw.polygon(self.screen, (255, 100, 0), points)

        # ==========================================
        # 2. SIDE THRUSTERS (Blue Bubbles)
        # ==========================================
        side_power = getattr(self.env, 'side_engine_power', 0.0)
        
        # 1. Spawn new bubbles if the AI is pushing the stick
        if abs(side_power) > 0.05:
            nozzle_local_y = ROCKET_HEIGHT / 2.0
            direction = 1 if side_power > 0 else -1
            
            # Where on the rocket the bubble is born
            start_x = pos.x - math.sin(angle) * nozzle_local_y + (math.cos(angle) * ROCKET_H_WIDTH * direction)
            start_y = pos.y + math.cos(angle) * nozzle_local_y + (math.sin(angle) * ROCKET_H_WIDTH * direction)
            
            # Spawn 2 bubbles per frame for a nice thick trail
            for _ in range(2):
                # Give it a random burst speed pushing away from the rocket
                speed = random.uniform(2.0, 6.0)
                vel_x = math.cos(angle) * speed * direction
                vel_y = math.sin(angle) * speed * direction
                
                # Add the new bubble to our memory bank. 
                # "life" starts at 1.0 (100%) and will drop to 0.
                self.side_particles.append({
                    "x": start_x,
                    "y": start_y,
                    "vx": vel_x,
                    "vy": vel_y,
                    "life": 1.0
                })

        # 2. Update and draw all existing bubbles
        living_particles = []
        for p in self.side_particles:
            # Move the bubble through space
            p["x"] += p["vx"] * (1.0 / FPS)
            p["y"] += p["vy"] * (1.0 / FPS)
            
            # Age the bubble (loses 5% of its life every frame)
            p["life"] -= 0.05 
            
            # If it is still alive, draw it!
            if p["life"] > 0:
                living_particles.append(p) # Save it for the next frame
                
                # The radius shrinks as the bubble gets older
                radius = int(4 * p["life"])
                if radius > 0:
                    screen_pos = to_screen(p["x"], p["y"])
                    # Draw a solid Cyan (0, 255, 255) circle
                    pygame.draw.circle(self.screen, (0, 255, 255), screen_pos, radius)
                    
        # Update our memory bank to only keep the ones that haven't popped yet
        self.side_particles = living_particles

    def _draw_hud(self):
        vel = self.env.lander.linearVelocity
        pos = self.env.lander.position
        
        # Safely get the values from the environment
        drag_val = getattr(self.env, 'current_drag', 0.0)
        torque_val = getattr(self.env, 'current_torque', 0.0)
        
        # White Text for visibility against Sky
        texts = [
            f"Altitude: {abs(pos.y):.1f} m",
            f"X Vel: {abs(vel.x):.1f} m/s",
            f"Y Vel: {abs(vel.y):.1f} m/s",
            f"Angle: {math.degrees(self.env.lander.angle):.1f}",
            f"Aero Drag: {drag_val:.1f} N",               
            f"Flap Torque: {abs(torque_val):.1f} Nm"      
        ]
        
        for i, t in enumerate(texts):
            label = self.font.render(t, True, (255, 255, 255)) 
            self.screen.blit(label, (10, 10 + (i * 20)))

        # --- MOVED FUEL BAR LOGIC (TOP RIGHT) ---
        # 1. Get the current fuel and calculate the percentage
        fuel_left = getattr(self.env, 'fuel_left', 0.0)
        fuel_ratio = max(0.0, min(1.0, fuel_left / INITIAL_FUEL))
        
        # 2. Set the size of the bar
        bar_w = 150
        bar_h = 15
        
        # Set the position to the Top Right
        # Subtract the bar width and a 20-pixel margin from the screen width
        bar_x = VIEWPORT_W - bar_w - 20 
        bar_y = 35 # Moved down to 35 to make room for the text above it

        # 3. Draw the "FUEL" Text Label
        fuel_label = self.font.render("FUEL", True, (255, 255, 255))
        # Place the text aligned with the left edge of the bar, but up near the top (y=10)
        self.screen.blit(fuel_label, (bar_x, 10))

        # 4. Draw the background (Empty / Red)
        pygame.draw.rect(self.screen, (150, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        
        # 5. Draw the foreground (Full / Green)
        fill_w = int(bar_w * fuel_ratio)
        if fill_w > 0:
            pygame.draw.rect(self.screen, FUEL_BAR_COLOR_FULL, (bar_x, bar_y, fill_w, bar_h))
            
        # 6. Draw the white outline border
        pygame.draw.rect(self.screen, FUEL_BAR_BORDER, (bar_x, bar_y, bar_w, bar_h), 2)
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None