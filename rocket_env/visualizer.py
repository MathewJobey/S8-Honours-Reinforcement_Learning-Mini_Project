import pygame
import math
import numpy as np
import random 
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
    
    def init_window(self):
        """Initializes Pygame window and generates static stars."""
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
        
        # Pad Rect
        pad_rect = (pad_screen_x - pad_w_px//2, pad_screen_y, pad_w_px, pad_h_px)
        pygame.draw.rect(self.screen, PAD_COLOR_1, pad_rect)
        pygame.draw.rect(self.screen, (0,0,0), pad_rect, 2) # Outline

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

        # Fins
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

    def _draw_exhaust(self):
        power = getattr(self.env, 'main_engine_power', 0.0)
        self.current_flame_power += (power - self.current_flame_power) * 0.2
        if self.current_flame_power < 0.05: return

        pos = self.env.lander.position
        angle = self.env.lander.angle
        
        # Nozzle is at rocket position (0,0 local)
        nozzle_x = pos.x 
        nozzle_y = pos.y 

        # Flicker
        t = pygame.time.get_ticks() * 0.05
        flicker = (math.sin(t) * 0.1) + (math.cos(t * 3) * 0.05)
        
        flame_len = (self.current_flame_power * 3.0) + flicker
        flame_w = 0.5 * self.current_flame_power

        def to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * (y - self.camera_y)) - 20
            return (int(sx), int(sy))

        # Tip of flame
        tip_x = nozzle_x - math.sin(angle) * flame_len
        tip_y = nozzle_y - math.cos(angle) * flame_len
        
        # Base of flame
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

    def _draw_hud(self):
        vel = self.env.lander.linearVelocity
        pos = self.env.lander.position
        
        # Black Text for visibility against Sky
        texts = [
            f"Altitude: {pos.y:.1f} m",
            f"X Vel: {vel.x:.1f} m/s",
            f"Y Vel: {vel.y:.1f} m/s",
            f"Angle: {math.degrees(self.env.lander.angle):.1f}"
        ]
        
        for i, t in enumerate(texts):
            label = self.font.render(t, True, (0, 0, 0)) 
            self.screen.blit(label, (10, 10 + (i * 20)))

        # Fuel Bar
        fuel_pct = max(0, self.env.fuel_left / INITIAL_FUEL)
        bar_x = VIEWPORT_W - 220
        bar_y = 30
        
        pygame.draw.rect(self.screen, (255,255,255), (bar_x, bar_y, 200, 20), 2)
        pygame.draw.rect(self.screen, (0,255,0), (bar_x+2, bar_y+2, 196 * fuel_pct, 16))
        
        lbl = self.font.render("FUEL", True, (0,0,0))
        self.screen.blit(lbl, (bar_x, bar_y - 20))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None