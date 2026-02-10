import pygame
import math
import numpy as np
from .settings import * #importing all the global constants

class RocketVisualizer:
    def __init__(self, env):
        self.env = env
        self.screen = None
        self.clock = None   
        self.font = None
        self.current_flame_power = 0.0
    
    def init_window(self):
        """Initializes Pygame window if its not already made"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
            
    def render(self,mode="human"):
        """Renders the current state of the world to the screen"""
        self.init_window()

        # 1. Background
        self.screen.fill(SKY_COLOR)

        # 2. Ground Line
        pygame.draw.line(
            self.screen, GROUND_COLOR, 
            (0, VIEWPORT_H - 10), (VIEWPORT_W, VIEWPORT_H - 10), 1
        )
        
        # 3. Landing Pad (Yellow Flags)
        pad_center = VIEWPORT_W / 2
        pad_width = 80 
        pygame.draw.rect(self.screen, PAD_COLOR, (pad_center - pad_width/2, VIEWPORT_H - 15, pad_width, 5))

        # 4. Draw Rocket Parts (Body + Legs)
        self._draw_physics_objects()

        # 5. Draw Exhaust & HUD
        self._draw_exhaust()
        self._draw_hud()

        # 6. Display Flip
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_physics_objects(self):
        """Draws the rocket body and legs, plus cosmetic details (Nose Cone, Fins)."""
        for obj in self.env.drawlist:
            # 1. Draw the Base Physics Shape (The Rectangle)
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                
                pixel_path = [
                    (SCALE * v[0] + VIEWPORT_W / 2, VIEWPORT_H - SCALE * v[1] - 10)
                    for v in path
                ]
                
                # Color logic
                if obj == self.env.lander:
                    color = (150, 150, 150)  # Light grey for rocket body
                else:
                    color = LEG_COLOR
                
                pygame.draw.polygon(self.screen, color, pixel_path)

            # 2. If this is the Rocket Body, draw the "Skin" (Decorations)
            if obj == self.env.lander:
                self._draw_rocket_details(obj)

    def _draw_rocket_details(self, lander):
        """Draws cosmetic Nose Cone, Fins, and Window using METRIC dimensions."""
        pos = lander.position
        angle = lander.angle
        
        # Helper to rotate local points to screen points
        def transform_point(local_x, local_y):
            # Rotate
            rot_x = local_x * math.cos(angle) - local_y * math.sin(angle)
            rot_y = local_x * math.sin(angle) + local_y * math.cos(angle)
            # Translate to World
            world_x = pos.x + rot_x
            world_y = pos.y + rot_y
            # Scale to Screen
            screen_x = (SCALE * world_x) + (VIEWPORT_W / 2)
            screen_y = VIEWPORT_H - (SCALE * world_y) - 10
            return (screen_x, screen_y)

        # --- USE METRIC CONSTANTS FROM SETTINGS.PY ---
        # This ensures the visuals match the zoom level (SCALE)
        half_w = ROCKET_H_WIDTH
        body_h = ROCKET_HEIGHT
        nose_h = NOSE_HEIGHT

        # A. NOSE CONE (Triangle on top)
        # Base is at body_h, Tip is at body_h + nose_h
        nose_points = [
            transform_point(-half_w, body_h), # Top Left of body
            transform_point(half_w, body_h),  # Top Right of body
            transform_point(0, body_h + nose_h) # Tip of nose
        ]
        pygame.draw.polygon(self.screen, (255, 50, 50), nose_points) # Light Red Nose

        # B. FINS (Triangles at bottom)
        # Dimensions relative to rocket size
        fin_h = body_h * 0.4   # Fins go up 40% of body
        fin_w = half_w * 0.8   # Fins stick out 80% of width
        fin_drop = body_h * 0.1 # Fins drop slightly below body

        # Left Fin
        left_fin = [
            transform_point(-half_w, 0),          # Body attach point (bottom)
            transform_point(-half_w, fin_h),      # Body attach point (top)
            transform_point(-half_w - fin_w, -fin_drop)  # Wing tip
        ]
        pygame.draw.polygon(self.screen, (255, 50, 50), left_fin) # Light Red

        # Right Fin
        right_fin = [
            transform_point(half_w, 0),
            transform_point(half_w, fin_h),
            transform_point(half_w + fin_w, -fin_drop)
        ]
        pygame.draw.polygon(self.screen, (255, 50, 50), right_fin)

        # C. COCKPIT WINDOW (Blue Circle)
        # Located near the top (75% up the body)
        window_center = transform_point(0, body_h * 0.75)
        # Radius is 40% of the half-width
        radius = int(SCALE * (half_w * 0.4))
        pygame.draw.circle(self.screen, (0, 200, 255), (int(window_center[0]), int(window_center[1])), radius)
        

    def _draw_exhaust(self):
        """Draws a multi-layered, flickering flame for realistic effect."""
        power = getattr(self.env, 'main_engine_power', 0.0)
        if power < 0.05: return

        pos = self.env.lander.position
        angle = self.env.lander.angle
        
        # 1. Nozzle Position (Base of the flame)
        nozzle_x = pos.x - math.sin(angle) * (18/SCALE)
        nozzle_y = pos.y - math.cos(angle) * (18/SCALE)

        # 2. Define Flame Layers (Color, Width, Length)
        # We draw from largest (outer) to smallest (inner)
        flame_layers = [
            # Color (R,G,B)       Width   Length Multiplier
            ((255, 100, 0),       0.4,    1.8),  # Outer Orange (Big & Wide)
            ((255, 200, 0),       0.25,   1.5),  # Middle Yellow
            ((255, 255, 255),     0.12,   1.2)   # Inner White (Hot Core)
        ]

        def to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * y) - 10
            return (int(sx), int(sy))

        # 3. Draw Each Layer
        for color, width_base, len_mult in flame_layers:
            # A. Add "Flicker" (Random noise to make it feel alive)
            # The harder the engine pushes, the more stable the flame becomes
            flicker_w = np.random.uniform(-0.05, 0.05)
            flicker_l = np.random.uniform(-0.1, 0.1)
            
            current_w = width_base + flicker_w
            current_l = (power * len_mult) + flicker_l

            # B. Calculate Vertices
            # Tip of the flame
            tip_x = nozzle_x - math.sin(angle) * current_l
            tip_y = nozzle_y - math.cos(angle) * current_l
            
            # Base corners (Left and Right of nozzle)
            base_left_x = nozzle_x - math.cos(angle) * current_w
            base_left_y = nozzle_y + math.sin(angle) * current_w
            
            base_right_x = nozzle_x + math.cos(angle) * current_w
            base_right_y = nozzle_y - math.sin(angle) * current_w

            # C. Draw Triangle
            points = [
                to_screen(base_left_x, base_left_y),
                to_screen(base_right_x, base_right_y),
                to_screen(tip_x, tip_y)
            ]
            pygame.draw.polygon(self.screen, color, points)

    def _draw_hud(self):
        """Draws the Fuel Bar (Right) and Telemetry Text (Left)."""
        
        # --- 1. LEFT SIDE: Telemetry (Speed & Altitude) ---
        vel = self.env.lander.linearVelocity
        pos = self.env.lander.position
        
        telemetry_texts = [
            f"X Speed: {abs(vel.x):.2f} m/s",
            f"Y Speed: {abs(vel.y):.2f} m/s",
            f"Altitude: {pos.y:.2f} m"
        ]
        
        for i, text in enumerate(telemetry_texts):
            label = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(label, (10, 10 + (i * 20)))

        # --- 2. RIGHT SIDE: Fuel Indicator ---
        fuel_pct = max(0, self.env.fuel_left / INITIAL_FUEL)
        
        # Layout Constants
        bar_width = 200
        bar_height = 20
        right_margin = 20
        top_margin = 10
        
        # Calculate Positions (Aligning to the right side of the screen)
        # We use VIEWPORT_W to find the right edge
        bar_x = VIEWPORT_W - bar_width - right_margin
        bar_y = top_margin + 25  # Lowered to make room for "FUEL" text
        
        # A. Draw "FUEL" Heading
        # We align it with the start of the bar
        heading_label = self.font.render("FUEL", True, (255, 255, 255))
        self.screen.blit(heading_label, (bar_x, top_margin))
        
        # B. Draw Fuel Bar
        # Border (White)
        pygame.draw.rect(self.screen, FUEL_BAR_BORDER, (bar_x, bar_y, bar_width + 4, bar_height + 4), 2)
        # Fill (Green)
        pygame.draw.rect(self.screen, FUEL_BAR_COLOR_FULL, (bar_x + 2, bar_y + 2, int(bar_width * fuel_pct), bar_height))
        
        # C. Draw Percentage Number (Left of the bar)
        pct_text = f"{fuel_pct * 100:.1f}%"
        pct_label = self.font.render(pct_text, True, (255, 255, 255))
        
        # Position: 10 pixels to the left of the bar
        pct_x = bar_x - pct_label.get_width() - 10 
        self.screen.blit(pct_label, (pct_x, bar_y))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None