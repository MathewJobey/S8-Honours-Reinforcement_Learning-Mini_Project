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
        """Draws the rocket body and legs based on Box2D coordinates."""
        for obj in self.env.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                
                # Convert Box2D coords to Screen coords
                pixel_path = [
                    (SCALE * v[0] + VIEWPORT_W / 2, VIEWPORT_H - SCALE * v[1] - 10)
                    for v in path
                ]
                
                color = LANDER_COLOR if obj == self.env.lander else LEG_COLOR
                pygame.draw.polygon(self.screen, color, pixel_path)

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