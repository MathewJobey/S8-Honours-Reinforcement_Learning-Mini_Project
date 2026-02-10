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
        """Draws visual feedback for engines."""
        if self.env.lander.awake:
            pos = self.env.lander.position
            angle = self.env.lander.angle
            
            # Calculate bottom of rocket
            bottom_x = pos.x - math.sin(angle) * (18/SCALE)
            bottom_y = pos.y - math.cos(angle) * (18/SCALE)
            
            screen_x = (SCALE * bottom_x) + (VIEWPORT_W / 2)
            screen_y = VIEWPORT_H - (SCALE * bottom_y) - 10
            
            # Simple exhaust flame
            pygame.draw.circle(self.screen, EXHAUST_COLOR, (int(screen_x), int(screen_y)), 5)

    def _draw_hud(self):
        """Draws the Fuel Bar and Telemetry Text."""
        # A. Fuel Bar
        fuel_pct = max(0, self.env.fuel_left / INITIAL_FUEL)
        pygame.draw.rect(self.screen, FUEL_BAR_BORDER, (10, 10, 204, 24), 2) 
        pygame.draw.rect(self.screen, FUEL_BAR_COLOR_FULL, (12, 12, int(200 * fuel_pct), 20))
        
        # B. Telemetry
        vel = self.env.lander.linearVelocity
        pos = self.env.lander.position
        
        # Create text surfaces
        texts = [
            f"X Speed: {vel.x:.2f} m/s",
            f"Y Speed: {vel.y:.2f} m/s",
            f"Altitude: {pos.y:.2f} m"
        ]
        
        for i, text in enumerate(texts):
            label = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(label, (10, 40 + (i * 20)))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None