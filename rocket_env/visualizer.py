import pygame
import math
import numpy as np
import random 
from .settings import * # Import all global constants including ROCKET dimensions

class RocketVisualizer:
    def __init__(self, env):
        self.env = env
        self.screen = None
        self.clock = None   
        self.font = None
        self.current_flame_power = 0.0
        self.stars = [] # Stores star positions
    
    def init_window(self):
        """Initializes Pygame window and generates static stars."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

            # --- GENERATE STARS ONCE ---
            # These positions stay fixed as long as the window is open
            self.stars = []
            for _ in range(150):
                x = random.randint(0, VIEWPORT_W)
                y = random.randint(0, VIEWPORT_H - 50) 
                radius = random.randint(1, 2)
                self.stars.append((x, y, radius))
            
    def render(self, mode="human"):
        """Renders the world with stars, ground, and hazard pad."""
        self.init_window()

        # 1. Background (Sky)
        self.screen.fill(SKY_COLOR)

        # --- DRAW STARS ---
        for x, y, radius in self.stars:
            # Simple white dots
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius)

        # --- 2. GROUND (Light Green) ---
        ground_height = 20
        ground_y = VIEWPORT_H - ground_height
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, ground_y, VIEWPORT_W, ground_height))
        # Ground outline
        pygame.draw.line(self.screen, (50, 100, 50), (0, ground_y), (VIEWPORT_W, ground_y), 2)
        
        # --- 3. LANDING PAD (Hazard Stripes) ---
        pad_w = int(PAD_WIDTH_METERS * SCALE)
        pad_h = int(PAD_HEIGHT_METERS * SCALE)
        if pad_w < 1: pad_w = 1
        if pad_h < 1: pad_h = 1
        
        pad_x = (VIEWPORT_W / 2) - (pad_w / 2)
        pad_y = ground_y - pad_h 

        # Pad Surface for clipping
        pad_surf = pygame.Surface((pad_w, pad_h))
        pad_surf.fill(PAD_COLOR_1) # Yellow

        # Stripes
        stripe_w = 0.5 * SCALE  
        gap = 0.5 * SCALE       
        slant = 0.2 * SCALE     
        
        for i in range(int(-pad_h), int(pad_w + pad_h), int(stripe_w + gap)):
            points = [
                (i, pad_h),                 
                (i + stripe_w, pad_h),      
                (i + stripe_w + slant, 0),  
                (i + slant, 0)              
            ]
            pygame.draw.polygon(pad_surf, PAD_COLOR_2, points)

        # Blit pad and draw outline
        self.screen.blit(pad_surf, (pad_x, pad_y))
        pygame.draw.rect(self.screen, (0,0,0), (pad_x, pad_y, pad_w, pad_h), 2)

        # 4. Draw Rocket Parts
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
        """Draws legs in BLACK and hides the physics rocket body."""
        BLACK = (25, 25, 25)
        
        for obj in self.env.drawlist:
            # 1. Leg: Draw black polygon
            if obj != self.env.lander:
                for f in obj.fixtures:
                    trans = f.body.transform
                    path = [trans * v for v in f.shape.vertices]
                    pixel_path = [
                        (SCALE * v[0] + VIEWPORT_W / 2, VIEWPORT_H - SCALE * v[1] - 20)
                        for v in path
                    ]
                    pygame.draw.polygon(self.screen, BLACK, pixel_path)
            # 2. Rocket: Call details function
            elif obj == self.env.lander:
                self._draw_rocket_details(obj)

    def _draw_rocket_details(self, lander):
        """Draws the rocket with clean, sharp outlines."""
        pos = lander.position
        angle = lander.angle
        
        # Helper: Rotate and Scale to screen
        def transform_point(local_x, local_y):
            rot_x = local_x * math.cos(angle) - local_y * math.sin(angle)
            rot_y = local_x * math.sin(angle) + local_y * math.cos(angle)
            screen_x = (SCALE * (pos.x + rot_x)) + (VIEWPORT_W / 2)
            screen_y = VIEWPORT_H - (SCALE * (pos.y + rot_y)) - 20
            return (screen_x, screen_y)

        # Colors & Dimensions
        WHITE = (245, 245, 245)
        BLACK = (25, 25, 25)
        RED = (255, 50, 50)
        BLUE = (0, 200, 255)
        half_w = ROCKET_H_WIDTH
        body_h = ROCKET_HEIGHT
        nose_h = NOSE_HEIGHT

        # ================= BODY =================
        body_points = [
            transform_point(-half_w, 0),
            transform_point( half_w, 0),
            transform_point( half_w, body_h),
            transform_point(-half_w, body_h),
        ]
        # Fill White
        pygame.draw.polygon(self.screen, WHITE, body_points)
        # Sharp Black Outline (Thickness 3)
        pygame.draw.polygon(self.screen, BLACK, body_points, 3)

        # ================= NOSE =================
        nose_points = [
            transform_point(-half_w, body_h),
            transform_point( half_w, body_h),
            transform_point(0, body_h + nose_h)
        ]
        pygame.draw.polygon(self.screen, RED, nose_points)
        pygame.draw.polygon(self.screen, BLACK, nose_points, 2)

        # ================= FINS =================
        fin_h = body_h * 0.4
        fin_w = half_w * 0.8
        fin_drop = body_h * 0.1

        # Left Fin
        left_fin = [
            transform_point(-half_w, 0),
            transform_point(-half_w, fin_h),
            transform_point(-half_w - fin_w, -fin_drop)
        ]
        pygame.draw.polygon(self.screen, RED, left_fin)
        pygame.draw.polygon(self.screen, BLACK, left_fin, 2)

        # Right Fin
        right_fin = [
            transform_point(half_w, 0),
            transform_point(half_w, fin_h),
            transform_point(half_w + fin_w, -fin_drop)
        ]
        pygame.draw.polygon(self.screen, RED, right_fin)
        pygame.draw.polygon(self.screen, BLACK, right_fin, 2)

        # ================= ENGINE =================
        engine_h = 0.2
        engine_points = [
            transform_point(-half_w * 0.6, -engine_h),
            transform_point( half_w * 0.6, -engine_h),
            transform_point( half_w * 0.6, 0),
            transform_point(-half_w * 0.6, 0),
        ]
        pygame.draw.polygon(self.screen, BLACK, engine_points)

        # ================= WINDOW =================
        window_center = transform_point(0, body_h * 0.75)
        radius = int(SCALE * (half_w * 0.4))
        # Fill Blue
        pygame.draw.circle(self.screen, BLUE, (int(window_center[0]), int(window_center[1])), radius)
        # Outline Black
        pygame.draw.circle(self.screen, BLACK, (int(window_center[0]), int(window_center[1])), radius, 2)


    def _draw_exhaust(self):
        """Draws a multi-layered, flickering flame."""
        power = getattr(self.env, 'main_engine_power', 0.0)
        self.current_flame_power += (power - self.current_flame_power) * 0.2
        if self.current_flame_power < 0.05: return

        pos = self.env.lander.position
        angle = self.env.lander.angle
        
        # Offset relative to center (0,0). Nozzle tip is at y = -0.2m
        offset_y = -0.2 
        nozzle_x = pos.x - (offset_y * math.sin(angle))
        nozzle_y = pos.y + (offset_y * math.cos(angle))

        # Time-based flicker
        t = pygame.time.get_ticks() * 0.02
        flicker_global = (math.sin(t) * 0.05) + (math.cos(t * 3) * 0.02)

        flame_layers = [
            ((255, 100, 0), 0.4, 1.8),
            ((255, 200, 0), 0.25, 1.5),
            ((255, 255, 255), 0.12, 1.2)
        ]

        def to_screen(x, y):
            sx = (SCALE * x) + (VIEWPORT_W / 2)
            sy = VIEWPORT_H - (SCALE * y) - 20
            return (int(sx), int(sy))

        for color, width_base, len_mult in flame_layers:
            w_metric = width_base + (flicker_global * 0.05)
            l_metric = (self.current_flame_power * len_mult) + flicker_global

            tip_x = nozzle_x - math.sin(angle) * l_metric
            tip_y = nozzle_y - math.cos(angle) * l_metric
            
            base_left_x = nozzle_x - math.cos(angle) * w_metric
            base_left_y = nozzle_y + math.sin(angle) * w_metric
            
            base_right_x = nozzle_x + math.cos(angle) * w_metric
            base_right_y = nozzle_y - math.sin(angle) * w_metric

            points = [
                to_screen(base_left_x, base_left_y),
                to_screen(base_right_x, base_right_y),
                to_screen(tip_x, tip_y)
            ]
            pygame.draw.polygon(self.screen, color, points)


    def _draw_hud(self):
        """Draws the Fuel Bar and Telemetry."""
        # Telemetry
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

        # Fuel Bar
        fuel_pct = max(0, self.env.fuel_left / INITIAL_FUEL)
        bar_width, bar_height = 200, 20
        right_margin, top_margin = 20, 10
        bar_x = VIEWPORT_W - bar_width - right_margin
        bar_y = top_margin + 25 
        
        heading_label = self.font.render("FUEL", True, (255, 255, 255))
        self.screen.blit(heading_label, (bar_x, top_margin))
        
        # Border
        pygame.draw.rect(self.screen, FUEL_BAR_BORDER, (bar_x-1, bar_y-1, bar_width + 6, bar_height + 6), 2)
        # Fill
        pygame.draw.rect(self.screen, FUEL_BAR_COLOR_FULL, (bar_x + 2, bar_y + 2, int(bar_width * fuel_pct), bar_height))
        
        pct_text = f"{fuel_pct * 100:.1f}%"
        pct_label = self.font.render(pct_text, True, (255, 255, 255))
        self.screen.blit(pct_label, (bar_x - pct_label.get_width() - 10, bar_y))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None