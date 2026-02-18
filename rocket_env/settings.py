# Configuration Constants

# Screen / Rendering
FPS = 60
SCALE =10.0   #  affects how large the screen elements are
VIEWPORT_W = 700
VIEWPORT_H = 750

## Rocket Physics
# Starship is heavy! We need massive thrust to catch the fall.
# Increased from 300.0 to 1000.0 to handle the larger mass.
MAIN_ENGINE_POWER = 1000.0 
SIDE_ENGINE_POWER = 50.0   # Stronger cold-gas thrusters for the "flip"
INITIAL_FUEL = 100.0
FUEL_CONSUMPTION_RATE = 5.0

# --- STARSHIP DIMENSIONS (Based on your diagram) ---
# Real Starship: 50m High, 9m Wide (Ratio ~5.5:1)
# Sim Starship: 10m High, 1.8m Wide (Scaled down 1:5 for stability)
ROCKET_HEIGHT = 10.0   
ROCKET_H_WIDTH = 0.9   # Total width = 1.8m. (10 / 1.8 = 5.55 Ratio)(its half-width is 0.9m)
NOSE_HEIGHT = 2.5      # The nose cone is roughly top 1/4th

# Landing Zone
PAD_WIDTH_METERS = 20.0  # Slightly wider pad for the larger ship
PAD_HEIGHT_METERS = 1.0

# Terrain
GRAVITY_X = 0.0
GRAVITY_Y = -9.8  
WIND_POWER_MAX = 1.0

# Colors (R, G, B)
LANDER_COLOR = (192, 192, 192)      # Stainless Steel Silver
LEG_COLOR = (40, 40, 40)            # Dark Grey Legs
FUEL_BAR_COLOR_FULL = (0, 255, 0)   # Green
FUEL_BAR_BORDER = (255, 255, 255)   # White
SKY_COLOR = (0,0,128)           # Dark Blue
GROUND_COLOR = (34, 139, 34)    # Dark Green
PAD_COLOR_1 = (255, 230, 0)       # Hazard Yellow
PAD_COLOR_2 = (20, 20, 20)        # Hazard Black (Stripes)