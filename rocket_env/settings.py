# Configuration Constants

# Screen / Rendering
FPS = 60
SCALE =5.0   #  affects how large the screen elements are
VIEWPORT_W = 800
VIEWPORT_H = 800

# Rocket Physics
# Strong engine to catch the fall (TWR > 2.0)
MAIN_ENGINE_POWER = 300.0 
SIDE_ENGINE_POWER = 20.0
INITIAL_FUEL = 100.0
FUEL_CONSUMPTION_RATE = 5.0

# Rocket Dimensions (Meters)
ROCKET_H_WIDTH = 1.0  # Wider body = more air resistance
ROCKET_HEIGHT = 5.0   
NOSE_HEIGHT = 2.0

# Landing Zone
PAD_WIDTH_METERS = 10.0  # Bigger pad for high-speed landings
PAD_HEIGHT_METERS = 1.0

# Terrain
GRAVITY_X = 0.0
GRAVITY_Y = -9.8  
WIND_POWER_MAX = 1.0

# Colors (R, G, B)
LANDER_COLOR = (128, 102, 230)      # Purple
LEG_COLOR = (200, 200, 200)         # Gray
FUEL_BAR_COLOR_FULL = (0, 255, 0)   # Green
FUEL_BAR_BORDER = (255, 255, 255)   # White
SKY_COLOR = (0,0,128)           # Dark Blue
GROUND_COLOR = (34, 139, 34)    # Dark Green
PAD_COLOR_1 = (255, 230, 0)       # Hazard Yellow
PAD_COLOR_2 = (20, 20, 20)        # Hazard Black (Stripes)