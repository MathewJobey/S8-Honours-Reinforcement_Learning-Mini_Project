# Configuration Constants

# Screen / Rendering
FPS = 60
SCALE =20.0   #  affects how large the screen elements are
VIEWPORT_W = 1000
VIEWPORT_H = 700

# Rocket Physics
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_FUEL = 100.0
FUEL_CONSUMPTION_RATE = 5.0  

# --- Rocket Dimensions (In Meters) ---
# Decoupling these from SCALE allows the zoom to work properly.
ROCKET_H_WIDTH = 0.5  # Half-width (Total width = 1.0 meter)
ROCKET_HEIGHT = 2.5   # Total Height
NOSE_HEIGHT = 1.0     # Height of the nose cone

# Landing Zone
PAD_WIDTH_PIXELS = 80
PAD_HEIGHT_PIXELS = 10
PAD_WIDTH_METERS = PAD_WIDTH_PIXELS / SCALE

# Terrain / World
GRAVITY_X = 0.0
GRAVITY_Y = -10.0  
WIND_POWER_MAX =1.0 # <--- CHANGED: Set to 0.0 to disable wind for now

# Colors (R, G, B)
LANDER_COLOR = (128, 102, 230)      # Purple
LEG_COLOR = (200, 200, 200)         # Gray
FUEL_BAR_COLOR_FULL = (0, 255, 0)   # Green
FUEL_BAR_BORDER = (255, 255, 255)   # White
SKY_COLOR = (0, 0, 0)               # Black
GROUND_COLOR = (255, 255, 255)      # White
PAD_COLOR = (255, 255, 0)           # Yellow
EXHAUST_COLOR = (255, 100, 0)       # Orange