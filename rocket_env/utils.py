import math

def get_true_altitude(lander, half_width, height):
    # Step 1: Define the four corners using the blueprint numbers handed to us
    bottom_left_corner = (-half_width, 0)
    bottom_right_corner = (half_width, 0)
    top_right_corner = (half_width, height)
    top_left_corner = (-half_width, height)
    
    # Step 2: Ask the physics engine where these corners are right now
    world_bl = lander.GetWorldPoint(bottom_left_corner)
    world_br = lander.GetWorldPoint(bottom_right_corner)
    world_tr = lander.GetWorldPoint(top_right_corner)
    world_tl = lander.GetWorldPoint(top_left_corner)
    
    # Step 3: Find the absolute lowest Y-coordinate among all four corners
    lowest_y = min(world_bl.y, world_br.y, world_tr.y, world_tl.y)
    
    # Step 4: Calculate distance to the top of the 1-meter concrete pad
    true_altitude = lowest_y - 1.0
    
    # Step 5: Hand the final answer back to the main game
    return true_altitude

def calculate_distance_reward(pos_x, true_altitude, previous_distance):
    # Step 1: Calculate the exact straight-line distance right now
    current_distance = math.sqrt(pos_x**2 + true_altitude**2)
    
    # Step 2: Safety check for the very first frame
    if previous_distance is None:
        previous_distance = current_distance
        
    # Step 3: Calculate the exact distance traveled
    progress = previous_distance - current_distance
    
    # Step 4: Scale the progress
    distance_reward = progress * 10.0
    
    # Step 5: Hand BOTH the points and the new memory back to the main game
    return distance_reward, current_distance

def calculate_posture_reward(tilt_rad, previous_tilt):
    # Step 1: Safety check for the very first frame
    if previous_tilt is None:
        previous_tilt = tilt_rad
        
    # Step 2: Calculate the exact rotation traveled this frame
    # Positive = getting straighter. Negative = tipping over.
    progress = previous_tilt - tilt_rad
    
    # Step 3: Scale the progress into clean whole numbers
    # We multiply by 20.0 because straightening up is a smaller, harder movement than falling
    posture_reward = progress * 20.0
    
    # Step 4: Hand BOTH the points and the new memory back to the main game
    return posture_reward, tilt_rad

def calculate_x_reward(pos_x, previous_x_dist):
    # Step 1: We only care about the absolute distance from the center (0.0)
    current_x_dist = abs(pos_x)
    
    # Step 2: Safety check for the very first frame
    if previous_x_dist is None:
        previous_x_dist = current_x_dist
        
    # Step 3: Calculate the exact horizontal distance traveled
    # Positive = moving towards the center line. Negative = drifting away.
    progress = previous_x_dist - current_x_dist
    
    # Step 4: Scale the progress into clean whole numbers
    # We multiply by 10.0, just like the main distance tracker
    x_reward = progress * 10.0
    
    # Step 5: Hand BOTH the points and the new absolute distance back
    return x_reward, current_x_dist