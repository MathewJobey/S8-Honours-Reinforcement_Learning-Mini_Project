import gymnasium as gym
import rocket_env
import time
import numpy as np

# 1. Load the Phase 3 Environment (No AI model needed!)
env = gym.make("Phase3Final-v0", render_mode="human")

print("Starting the PURE FREEFALL test flight!")

episodes = 5 

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    
    print(f"\n--- Episode {ep + 1} Starting ---")
    
    while not done:
        # THE FIX: Hardcode the joystick to zero!
        # [Main Engine, Center Flaps, Nose Flaps]
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Pass the zero-action into the physics engine
        obs, reward, terminated, truncated, info = env.step(action)
        
        # DRAW THE FRAME! 
        # This tells PyGame to update the screen so you can see the movement
        env.render()
        
        # Check if the round is over (it hit the ground or flew off screen)
        done = terminated or truncated

    print(">>> Episode Finished.")
    
    # Pause for 2 seconds before the next episode starts
    # This gives you time to see where the rocket crashed
    time.sleep(2.0)

# Safely close the window
env.close()