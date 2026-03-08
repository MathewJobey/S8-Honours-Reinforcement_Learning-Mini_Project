import gymnasium as gym
import rocket_env
from stable_baselines3 import SAC
import time 
import os
import glob

# Step 1: The Radar Function
def get_latest_model():
    # Search the models folder for anything ending in .zip
    list_of_files = glob.glob("models/*.zip")
    
    # Safety Net: If the folder is empty, load the final save
    if not list_of_files:
        print("No checkpoints found. Loading the final model...")
        return "sac_phase3_final_v0"
        
    # Sort all found files by their creation time and grab the newest one
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Found latest checkpoint: {latest_file}")
    
    return latest_file

# Step 2: Load the Phase 3 Environment
env = gym.make("Phase3Final-v0", render_mode="human")

# Step 3: Automatically find the newest save file!
newest_model_path = get_latest_model()

# Step 4: Load the SAC Brain using the path we just found
model = SAC.load(newest_model_path)

print("Starting the PHASE 3 test flight with SAC!")

episodes = 5 

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    
    print(f"\n--- Episode {ep + 1} Starting ---")
    
    while not done:
        # Ask the AI for the best slider positions
        action, _states = model.predict(obs, deterministic=True)
        
        # Apply those slider positions in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # DRAW THE FRAME! 
        env.render()
        
        # Check if the round is over (Crashed or Landed)
        done = terminated or truncated

    print(">>> Episode Finished.")
    
    # Pause for 2 seconds before the next episode starts
    time.sleep(2.0)

# Safely close the window
env.close()