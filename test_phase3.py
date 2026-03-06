import gymnasium as gym
import rocket_env
from stable_baselines3 import PPO
import time 

# 1. Load the NEW Phase 2 Environment
env = gym.make("Phase3Final-v0", render_mode="human")

# 2. Load the NEW Phase 2 Brain
model = PPO.load("ppo_phase3_final_v3")

print("Starting the PHASE 3 test flight!")

episodes = 5 

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    
    print(f"\n--- Episode {ep + 1} Starting ---")
    
    while not done:
        # Ask the AI for the best move
        action, _states = model.predict(obs, deterministic=True)
        
        # Take that action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # DRAW THE FRAME! 
        # This tells PyGame to update the screen so you can see the movement
        env.render()
        
        # Check if the round is over
        done = terminated or truncated

    print(">>> Episode Finished.")
    
    # Pause for 2 seconds before the next episode starts
    # This gives you time to see where the rocket ended up
    time.sleep(2.0)
# Safely close the window
env.close()