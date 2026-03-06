import gymnasium as gym
import rocket_env  
from stable_baselines3 import PPO

# Load the NEW Phase 3 environment
env = gym.make("Phase3Final-v0")

model = PPO("MlpPolicy", env, verbose=1)

print("Starting PHASE 3 training... Please wait.")
model.learn(total_timesteps=500000)

# Save it under a new name to reflect the 3-thruster upgrade!
model.save("ppo_phase3_final_v3")
print("Training finished and model saved!")
env.close()