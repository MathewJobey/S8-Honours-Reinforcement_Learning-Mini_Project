import gymnasium as gym
import rocket_env  
from stable_baselines3 import PPO

# 1. Create the environment
env = gym.make("Phase1Hover-v0")

# 2. Create the AI model (the brain)
# "MlpPolicy" is a standard, beginner-friendly neural network
# verbose=1 lets you see the training progress on your screen
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the AI
print("Starting training... Please wait.")
# The AI will take 100,000 practice actions
model.learn(total_timesteps=500000)

# 4. Save the trained brain
# This saves a file named "ppo_phase1_hover.zip" in your folder
model.save("ppo_phase1_hover")
print("Training finished and model saved!")

# 5. Close the environment
env.close()