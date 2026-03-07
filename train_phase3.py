import gymnasium as gym
import rocket_env  
from stable_baselines3 import PPO

# Load the NEW Phase 3 environment
env = gym.make("Phase3Final-v0")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

print("Starting PHASE 3 training... Please wait.")
model.learn(total_timesteps=500000)

# Save it under a new name to reflect the 3-thruster upgrade!
model.save("ppo_phase3_final_v0")
print("Training finished and model saved!")
env.close()


#FOR KAGGLE NOTEBOOK
"""import gymnasium as gym
import rocket_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment (faster training)
env = make_vec_env("Phase3Final-v0", n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

print("Starting PHASE 3 training... Please wait.")

model.learn(total_timesteps=500000)

model.save("/kaggle/working/ppo_phase3_final_v0")

print("Training finished and model saved!")

env.close()"""