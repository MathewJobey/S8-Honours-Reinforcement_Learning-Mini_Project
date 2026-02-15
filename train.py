import gymnasium as gym
from stable_baselines3 import PPO
import os

from rocket_env.rocket_lander import RocketLander

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # CHANGED: 300k -> 1 Million. 
    # The "Belly Flop" is a complex move; the AI needs more practice time!
    TIMESTEPS = 100000        
    
    SAVE_PATH = "models/ppo_rocket_lander"
    LOG_DIR = "logs"

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # ---------------------------------------------------------
    # 2. Initialize the Environment
    # ---------------------------------------------------------
    # render_mode=None allows the simulation to run at maximum CPU speed.
    env = RocketLander(render_mode=None)

    # ---------------------------------------------------------
    # 3. Define the PPO Agent
    # ---------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
        # OPTIONAL: You can slightly increase exploration if it gets stuck
        ent_coef=0.01 #change from default 0.0 to encourage more exploration
    )

    print("\n" + "="*50)
    print(f"TRAINING INITIATED: {TIMESTEPS} steps")
    print("="*50 + "\n")

    # ---------------------------------------------------------
    # 4. Start Learning
    # ---------------------------------------------------------
    model.learn(total_timesteps=TIMESTEPS)

    # ---------------------------------------------------------
    # 5. Save the Trained Model
    # ---------------------------------------------------------
    model.save(SAVE_PATH)
    
    print("\n" + "="*50)
    print(f"TRAINING COMPLETE: Model saved as '{SAVE_PATH}.zip'")
    print("="*50 + "\n")
    
    env.close()

if __name__ == "__main__":
    main()