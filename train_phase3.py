import gymnasium as gym
import rocket_env
import os
import glob
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Step 1: Create a safe folder for the backups
os.makedirs("./models", exist_ok=True)

# Step 2: Create the parallel environments
env = make_vec_env("Phase3Final-v0", n_envs=4)

# Step 3: The Radar Function
def get_latest_model():
    list_of_files = glob.glob("models/*.zip")
    if not list_of_files:
        return None  # Return nothing if the folder is empty
    return max(list_of_files, key=os.path.getctime)

# Step 4: Ask the user to Load or Create the Brain
latest_model_path = get_latest_model()

if latest_model_path is not None:
    # THE FIX: Pause the script and ask the user what to do!
    print(f"\nRadar found an existing save file: {latest_model_path}")
    choice = input("Do you want to resume training from this save? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("Resuming training from the previous session...")
        model = SAC.load(latest_model_path, env=env, device="cuda")
    else:
        print("Erasing old saves and starting a brand new brain...")
        # Clean the slate: delete old checkpoints so they don't mix with the new run
        for old_file in glob.glob("models/*.zip"):
            os.remove(old_file)
            
        # Create the new brain
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4, 
            batch_size=256,
            gamma=0.995,
            device="cuda", 
            verbose=1
        )
else:
    print("No save file found. Starting a brand new brain...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4, 
        batch_size=256,
        gamma=0.995,
        device="cuda", 
        verbose=1
    )

# Step 5: Setup the Auto-Save (Checkpointing)
checkpoint_callback = CheckpointCallback(
    save_freq=25000, 
    save_path="./models/",
    name_prefix="sac_phase3"
)

# Step 6: Train the model (and don't reset the counter!)
model.learn(total_timesteps=1500000, callback=checkpoint_callback, reset_num_timesteps=False)

# Step 7: Save the highly-tuned final model
model.save("sac_phase3_final_v0")
print("Training finished and model saved!")