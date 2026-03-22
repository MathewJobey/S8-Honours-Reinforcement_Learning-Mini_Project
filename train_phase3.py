import gymnasium as gym
import rocket_env
import os
import glob
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import load_results

# ==========================================
# STEP 1: THE DUAL-FOLDER MANAGER & THE ARTIST
# ==========================================
def get_synchronized_run_folders():
    rewards_base = "./rewards_graphs"
    models_base = "./models"
    
    # Ensure the base folders exist
    os.makedirs(rewards_base, exist_ok=True)
    os.makedirs(models_base, exist_ok=True)
    
    # Find the highest "Run X" number across the folders
    existing_runs = []
    for d in os.listdir(rewards_base):
        if d.startswith("Run "):
            try:
                num = int(d.split(" ")[1])
                existing_runs.append(num)
            except ValueError:
                pass
                
    max_run = max(existing_runs) if existing_runs else 0
    return rewards_base, models_base, max_run

class RewardPlotCallback(BaseCallback):
    def __init__(self, log_dir, plot_freq, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self.last_plot_step = 0 # <--- The memory tracker!

    def _on_step(self) -> bool:
        # Use true timesteps so it actually draws!
        if self.num_timesteps - self.last_plot_step >= self.plot_freq:
            self.plot_rewards()
            self.last_plot_step = self.num_timesteps
        return True

    def plot_rewards(self):
        try:
            # 1. Load the secret spreadsheet
            df = load_results(self.log_dir)
            if len(df) < 10:
                return 
            
            # ==========================================
            # GRAPH 1: THE REWARD GRAPH (Your original code)
            # ==========================================
            fig = plt.figure(figsize=(10, 5))
            plt.plot(df['r'].values, label='episode reward', alpha=0.8, color='tab:blue')
            moving_avg = df['r'].rolling(window=100, min_periods=1).mean()
            plt.plot(moving_avg.values, label='moving avg', color='tab:orange', linewidth=2)
            
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.legend()
            
            plot_path = os.path.join(self.log_dir, f"reward_graph_step_{self.num_timesteps}.png")
            plt.savefig(plot_path)
            plt.close(fig) 
            
            # ==========================================
            # GRAPH 2: THE EPISODE LENGTH GRAPH (The new addition!)
            # ==========================================
            # 2. Create a brand new blank canvas for the second picture
            fig2 = plt.figure(figsize=(10, 5))
            
            # 3. Plot the raw episode lengths (the 'l' column) in blue
            plt.plot(df['l'].values, label='episode length', alpha=0.8, color='tab:blue')
            
            # 4. Calculate the 100-episode moving average for the length
            moving_avg_len = df['l'].rolling(window=100, min_periods=1).mean()
            
            # 5. Plot the moving average in GREEN
            plt.plot(moving_avg_len.values, label='moving avg', color='tab:red', linewidth=2)
            
            # 6. Add labels so we know what we are looking at
            plt.xlabel('episode')
            plt.ylabel('episode length (frames)')
            plt.legend()
            
            # 7. Save this second picture with a new name
            plot_path2 = os.path.join(self.log_dir, f"length_graph_step_{self.num_timesteps}.png")
            plt.savefig(plot_path2)
            
            # 8. Throw away the canvas to keep the computer's memory clean
            plt.close(fig2)
            
            print(f"\n[SUCCESS: Reward AND Length graphs saved at {self.num_timesteps} steps!]")
            
        except Exception as e:
            print(f"\n[ERROR: Failed to draw graphs! Reason: {e}]")

# ==========================================
# STEP 2: RADAR & USER INPUT
# ==========================================
rewards_base, models_base, max_run = get_synchronized_run_folders()

latest_model_path = None
is_new_run = True 

# If there is at least one run, point the radar at that specific folder!
if max_run > 0:
    latest_models_dir = os.path.join(models_base, f"Run {max_run}")
    list_of_files = glob.glob(f"{latest_models_dir}/*.zip")
    
    if list_of_files:
        latest_model_path = max(list_of_files, key=os.path.getctime)
        print(f"\nRadar found an existing save file in Run {max_run}: {os.path.basename(latest_model_path)}")
        choice = input("Do you want to resume training from this save? (y/n): ").strip().lower()
        if choice == 'y':
            is_new_run = False

# ==========================================
# STEP 3: CREATE THE MATCHING FOLDERS
# ==========================================
if is_new_run:
    max_run += 1 # Mathematically advance to the next run number

# Create the exact paths for THIS specific run
current_rewards_dir = os.path.join(rewards_base, f"Run {max_run}")
current_models_dir = os.path.join(models_base, f"Run {max_run}")

# Physically build the folders on your hard drive
os.makedirs(current_rewards_dir, exist_ok=True)
os.makedirs(current_models_dir, exist_ok=True)

print(f"\nLogging Graphs and CSVs to: {current_rewards_dir}")
print(f"Saving AI Brain files to: {current_models_dir}\n")

# ==========================================
# STEP 4: ENVIRONMENT & BRAIN SETUP
# ==========================================
env = make_vec_env("Phase3Final-v0", n_envs=16, monitor_dir=current_rewards_dir)

if not is_new_run:
    print("Resuming training from the previous session...")
    model = SAC.load(latest_model_path, env=env, device="cuda")
    
    # ---> THE NEW RULE: Load the Flashcards <---
    # Construct the exact path where the massive memory file should be
    buffer_path = os.path.join(latest_models_dir, "sac_phase3_replay_buffer.pkl")
    
    # Check if the file actually exists on your hard drive before trying to load it
    if os.path.exists(buffer_path):
        print("Loading memory flashcards into RAM... (This might take a few seconds)")
        model.load_replay_buffer(buffer_path)
    else:
        print("No memory flashcards found. Starting with a fresh memory bank.")
else:
    print("Starting a brand new brain! (Old runs are kept safe)")
    # 1. Define the massive brain structure (3 hidden layers, 512 neurons each)
    custom_architecture = dict(net_arch=[512, 512, 512])
    
    model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=2_000_000,
    batch_size=6250,
    learning_starts = 10000,
    policy_kwargs=custom_architecture,
    gamma=0.995,
    device="cuda",
    verbose=1
)

# ==========================================
# STEP 5: THE CALLBACK BUNDLE
# ==========================================
# 25000 total steps / 16 environments = 1562 ticks per save
checkpoint_callback = CheckpointCallback(
    save_freq=1562, 
    save_path=current_models_dir, 
    name_prefix="sac_phase3"
)

plot_callback = RewardPlotCallback(log_dir=current_rewards_dir, plot_freq=100000)
callback_list = CallbackList([checkpoint_callback, plot_callback])

# ==========================================
# STEP 6: START THE MARATHON (Infinite Mode)
# ==========================================
print("\nStarting infinite training! Press Ctrl+C in the terminal to stop and save at any time.")

try:
    # 1. The Try Block: Run the marathon
    # int(1e9) is 1,000,000,000 steps. Practically infinite!
    model.learn(total_timesteps=int(1e9), callback=callback_list, reset_num_timesteps=False)

except KeyboardInterrupt:
    # 2. The Except Block: Catch the manual stop
    # If you press Ctrl+C, Python jumps straight here instead of crashing
    print("\nTraining manually interrupted by user! Initiating safe shutdown...")

finally:
    # 3. The Finally Block: The guaranteed save
    # This runs whether it hits 1 billion steps OR you manually stopped it
    final_path = os.path.join(current_models_dir, "sac_phase3_final_v0")
    model.save(final_path)
    print(f"Training finished and brain saved to: {final_path}")
    
    # ---> THE NEW RULE: Save the Flashcards <---
    buffer_path = os.path.join(current_models_dir, "sac_phase3_replay_buffer.pkl")
    print("Saving massive memory buffer to disk... (Do NOT close the terminal yet!)")
    model.save_replay_buffer(buffer_path)
    print("Flashcards safely saved! You can now close the terminal.")