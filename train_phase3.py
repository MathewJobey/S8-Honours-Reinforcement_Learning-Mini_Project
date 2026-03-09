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

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            self.plot_rewards()
        return True

    def plot_rewards(self):
        try:
            df = load_results(self.log_dir)
            if len(df) < 10:
                return 
            
            fig = plt.figure(figsize=(10, 5))
            plt.plot(df['r'].values, label='episode reward', alpha=0.8, color='tab:blue')
            moving_avg = df['r'].rolling(window=100, min_periods=1).mean()
            plt.plot(moving_avg.values, label='moving avg', color='tab:orange', linewidth=2)
            
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.legend()
            
            plot_path = os.path.join(self.log_dir, f"reward_graph_step_{self.num_timesteps}.png")
            plt.savefig(plot_path)
            plt.close(fig) 
        except Exception as e:
            pass

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
env = make_vec_env("Phase3Final-v0", n_envs=4, monitor_dir=current_rewards_dir)

if not is_new_run:
    print("Resuming training from the previous session...")
    model = SAC.load(latest_model_path, env=env, device="cuda")
else:
    print("Starting a brand new brain! (Old runs are kept safe)")
    # The os.remove() deletion loop has been completely removed!
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4, 
        batch_size=256,
        gamma=0.995,
        device="cuda", 
        verbose=1
    )

# ==========================================
# STEP 5: THE CALLBACK BUNDLE
# ==========================================
checkpoint_callback = CheckpointCallback(
    save_freq=25000, 
    save_path=current_models_dir, # Rout the zip files to the specific models/Run X/ folder
    name_prefix="sac_phase3"
)

plot_callback = RewardPlotCallback(log_dir=current_rewards_dir, plot_freq=25000)
callback_list = CallbackList([checkpoint_callback, plot_callback])

# ==========================================
# STEP 6: START THE MARATHON
# ==========================================
model.learn(total_timesteps=1500000, callback=callback_list, reset_num_timesteps=False)

# Save the absolute final masterpiece inside the specific run folder
final_path = os.path.join(current_models_dir, "sac_phase3_final_v0")
model.save(final_path)
print("Training finished and model saved!")