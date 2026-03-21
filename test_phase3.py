import gymnasium as gym
import rocket_env
from stable_baselines3 import SAC
import time 
import os
import glob

# ---> THE FIX 1: The Coordinate Trick <---
# Force the graphics engine to spawn the window at the top-left of the monitor!
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# ==========================================
# STEP 1: THE UPGRADED RADAR
# ==========================================
def get_model_to_test():
    models_base = "./models"
    
    if not os.path.exists(models_base):
        print("Error: No 'models' folder found!")
        return None
        
    # Scan for all available Run folders
    existing_runs = []
    for d in os.listdir(models_base):
        if d.startswith("Run "):
            try:
                num = int(d.split(" ")[1])
                existing_runs.append(num)
            except ValueError:
                pass
                
    if not existing_runs:
        print("Error: No 'Run X' folders found inside the models directory!")
        return None

    # Sort the runs so they display nicely
    existing_runs.sort()
    max_run = max(existing_runs)
    
    # Ask the user which run they want to see
    print(f"\nAvailable AI Brains: {[f'Run {r}' for r in existing_runs]}")
    choice = input(f"Which run do you want to test? (Press Enter to default to Run {max_run}): ").strip()
    
    # Process the choice
    if choice == "":
        target_run = max_run
    else:
        try:
            target_run = int(choice)
            if target_run not in existing_runs:
                print(f"Run {target_run} not found. Defaulting to Run {max_run}.")
                target_run = max_run
        except ValueError:
            print(f"Invalid input. Defaulting to Run {max_run}.")
            target_run = max_run
            
    # Go into the chosen folder and find the newest .zip file
    target_dir = os.path.join(models_base, f"Run {target_run}")
    list_of_files = glob.glob(f"{target_dir}/*.zip")
    
    if not list_of_files:
        print(f"Error: No .zip checkpoints found inside Run {target_run}!")
        return None
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"\n>>> Locked onto checkpoint: {latest_file}")
    return latest_file

# ==========================================
# STEP 2: LOAD AND FLY
# ==========================================
newest_model_path = get_model_to_test()

if newest_model_path is None:
    print("Cannot start test flight. Exiting.")
else:
    # Load the environment and the AI
    env = gym.make("Phase3Final-v0", render_mode="human")
    model = SAC.load(newest_model_path, env=env)
    
    print("\nStarting the PHASE 3 test flight with SAC!")
    
    # ---> THE FIX: The Windows Focus Hacker <---
    # 1. Force the window to physically build itself right now
    env.reset()
    env.render()
    
    # 2. Talk directly to Windows to pull it to the front!
    # 2. Talk directly to Windows to pull it to the front!
    try:
        import pygame
        import ctypes
        
        # Grab the secret ID of the Pygame window
        hwnd = pygame.display.get_wm_info()['window']
        
        # HACK 1: Simulate pressing the 'ALT' key to bypass the Windows security lock
        ctypes.windll.user32.keybd_event(0x12, 0, 0, 0) # ALT down
        ctypes.windll.user32.SetForegroundWindow(hwnd)  # Demand focus
        ctypes.windll.user32.keybd_event(0x12, 0, 2, 0) # ALT up
        
        # HACK 2: Pin the window as "Always On Top" so the code editor cannot hide it
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        
    except Exception as e:
        pass
    
    episodes = 5 
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        # Create an empty piggy bank for this episode's score!
        total_reward = 0.0 
        
        print(f"\n--- Episode {ep + 1} Starting ---")
        
        while not done:
            # Ask the AI for the best slider positions
            action, _states = model.predict(obs, deterministic=True)
            
            # Apply those slider positions in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Add this frame's reward to our running total!
            total_reward += reward
            
            # DRAW THE FRAME! 
            env.render()
            
            # Check if the round is over
            done = terminated or truncated
    
        # Print the final score when the episode finishes!
        print(f">>> Episode Finished. Total Score: {total_reward:.2f}")
        time.sleep(2.0)
    
    env.close()