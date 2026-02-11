import gymnasium as gym
from stable_baselines3 import PPO
from rocket_env.rocket_lander import RocketLander
import time

def main():
    # ---------------------------------------------------------
    # 1. Load the Environment
    # ---------------------------------------------------------
    # render_mode="human" prepares the visualizer, but you still
    # need to call env.render() in the loop to update the screen.
    env = RocketLander(render_mode="human")
    
    # ---------------------------------------------------------
    # 2. Load the Trained Brain
    # ---------------------------------------------------------
    model_path = "models/ppo_rocket_lander"
    
    print(f"Loading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully! Launching visualization...")
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}.zip'")
        print("Did you run train.py first?")
        return

    # ---------------------------------------------------------
    # 3. Watch it Fly!
    # ---------------------------------------------------------
    episodes = 5
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_score = 0
        
        print(f"\n--- Episode {ep + 1} Starting ---")
        
        while not done:
            # 1. Ask the AI for the best move
            action, _states = model.predict(obs, deterministic=True)
            
            # 2. Run Physics
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. DRAW THE FRAME (This was missing!)
            env.render()  
            
            total_score += reward
            done = terminated or truncated
            
            # Optional: Slow down slightly if it's too fast, 
            # though pygame.clock inside render() usually handles this.
            # time.sleep(0.01) 

        print(f">>> Episode Finished. Total Score: {total_score:.2f}")
        
        # Pause to see the landing result before resetting
        time.sleep(2.0)

    env.close()

if __name__ == "__main__":
    main()