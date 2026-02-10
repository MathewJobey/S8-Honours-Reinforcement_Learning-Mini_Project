import gymnasium as gym
import rocket_env
import time  # Import time to add pauses

# 1. Create the environment
env = gym.make("RocketLander-v0", render_mode="human")

# 2. Reset the world
observation, info = env.reset()

print("Environment created!")
print("Press Ctrl+C to stop...")

# 3. Run the loop
for _ in range(2000):
    # Take a random action
    action = env.action_space.sample() 
    
    # Step the physics
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    if terminated or truncated:
        # --- IMPROVEMENT: PAUSE ON CRASH ---
        # If we crashed or landed, pause for 1 second so we can see what happened.
        print(f"Episode Finished. Reward: {reward:.2f}")
        time.sleep(1.0) 
        
        # Then reset
        observation, info = env.reset()

env.close()