import gymnasium as gym
import rocket_env
import time

# Create Env
env = gym.make("RocketLander-v0", render_mode="human")
observation, info = env.reset()

print("Environment created!")
print("Press Ctrl+C to stop...")

# Variable to track the TOTAL score of the current episode
total_score = 0.0

for _ in range(2000):
    # Random Action (Just for testing visuals)
    action = env.action_space.sample() 
    
    # Run Physics
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Add this frame's reward to the total
    total_score += reward
    
    # Draw
    env.render()
    
    if terminated or truncated:
        # Print the Final Report Card
        print(f">>> Episode Finished. Total Score: {total_score:.2f}")
        
        # Pause to let you see the landing/crash
        time.sleep(1.0)
        
        # Reset
        observation, info = env.reset()
        total_score = 0.0

env.close()             