import gymnasium as gym
import rocket_env  # This automatically runs the __init__.py we just wrote!

# 1. Create the environment using the ID we registered
env = gym.make("RocketLander-v0", render_mode="human")

# 2. Reset the world to start
observation, info = env.reset()

print("Environment created successfully!")
print(f"Observation Space: {env.observation_space.shape[0]} inputs")
print("Press Ctrl+C to stop...")

# 3. Run the loop
for _ in range(1000):
    # Take a random action (Main Engine 0-1, Steering -1 to 1)
    action = env.action_space.sample() 
    
    # Step the physics
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the frame (this relies on the render() method, which is currently empty!)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()