import gymnasium as gym
import rocket_env  # Registers the environment
import pygame

def run_drop_test():
    # Create the environment with render_mode='human' to see the window
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()

    print("--- DROP TEST STARTED ---")
    print("Watch the Y-Velocity on the HUD.")
    print("It should increase, then stabilize around -40 to -60 m/s.")

    running = True
    while running:
        # Action [0, 0] means Main Engine OFF, Side Thrusters OFF
        # We are purely observing gravity vs drag.
        action = [0, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the frame
        env.render()

        # Check for Pygame quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated:
            print("Impact! Resetting...")
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    run_drop_test()