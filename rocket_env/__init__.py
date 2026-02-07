from gymnasium.envs.registration import register

# This tells Gymnasium that "RocketLander-v0" exists 
# and where to find the class (rocket_env.rocket_lander:RocketLander).
register(
    id='RocketLander-v0',
    entry_point='rocket_env.rocket_lander:RocketLander',
    max_episode_steps=1000, #If the rocket doesn't land within 1000 frames (20 seconds), the game is forced to end (preventing it from hovering forever).
)