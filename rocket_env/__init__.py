from gymnasium.envs.registration import register

# 1. Your original environment (Full game)
register(
    id='RocketLander-v0',
    entry_point='rocket_env.rocket_lander:RocketLander',
    max_episode_steps=1000, 
)

# 2. Your new Phase 1 training environment
register(
    id='Phase1Hover-v0',
    entry_point='rocket_env.phase1_hover:Phase1Hover',
    max_episode_steps=2000, 
)