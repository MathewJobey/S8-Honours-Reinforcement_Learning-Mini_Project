from gymnasium.envs.registration import register

# 1. Your original environment (Full game)
register(
    id='RocketLander-v0',
    entry_point='rocket_env.rocket_lander:RocketLander',
    max_episode_steps=1000, 
)

# 2. Your new Phase 1 training environment
register(
    id='Phase2Descent-v0',
    entry_point='rocket_env.phase2_descent:Phase2Descent',
    max_episode_steps=2000, 
)

# 3. Your new Phase 3 training environment
register(
    id='Phase3Final-v0',
    entry_point='rocket_env.phase3_final:Phase3Final',
    max_episode_steps=2000, 
)