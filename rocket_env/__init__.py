from gymnasium.envs.registration import register
# 3. Your new Phase 3 training environment
register(
    id='Phase3Final-v0',
    entry_point='rocket_env.phase3_final:Phase3Final',
    max_episode_steps=1100, 
)