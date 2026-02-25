from gymnasium.envs.registration import register
register(
    id="GridWorld-v0",
    entry_point="gridworld.gymnasium_env.envs:GridWorldEnv",
)
