from arena_shooter_rl import train_ppo

model, vec_env = train_ppo(
    total_timesteps=2_000_000,
    n_envs=8,
    model_name="ppo_arena_shooter_pure_selfplay",
    checkpoint_dir="checkpoints",
)
vec_env.close()