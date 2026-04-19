from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from arena_shooter_rl import (
    ArenaShooterEnv,
    EpisodeStatsCallback,
    SelfPlayCallback,
    SelfPlayManager,
    build_model_opponent,
    build_vec_env,
    evaluate_policy_rollout,
    human_action_from_inputs,
    linear_schedule,
    load_model_bundle,
    normalize_single_obs,
)


def _model_stem(model_path: str) -> str:
    return Path(model_path).stem


def _vecnorm_path_for(model_path: str) -> Path:
    return Path(model_path).with_name(f"{_model_stem(model_path)}_vecnormalize.pkl")


class ScriptedCurriculumCallback:
    @staticmethod
    def build(total_timesteps: int, start_difficulty: float = 0.20, end_difficulty: float = 0.95, verbose: int = 0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for training.") from exc

        class _ScriptedCurriculumCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=verbose)
                self.last_bucket = -1

            def _on_step(self) -> bool:
                progress = min(1.0, self.num_timesteps / max(1, total_timesteps))
                difficulty = start_difficulty + (end_difficulty - start_difficulty) * progress
                self.training_env.env_method("set_scripted_difficulty", difficulty)
                bucket = int(progress * 10)
                if self.verbose > 0 and bucket != self.last_bucket:
                    self.last_bucket = bucket
                    print(f"[scripted curriculum] timesteps={self.num_timesteps} scripted_difficulty={difficulty:.2f}")
                return True

        return _ScriptedCurriculumCallback()


def train_basic_bot(
    total_timesteps: int = 600_000,
    n_envs: int = 8,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "arena_basic_bot",
    start_difficulty: float = 0.20,
    end_difficulty: float = 0.95,
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training.") from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(n_envs=n_envs, scripted_difficulty=start_difficulty, normalize=True)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(2.5e-4, 5e-5),
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(checkpoint_path / "tb_logs"),
        verbose=1,
        device="auto",
    )

    callbacks = CallbackList(
        [
            EpisodeStatsCallback.build(verbose=1),
            ScriptedCurriculumCallback.build(
                total_timesteps=total_timesteps,
                start_difficulty=start_difficulty,
                end_difficulty=end_difficulty,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=max(50_000 // max(1, n_envs), 1),
                save_path=str(checkpoint_path / "intermediate_basic"),
                name_prefix=model_name,
            ),
        ]
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model_path = checkpoint_path / f"{model_name}.zip"
    vecnorm_path = checkpoint_path / f"{model_name}_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    print(f"saved basic model -> {model_path}")
    print(f"saved basic vecnormalize -> {vecnorm_path}")
    return str(model_path)


def record_human_demos(
    opponent_model_path: str,
    out_dir: str = "demos/human",
    start_seed: int = 42,
):
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for demo recording.") from exc

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = ArenaShooterEnv(render_mode="human")
    env.set_opponent_policy(build_model_opponent(opponent_model_path))

    obs, info = env.reset(seed=start_seed)
    env._episode_opponent_source = "model"
    env._episode_opponent_style = _model_stem(opponent_model_path)

    episode_idx = len(list(save_dir.glob("demo_*.npz")))
    obs_buf = []
    act_buf = []
    info_buf = []

    print("Recording demos...")
    print("Controls: WASD move, mouse aim, left click shoot, SPACE blink, Q/right click shield")
    print("Press R to reset and save current episode. Press ESC to quit.")

    running = True
    while running:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    if act_buf:
                        save_demo_episode(save_dir, episode_idx, obs_buf, act_buf, info_buf)
                        print(f"saved demo_{episode_idx:04d}.npz ({len(act_buf)} steps)")
                        episode_idx += 1
                    obs, info = env.reset(seed=start_seed + episode_idx)
                    env._episode_opponent_source = "model"
                    env._episode_opponent_style = _model_stem(opponent_model_path)
                    obs_buf, act_buf, info_buf = [], [], []

        if not running:
            break

        action = human_action_from_inputs(env)
        obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
        act_buf.append(np.asarray(action, dtype=np.int64).copy())

        obs, reward, terminated, truncated, info = env.step(action)
        info_buf.append(dict(info))

        if terminated or truncated:
            save_demo_episode(save_dir, episode_idx, obs_buf, act_buf, info_buf)
            print(
                f"saved demo_{episode_idx:04d}.npz ({len(act_buf)} steps) "
                f"result={info['result']} player_hp={info['player_hp']:.1f} enemy_hp={info['enemy_hp']:.1f}"
            )
            episode_idx += 1
            obs, info = env.reset(seed=start_seed + episode_idx)
            env._episode_opponent_source = "model"
            env._episode_opponent_style = _model_stem(opponent_model_path)
            obs_buf, act_buf, info_buf = [], [], []

    if act_buf:
        save_demo_episode(save_dir, episode_idx, obs_buf, act_buf, info_buf)
        print(f"saved demo_{episode_idx:04d}.npz ({len(act_buf)} steps)")

    env.close()


def save_demo_episode(save_dir: Path, episode_idx: int, obs_buf, act_buf, info_buf) -> None:
    np.savez_compressed(
        save_dir / f"demo_{episode_idx:04d}.npz",
        obs=np.asarray(obs_buf, dtype=np.float32),
        acts=np.asarray(act_buf, dtype=np.int64),
        infos=np.array(info_buf, dtype=object),
    )


def load_demo_dataset(demo_dir: str):
    demo_paths = sorted(Path(demo_dir).glob("demo_*.npz"))
    if not demo_paths:
        raise FileNotFoundError(f"No demos found in {demo_dir}")

    obs_all = []
    acts_all = []
    for path in demo_paths:
        data = np.load(path, allow_pickle=True)
        obs_all.append(np.asarray(data["obs"], dtype=np.float32))
        acts_all.append(np.asarray(data["acts"], dtype=np.int64))

    obs = np.concatenate(obs_all, axis=0)
    acts = np.concatenate(acts_all, axis=0)
    return obs, acts


def bc_finetune_from_demos(
    base_model_path: str,
    demo_dir: str = "demos/human",
    output_model_name: str = "arena_demo_finetuned",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 80,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for BC fine-tuning.") from exc

    obs, acts = load_demo_dataset(demo_dir)
    model, vec_norm = load_model_bundle(base_model_path)
    policy = model.policy
    device = policy.device

    if acts.shape[1] != 6:
        raise ValueError(f"Expected action dimension 6, got {acts.shape[1]}")

    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    acts_t = torch.as_tensor(acts, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    n = obs_t.shape[0]

    policy.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        total_samples = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_obs = obs_t[idx]
            batch_acts = acts_t[idx]

            dist = policy.get_distribution(batch_obs)
            if not hasattr(dist.distribution, "distribution"):
                categorical_list = dist.distribution
            else:
                categorical_list = dist.distribution.distribution

            loss = 0.0
            for head_idx, categorical in enumerate(categorical_list):
                loss = loss + F.cross_entropy(categorical.logits, batch_acts[:, head_idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_obs.shape[0])
            total_samples += int(batch_obs.shape[0])

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            mean_loss = total_loss / max(1, total_samples)
            print(f"[bc] epoch={epoch:03d} mean_loss={mean_loss:.6f}")

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = out_dir / f"{output_model_name}.zip"
    model.save(str(out_model_path))

    src_vecnorm = _vecnorm_path_for(base_model_path)
    dst_vecnorm = out_dir / f"{output_model_name}_vecnormalize.pkl"
    if src_vecnorm.exists():
        shutil.copy2(src_vecnorm, dst_vecnorm)
        print(f"copied vecnormalize -> {dst_vecnorm}")

    print(f"saved BC-finetuned model -> {out_model_path}")
    return str(out_model_path)


def continue_selfplay(
    base_model_path: str,
    total_timesteps: int = 400_000,
    n_envs: int = 8,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "arena_after_demos_selfplay",
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for self-play fine-tuning.") from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    def _factory():
        return Monitor(ArenaShooterEnv(render_mode=None, scripted_difficulty=0.5))

    base_vec_env = DummyVecEnv([_factory for _ in range(n_envs)])
    base_vec_env = VecMonitor(base_vec_env)

    src_vecnorm = _vecnorm_path_for(base_model_path)
    if src_vecnorm.exists():
        vec_env = VecNormalize.load(str(src_vecnorm), base_vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO.load(base_model_path, env=vec_env, device="auto")

    self_play_manager = SelfPlayManager(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        max_snapshots=8,
        snapshot_interval=100_000,
        deterministic=True,
    )

    callbacks = CallbackList(
        [
            EpisodeStatsCallback.build(verbose=1),
            SelfPlayCallback.build(self_play_manager, total_timesteps=total_timesteps, verbose=1),
            CheckpointCallback(
                save_freq=max(50_000 // max(1, n_envs), 1),
                save_path=str(checkpoint_path / "intermediate_selfplay"),
                name_prefix=model_name,
            ),
        ]
    )

    # SB3 recommends reset_num_timesteps=True when continuing training with a newly created env.
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=True)

    out_model_path = checkpoint_path / f"{model_name}.zip"
    out_vecnorm_path = checkpoint_path / f"{model_name}_vecnormalize.pkl"
    model.save(str(out_model_path))
    vec_env.save(str(out_vecnorm_path))
    print(f"saved self-play model -> {out_model_path}")
    print(f"saved self-play vecnormalize -> {out_vecnorm_path}")
    return str(out_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged training workflow for arena_shooter_rl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_basic = sub.add_parser("basic", help="Train a basic bot against scripted opponents")
    p_basic.add_argument("--steps", type=int, default=600_000)
    p_basic.add_argument("--envs", type=int, default=8)
    p_basic.add_argument("--name", type=str, default="arena_basic_bot")
    p_basic.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_basic.add_argument("--start-difficulty", type=float, default=0.20)
    p_basic.add_argument("--end-difficulty", type=float, default=0.95)

    p_record = sub.add_parser("record", help="Record human demos against a trained bot")
    p_record.add_argument("--model", type=str, required=True)
    p_record.add_argument("--out-dir", type=str, default="demos/human")
    p_record.add_argument("--seed", type=int, default=42)

    p_bc = sub.add_parser("bc", help="Behavior-cloning fine-tune a PPO policy from saved demos")
    p_bc.add_argument("--base-model", type=str, required=True)
    p_bc.add_argument("--demo-dir", type=str, default="demos/human")
    p_bc.add_argument("--name", type=str, default="arena_demo_finetuned")
    p_bc.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_bc.add_argument("--epochs", type=int, default=80)
    p_bc.add_argument("--batch-size", type=int, default=512)
    p_bc.add_argument("--lr", type=float, default=1e-4)

    p_self = sub.add_parser("selfplay", help="Continue training from a model using pure self-play")
    p_self.add_argument("--base-model", type=str, required=True)
    p_self.add_argument("--steps", type=int, default=400_000)
    p_self.add_argument("--envs", type=int, default=8)
    p_self.add_argument("--name", type=str, default="arena_after_demos_selfplay")
    p_self.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.cmd == "basic":
        train_basic_bot(
            total_timesteps=args.steps,
            n_envs=args.envs,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.name,
            start_difficulty=args.start_difficulty,
            end_difficulty=args.end_difficulty,
        )
    elif args.cmd == "record":
        record_human_demos(
            opponent_model_path=args.model,
            out_dir=args.out_dir,
            start_seed=args.seed,
        )
    elif args.cmd == "bc":
        bc_finetune_from_demos(
            base_model_path=args.base_model,
            demo_dir=args.demo_dir,
            output_model_name=args.name,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.cmd == "selfplay":
        continue_selfplay(
            base_model_path=args.base_model,
            total_timesteps=args.steps,
            n_envs=args.envs,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.name,
        )
    elif args.cmd == "eval":
        evaluate_policy_rollout(
            model_path=args.model,
            episodes=args.episodes,
            render=args.render,
        )


if __name__ == "__main__":
    main()
