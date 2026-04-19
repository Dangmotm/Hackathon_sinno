from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fighting_env import LocalDuelEnv


@dataclass
class Controls:
    left: int
    right: int
    jump: int
    attack: int
    block: int


PLAYER_CONTROLS = Controls(
    left=97,   # pygame.K_a
    right=100, # pygame.K_d
    jump=119,  # pygame.K_w
    attack=106, # pygame.K_j
    block=107,  # pygame.K_k
)


OBS_P1_SLICE = slice(0, 9)
OBS_P2_SLICE = slice(9, 18)
REL_X_IDX = 18
REL_Y_IDX = 19
DIST_IDX = 20
TIME_IDX = 21


def mirror_observation_for_player2(obs: np.ndarray) -> np.ndarray:
    """Convert a player_1-centric observation into a player_2-centric one.

    The trained PPO policy learns as player_1. To reuse it for player_2, we:
    1) swap player blocks,
    2) mirror horizontal quantities,
    3) keep distance and time unchanged.
    """
    mirrored = np.array(obs, dtype=np.float32, copy=True)
    p1 = np.array(obs[OBS_P1_SLICE], dtype=np.float32, copy=True)
    p2 = np.array(obs[OBS_P2_SLICE], dtype=np.float32, copy=True)

    # Swap player descriptors.
    mirrored[OBS_P1_SLICE] = p2
    mirrored[OBS_P2_SLICE] = p1

    # Mirror horizontal quantities for the new local perspective.
    mirrored[0] *= -1.0   # self x
    mirrored[2] *= -1.0   # self vx
    mirrored[8] *= -1.0   # self facing

    mirrored[9] *= -1.0   # opp x
    mirrored[11] *= -1.0  # opp vx
    mirrored[17] *= -1.0  # opp facing

    mirrored[REL_X_IDX] *= -1.0
    # REL_Y, distance, time are unchanged.
    return mirrored


def remap_action_from_player2_perspective(action: np.ndarray) -> np.ndarray:
    """Convert action chosen in mirrored coordinates back to world coordinates."""
    real_action = np.array(action, dtype=np.int64, copy=True)
    move = int(real_action[0])
    if move == 0:      # left in mirrored world -> right in real world
        real_action[0] = 2
    elif move == 2:    # right in mirrored world -> left in real world
        real_action[0] = 0
    return real_action


def build_model_opponent(model_path: str):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to load a trained model.") from exc

    model = PPO.load(model_path)

    def _policy(env: LocalDuelEnv, obs: np.ndarray) -> np.ndarray:
        mirrored_obs = mirror_observation_for_player2(obs)
        action, _ = model.predict(mirrored_obs, deterministic=True)
        return remap_action_from_player2_perspective(np.asarray(action, dtype=np.int64))

    return _policy


def keyboard_action() -> np.ndarray:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for local play.") from exc

    keys = pygame.key.get_pressed()

    move = 1
    if keys[PLAYER_CONTROLS.left] and not keys[PLAYER_CONTROLS.right]:
        move = 0
    elif keys[PLAYER_CONTROLS.right] and not keys[PLAYER_CONTROLS.left]:
        move = 2

    jump = 1 if keys[PLAYER_CONTROLS.jump] else 0
    attack = 1 if keys[PLAYER_CONTROLS.attack] else 0
    block = 1 if keys[PLAYER_CONTROLS.block] else 0
    return np.array([move, jump, attack, block], dtype=np.int64)


def run_local_match(model_path: str, seed: int = 42) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for local play.") from exc

    env = LocalDuelEnv(render_mode="human")
    env.set_opponent_policy(build_model_opponent(model_path))

    obs, info = env.reset(seed=seed)
    print("Controls: A/D move, W jump, J attack, K block, R reset round, ESC quit")

    running = True
    round_idx = 1

    while running:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    round_idx += 1
                    obs, info = env.reset(seed=seed + round_idx)
                    print(f"[round {round_idx}] manual reset")

        if not running:
            break

        action = keyboard_action()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                f"[round {round_idx}] result={info['result']} "
                f"player_hp={info['player_hp']:.1f} enemy_hp={info['enemy_hp']:.1f}"
            )
            round_idx += 1
            obs, info = env.reset(seed=seed + round_idx)

    env.close()


if __name__ == "__main__":
    DEFAULT_MODEL_PATH = "checkpoints/ppo_local_duel.zip"
    run_local_match(DEFAULT_MODEL_PATH)
