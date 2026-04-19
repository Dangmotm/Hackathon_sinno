from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


Action = np.ndarray
OpponentPolicy = Callable[["LocalDuelEnv", np.ndarray], Action]


@dataclass
class EnvConfig:
    arena_width: float = 12.0
    max_hp: float = 100.0
    fps: int = 30
    max_steps: int = 900  # 30 seconds at 30 FPS

    # Movement / physics
    move_speed: float = 0.28
    block_move_speed_scale: float = 0.45
    jump_speed: float = 0.82
    gravity: float = 0.06
    max_fall_speed: float = 1.25

    # Combat
    attack_range: float = 1.35
    attack_vertical_tolerance: float = 0.85
    attack_damage: float = 12.0
    attack_startup: int = 2
    attack_active: int = 2
    attack_recovery: int = 8
    block_damage_scale: float = 0.2

    # Reward shaping
    reward_win: float = 3.0
    reward_loss: float = -3.0
    reward_damage_scale: float = 0.05
    reward_step_penalty: float = -0.001
    reward_close_distance: float = 0.002
    reward_missed_attack: float = -0.01
    reward_timeout_hp_scale: float = 0.75

    @property
    def attack_total_lock(self) -> int:
        return self.attack_startup + self.attack_active + self.attack_recovery


@dataclass
class FighterState:
    x: float
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hp: float = 100.0
    grounded: bool = True
    facing: int = 1  # +1 -> facing right, -1 -> facing left
    blocking: bool = False

    startup_timer: int = 0
    active_timer: int = 0
    recovery_timer: int = 0
    has_hit_during_current_attack: bool = False

    def attack_lock_remaining(self) -> int:
        return self.startup_timer + self.active_timer + self.recovery_timer

    def is_attacking(self) -> bool:
        return self.startup_timer > 0 or self.active_timer > 0 or self.recovery_timer > 0

    def can_start_attack(self) -> bool:
        return self.attack_lock_remaining() == 0

    def begin_attack(self, cfg: EnvConfig) -> None:
        self.startup_timer = cfg.attack_startup
        self.active_timer = 0
        self.recovery_timer = 0
        self.has_hit_during_current_attack = False
        self.blocking = False

    def tick_attack_timers(self, cfg: EnvConfig) -> None:
        if self.startup_timer > 0:
            self.startup_timer -= 1
            if self.startup_timer == 0:
                self.active_timer = cfg.attack_active
            return

        if self.active_timer > 0:
            self.active_timer -= 1
            if self.active_timer == 0:
                self.recovery_timer = cfg.attack_recovery
            return

        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.has_hit_during_current_attack = False


class LocalDuelEnv(gym.Env):
    """A minimal 2D local fighting environment for RL.

    Observation:
        22-dim float32 vector in [-1, 1] / [0, 1], describing both fighters,
        relative geometry, and time remaining.

    Action (MultiDiscrete[3, 2, 2, 2]):
        [move, jump, attack, block]
        - move: 0=left, 1=idle, 2=right
        - jump: 0=no,   1=yes
        - attack: 0=no, 1=yes
        - block: 0=no,  1=yes

    Notes:
        - The learning agent always controls player_1.
        - player_2 is controlled by a scripted policy by default, or by a custom
          opponent policy that can be injected later.
        - Physics are intentionally simple to keep training stable and fast.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None,
        opponent_policy: Optional[OpponentPolicy] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = config or EnvConfig()
        self.metadata["render_fps"] = self.cfg.fps

        # Observation features are normalized into [-1, 1] or [0, 1].
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(22,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.array([3, 2, 2, 2], dtype=np.int64))

        self.opponent_policy = opponent_policy or self._scripted_opponent_policy

        self.player_1 = FighterState(x=2.0, hp=self.cfg.max_hp, facing=1)
        self.player_2 = FighterState(x=self.cfg.arena_width - 2.0, hp=self.cfg.max_hp, facing=-1)
        self.step_count = 0

        self._window = None
        self._clock = None
        self._surface_size = (960, 360)

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)

        self.step_count = 0
        self.player_1 = FighterState(x=2.0, hp=self.cfg.max_hp, facing=1)
        self.player_2 = FighterState(x=self.cfg.arena_width - 2.0, hp=self.cfg.max_hp, facing=-1)

        # Small randomized spawn offset helps reduce overfitting to exact positions.
        spawn_jitter = 0.35
        self.player_1.x += float(self.np_random.uniform(-spawn_jitter, spawn_jitter))
        self.player_2.x += float(self.np_random.uniform(-spawn_jitter, spawn_jitter))
        self.player_1.x = float(np.clip(self.player_1.x, 1.0, self.cfg.arena_width / 2 - 1.0))
        self.player_2.x = float(np.clip(self.player_2.x, self.cfg.arena_width / 2 + 1.0, self.cfg.arena_width - 1.0))
        self._update_facing()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: Action):
        action = np.asarray(action, dtype=np.int64)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_obs = self._get_obs()
        prev_distance = abs(self.player_2.x - self.player_1.x)
        prev_p1_hp = self.player_1.hp
        prev_p2_hp = self.player_2.hp

        enemy_action = np.asarray(self.opponent_policy(self, prev_obs), dtype=np.int64)
        enemy_action = np.clip(enemy_action, self.action_space.nvec * 0, self.action_space.nvec - 1)

        self.step_count += 1

        self._apply_intention(self.player_1, self.player_2, action)
        self._apply_intention(self.player_2, self.player_1, enemy_action)

        self._integrate_physics(self.player_1)
        self._integrate_physics(self.player_2)
        self._update_facing()

        damage_to_p2 = self._resolve_hit(self.player_1, self.player_2)
        damage_to_p1 = self._resolve_hit(self.player_2, self.player_1)

        self.player_1.tick_attack_timers(self.cfg)
        self.player_2.tick_attack_timers(self.cfg)

        obs = self._get_obs()

        terminated = self.player_1.hp <= 0.0 or self.player_2.hp <= 0.0
        truncated = self.step_count >= self.cfg.max_steps

        reward = self._compute_reward(
            action=action,
            prev_distance=prev_distance,
            damage_dealt=damage_to_p2,
            damage_taken=damage_to_p1,
            terminated=terminated,
            truncated=truncated,
        )

        info = self._get_info(
            prev_p1_hp=prev_p1_hp,
            prev_p2_hp=prev_p2_hp,
            enemy_action=enemy_action,
            damage_to_p1=damage_to_p1,
            damage_to_p2=damage_to_p2,
        )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required for render_mode='human' or 'rgb_array'.") from exc

        width, height = self._surface_size
        scale_x = width / self.cfg.arena_width
        floor_y = height - 70

        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(self._surface_size)
            pygame.display.set_caption("Local Duel Env")

        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface(self._surface_size)
        canvas.fill((245, 245, 245))

        # Floor
        pygame.draw.line(canvas, (40, 40, 40), (0, floor_y), (width, floor_y), 3)

        # Health bars
        self._draw_health_bar(canvas, 30, 20, 360, 22, self.player_1.hp / self.cfg.max_hp, (70, 140, 255))
        self._draw_health_bar(canvas, width - 390, 20, 360, 22, self.player_2.hp / self.cfg.max_hp, (255, 110, 110))

        # Fighters
        self._draw_fighter(canvas, self.player_1, floor_y, scale_x, color=(70, 140, 255))
        self._draw_fighter(canvas, self.player_2, floor_y, scale_x, color=(255, 110, 110))

        # Timer text
        font = pygame.font.SysFont("consolas", 24)
        time_left = max(0.0, (self.cfg.max_steps - self.step_count) / self.cfg.fps)
        text = font.render(f"{time_left:05.1f}s", True, (20, 20, 20))
        canvas.blit(text, (width // 2 - text.get_width() // 2, 16))

        if self.render_mode == "human":
            assert self._window is not None
            self._window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            assert self._clock is not None
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self) -> None:
        if self._window is not None or self._clock is not None:
            try:
                import pygame

                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass
            finally:
                self._window = None
                self._clock = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def set_opponent_policy(self, opponent_policy: OpponentPolicy) -> None:
        self.opponent_policy = opponent_policy

    # ---------------------------------------------------------------------
    # Internal logic
    # ---------------------------------------------------------------------
    def _apply_intention(self, actor: FighterState, opponent: FighterState, action: Action) -> None:
        move_code, jump_code, attack_code, block_code = [int(x) for x in action]

        move_dir = (-1 if move_code == 0 else 0 if move_code == 1 else 1)
        wants_jump = jump_code == 1
        wants_attack = attack_code == 1
        wants_block = block_code == 1 and not wants_attack

        # Start attack if available.
        if wants_attack and actor.can_start_attack():
            actor.begin_attack(self.cfg)

        # Only allow blocking while idle and not airborne attacking.
        actor.blocking = bool(wants_block and actor.can_start_attack() and actor.grounded)

        speed = self.cfg.move_speed * (self.cfg.block_move_speed_scale if actor.blocking else 1.0)
        if actor.is_attacking() and actor.recovery_timer > 0:
            actor.vx = 0.0
        else:
            actor.vx = move_dir * speed

        if wants_jump and actor.grounded and not actor.blocking:
            actor.vy = self.cfg.jump_speed
            actor.grounded = False

        # Face opponent whenever there is clear horizontal separation.
        dx = opponent.x - actor.x
        if abs(dx) > 1e-6:
            actor.facing = 1 if dx > 0 else -1

    def _integrate_physics(self, fighter: FighterState) -> None:
        fighter.x += fighter.vx
        fighter.x = float(np.clip(fighter.x, 0.5, self.cfg.arena_width - 0.5))

        if not fighter.grounded:
            fighter.vy -= self.cfg.gravity
            fighter.vy = float(max(fighter.vy, -self.cfg.max_fall_speed))
            fighter.y += fighter.vy
            if fighter.y <= 0.0:
                fighter.y = 0.0
                fighter.vy = 0.0
                fighter.grounded = True
        else:
            fighter.y = 0.0
            fighter.vy = 0.0

    def _resolve_hit(self, attacker: FighterState, defender: FighterState) -> float:
        if attacker.active_timer <= 0 or attacker.has_hit_during_current_attack:
            return 0.0

        dx = defender.x - attacker.x
        dy = abs(defender.y - attacker.y)
        in_front = dx * attacker.facing >= 0.0
        in_range = abs(dx) <= self.cfg.attack_range and dy <= self.cfg.attack_vertical_tolerance

        if not (in_front and in_range):
            return 0.0

        damage = self.cfg.attack_damage
        if defender.blocking and defender.facing == -attacker.facing:
            damage *= self.cfg.block_damage_scale

        defender.hp = float(max(0.0, defender.hp - damage))
        attacker.has_hit_during_current_attack = True

        # Small pushback for readability.
        defender.x += attacker.facing * 0.12
        defender.x = float(np.clip(defender.x, 0.5, self.cfg.arena_width - 0.5))
        return float(damage)

    def _update_facing(self) -> None:
        dx = self.player_2.x - self.player_1.x
        if abs(dx) > 1e-6:
            self.player_1.facing = 1 if dx > 0 else -1
            self.player_2.facing = -self.player_1.facing

    def _compute_reward(
        self,
        action: Action,
        prev_distance: float,
        damage_dealt: float,
        damage_taken: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward = 0.0
        reward += self.cfg.reward_damage_scale * damage_dealt
        reward -= self.cfg.reward_damage_scale * damage_taken
        reward += self.cfg.reward_step_penalty

        current_distance = abs(self.player_2.x - self.player_1.x)
        if prev_distance > self.cfg.attack_range and current_distance < prev_distance:
            reward += self.cfg.reward_close_distance

        attack_pressed = int(action[2]) == 1
        if attack_pressed and damage_dealt <= 0.0:
            reward += self.cfg.reward_missed_attack

        if terminated:
            if self.player_2.hp <= 0.0 and self.player_1.hp > 0.0:
                reward += self.cfg.reward_win
            elif self.player_1.hp <= 0.0 and self.player_2.hp > 0.0:
                reward += self.cfg.reward_loss
            else:
                # draw
                reward += 0.0

        if truncated and not terminated:
            hp_delta = (self.player_1.hp - self.player_2.hp) / self.cfg.max_hp
            reward += self.cfg.reward_timeout_hp_scale * float(hp_delta)

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        p1 = self.player_1
        p2 = self.player_2
        rel_x = p2.x - p1.x
        rel_y = p2.y - p1.y
        distance = abs(rel_x)
        time_left = max(0.0, self.cfg.max_steps - self.step_count) / self.cfg.max_steps

        obs = np.array(
            [
                # player 1
                self._norm_x(p1.x),
                self._norm_y(p1.y),
                self._norm_vx(p1.vx),
                self._norm_vy(p1.vy),
                p1.hp / self.cfg.max_hp,
                float(p1.grounded),
                p1.attack_lock_remaining() / self.cfg.attack_total_lock,
                float(p1.blocking),
                float(p1.facing),
                # player 2
                self._norm_x(p2.x),
                self._norm_y(p2.y),
                self._norm_vx(p2.vx),
                self._norm_vy(p2.vy),
                p2.hp / self.cfg.max_hp,
                float(p2.grounded),
                p2.attack_lock_remaining() / self.cfg.attack_total_lock,
                float(p2.blocking),
                float(p2.facing),
                # relative geometry / clock
                np.clip(rel_x / self.cfg.arena_width, -1.0, 1.0),
                np.clip(rel_y / max(1.0, self.cfg.jump_speed * 2.0), -1.0, 1.0),
                np.clip(distance / self.cfg.arena_width, 0.0, 1.0),
                np.clip(time_left, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(
        self,
        prev_p1_hp: Optional[float] = None,
        prev_p2_hp: Optional[float] = None,
        enemy_action: Optional[np.ndarray] = None,
        damage_to_p1: float = 0.0,
        damage_to_p2: float = 0.0,
    ) -> dict[str, Any]:
        result = "ongoing"
        if self.player_1.hp <= 0.0 and self.player_2.hp <= 0.0:
            result = "draw"
        elif self.player_2.hp <= 0.0:
            result = "win"
        elif self.player_1.hp <= 0.0:
            result = "loss"
        elif self.step_count >= self.cfg.max_steps:
            result = "timeout"

        return {
            "step": self.step_count,
            "player_hp": float(self.player_1.hp),
            "enemy_hp": float(self.player_2.hp),
            "distance": float(abs(self.player_2.x - self.player_1.x)),
            "damage_to_player": float(damage_to_p1),
            "damage_to_enemy": float(damage_to_p2),
            "player_hp_delta": None if prev_p1_hp is None else float(self.player_1.hp - prev_p1_hp),
            "enemy_hp_delta": None if prev_p2_hp is None else float(self.player_2.hp - prev_p2_hp),
            "enemy_action": None if enemy_action is None else np.asarray(enemy_action, dtype=np.int64),
            "result": result,
        }

    # ---------------------------------------------------------------------
    # Default opponent
    # ---------------------------------------------------------------------
    def _scripted_opponent_policy(self, env: "LocalDuelEnv", obs: np.ndarray) -> Action:
        p1 = env.player_1
        p2 = env.player_2
        distance = abs(p1.x - p2.x)

        move = 1  # idle
        jump = 0
        attack = 0
        block = 0

        # If player is actively swinging and close, try to block.
        if p1.active_timer > 0 and distance <= env.cfg.attack_range * 1.15 and p2.can_start_attack():
            block = 1
        else:
            if distance > env.cfg.attack_range * 1.05:
                move = 2 if p1.x > p2.x else 0
            else:
                r = float(env.np_random.uniform())
                if p2.can_start_attack() and r < 0.55:
                    attack = 1
                elif r < 0.75:
                    move = 0 if p1.x > p2.x else 2  # small retreat
                else:
                    move = 1

        # Occasional jump if trapped close.
        if distance < 1.0 and p2.grounded and float(env.np_random.uniform()) < 0.05:
            jump = 1

        return np.array([move, jump, attack, block], dtype=np.int64)

    # ---------------------------------------------------------------------
    # Render helpers
    # ---------------------------------------------------------------------
    def _draw_health_bar(self, canvas, x: int, y: int, w: int, h: int, ratio: float, fill_color) -> None:
        try:
            import pygame
        except ImportError:
            return

        ratio = float(np.clip(ratio, 0.0, 1.0))
        pygame.draw.rect(canvas, (40, 40, 40), (x, y, w, h), width=2)
        pygame.draw.rect(canvas, (220, 220, 220), (x + 2, y + 2, w - 4, h - 4))
        pygame.draw.rect(canvas, fill_color, (x + 2, y + 2, int((w - 4) * ratio), h - 4))

    def _draw_fighter(self, canvas, fighter: FighterState, floor_y: int, scale_x: float, color) -> None:
        try:
            import pygame
        except ImportError:
            return

        cx = int(fighter.x * scale_x)
        foot_y = int(floor_y - fighter.y * 90)
        body_w = 36
        body_h = 62

        body_rect = pygame.Rect(cx - body_w // 2, foot_y - body_h, body_w, body_h)
        pygame.draw.rect(canvas, color, body_rect, border_radius=8)

        # Head
        pygame.draw.circle(canvas, color, (cx, foot_y - body_h - 16), 14)

        # Facing marker
        marker_dx = 18 * fighter.facing
        pygame.draw.line(
            canvas,
            (20, 20, 20),
            (cx, foot_y - body_h // 2),
            (cx + marker_dx, foot_y - body_h // 2),
            3,
        )

        # Attack range hint during active frames
        if fighter.active_timer > 0:
            hit_x = cx + int(self.cfg.attack_range * scale_x * fighter.facing)
            pygame.draw.line(
                canvas,
                (255, 180, 0),
                (cx, foot_y - body_h // 2),
                (hit_x, foot_y - body_h // 2),
                5,
            )

        # Shield effect while blocking
        if fighter.blocking:
            pygame.draw.circle(canvas, (80, 220, 160), (cx, foot_y - body_h // 2), 32, width=4)

    # ---------------------------------------------------------------------
    # Normalization helpers
    # ---------------------------------------------------------------------
    def _norm_x(self, x: float) -> float:
        return float(np.clip((2.0 * x / self.cfg.arena_width) - 1.0, -1.0, 1.0))

    def _norm_y(self, y: float) -> float:
        return float(np.clip(y / max(1.0, self.cfg.jump_speed * 2.0), 0.0, 1.0))

    def _norm_vx(self, vx: float) -> float:
        return float(np.clip(vx / self.cfg.move_speed, -1.0, 1.0))

    def _norm_vy(self, vy: float) -> float:
        return float(np.clip(vy / max(self.cfg.jump_speed, self.cfg.max_fall_speed), -1.0, 1.0))


def make_env(render_mode: Optional[str] = None, opponent_policy: Optional[OpponentPolicy] = None) -> LocalDuelEnv:
    return LocalDuelEnv(render_mode=render_mode, opponent_policy=opponent_policy)


class EpisodeStatsCallback:
    """Lazy wrapper around SB3 callback so env users can import this file without SB3 installed."""

    @staticmethod
    def build(verbose: int = 0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

        class _EpisodeStatsCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose=verbose)
                self.episode_count = 0
                self.last_mean_reward = 0.0

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                rewards = self.locals.get("rewards", [])
                if infos is None or dones is None:
                    return True

                for idx, done in enumerate(dones):
                    if not done:
                        continue
                    self.episode_count += 1
                    info = infos[idx] if idx < len(infos) else {}
                    reward = float(rewards[idx]) if rewards is not None and idx < len(rewards) else float("nan")
                    self.last_mean_reward = reward
                    if self.verbose > 0:
                        print(
                            f"[episode {self.episode_count}] result={info.get('result')} "
                            f"player_hp={info.get('player_hp'):.1f} enemy_hp={info.get('enemy_hp'):.1f} "
                            f"reward={reward:.3f}"
                        )
                return True

        return _EpisodeStatsCallback(verbose=verbose)


def build_vec_env(n_envs: int = 1):
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    def _factory():
        env = LocalDuelEnv(render_mode=None)
        return Monitor(env)

    env_fns = [_factory for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


def train_ppo(
    total_timesteps: int = 200_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "ppo_local_duel",
):
    """Train PPO on the local duel env.

    Returns:
        model, vec_env
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(n_envs=n_envs)
    callback = EpisodeStatsCallback.build(verbose=1)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(checkpoint_path / "tb_logs"),
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    save_path = checkpoint_path / f"{model_name}.zip"
    model.save(str(save_path))
    print(f"saved model -> {save_path}")
    return model, vec_env


def evaluate_policy_rollout(model_path: str, episodes: int = 3, render: bool = False) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    env = LocalDuelEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        print(
            f"[eval episode {ep}] result={info['result']} "
            f"player_hp={info['player_hp']:.1f} enemy_hp={info['enemy_hp']:.1f} total_reward={total_reward:.3f}"
        )

    env.close()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = LocalDuelEnv(render_mode=None)
    check_env(env, skip_render_check=True)

    obs, info = env.reset(seed=42)
    print("obs shape:", obs.shape)
    print("info:", info)

    total_reward = 0.0
    for _ in range(120):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print("episode result:", info["result"])
    print("total_reward:", round(total_reward, 3))
    env.close()
