from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Action = np.ndarray
OpponentPolicy = Callable[["ArenaShooterEnv", np.ndarray], Action]


@dataclass
class EnvConfig:
    arena_size: float = 12.0
    agent_radius: float = 0.35
    max_hp: float = 100.0
    fps: int = 20
    max_steps: int = 1200

    move_speed: float = 0.28
    aim_sectors: int = 16

    projectile_speed: float = 0.48
    projectile_radius: float = 0.10
    projectile_damage: float = 22.0
    projectile_lifetime_steps: int = 24
    shot_cooldown: int = 5

    blink_distance: float = 1.6
    blink_cooldown: int = 42

    shield_duration: int = 36
    shield_cooldown: int = 9999
    shield_radius_bonus: float = 0.18

    reward_win: float = 3.0
    reward_loss: float = -3.0
    reward_damage_scale: float = 0.05
    reward_no_damage_penalty: float = -0.35
    no_damage_penalty_interval: int = 15
    reward_timeout_hp_scale: float = 1.0
    reward_timeout_extra_no_damage: float = -0.8

    action_repeat: int = 1


@dataclass
class Obstacle:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.h


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    owner: str  # "player" or "enemy"
    ttl: int


@dataclass
class AgentState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hp: float = 100.0
    aim_dx: float = 1.0
    aim_dy: float = 0.0
    shot_cooldown_remaining: int = 0
    blink_cooldown_remaining: int = 0
    shield_cooldown_remaining: int = 0
    shield_active_timer: int = 0
    shield_charges: int = 1

    def shot_cd_ratio(self, cfg: EnvConfig) -> float:
        return 0.0 if cfg.shot_cooldown <= 0 else float(self.shot_cooldown_remaining / cfg.shot_cooldown)

    def blink_cd_ratio(self, cfg: EnvConfig) -> float:
        return 0.0 if cfg.blink_cooldown <= 0 else float(self.blink_cooldown_remaining / cfg.blink_cooldown)

    def shield_cd_ratio(self, cfg: EnvConfig) -> float:
        denom = max(1, min(cfg.shield_cooldown, 120))
        return float(min(self.shield_cooldown_remaining, denom) / denom)

    def shield_active(self) -> bool:
        return self.shield_active_timer > 0


class ArenaShooterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None,
        opponent_policy: Optional[OpponentPolicy] = None,
        scripted_difficulty: float = 0.5,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = config or EnvConfig()
        self.metadata["render_fps"] = self.cfg.fps

        # move_x, move_y, aim_sector, shoot, blink, shield
        self.action_space = spaces.MultiDiscrete(
            np.array([3, 3, self.cfg.aim_sectors + 1, 2, 2, 2], dtype=np.int64)
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(39,), dtype=np.float32)

        self.scripted_difficulty = float(np.clip(scripted_difficulty, 0.0, 1.0))
        self.opponent_policy = opponent_policy or self._default_opponent_policy

        self.obstacles = self._build_default_obstacles()
        self.projectiles: list[Projectile] = []

        self.player_1 = AgentState(x=2.0, y=2.0, hp=self.cfg.max_hp)
        self.player_2 = AgentState(x=self.cfg.arena_size - 2.0, y=self.cfg.arena_size - 2.0, hp=self.cfg.max_hp)
        self.step_count = 0
        self.steps_since_last_damage_dealt = 0

        self._episode_opponent_spec: Optional[dict[str, Any]] = None
        self._episode_opponent_source: str = "scripted"
        self._episode_opponent_style: str = "balanced"

        self._window = None
        self._clock = None
        self._surface_size = (720, 720)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.steps_since_last_damage_dealt = 0
        self.projectiles = []
        self._episode_opponent_spec = None
        self._episode_opponent_source = "scripted"
        self._episode_opponent_style = str(self.np_random.choice(np.array(["aggressive", "balanced", "kiter", "strafer"], dtype=object)))

        spawn_candidates = [
            (1.6, 1.6, self.cfg.arena_size - 1.6, self.cfg.arena_size - 1.6),
            (1.6, self.cfg.arena_size - 1.6, self.cfg.arena_size - 1.6, 1.6),
            (1.6, self.cfg.arena_size / 2, self.cfg.arena_size - 1.6, self.cfg.arena_size / 2),
        ]
        choice = int(self.np_random.integers(0, len(spawn_candidates)))
        p1x, p1y, p2x, p2y = spawn_candidates[choice]

        jitter = 0.35
        self.player_1 = AgentState(
            x=float(np.clip(p1x + self.np_random.uniform(-jitter, jitter), 0.8, self.cfg.arena_size - 0.8)),
            y=float(np.clip(p1y + self.np_random.uniform(-jitter, jitter), 0.8, self.cfg.arena_size - 0.8)),
            hp=self.cfg.max_hp,
            aim_dx=1.0,
            aim_dy=0.0,
            shield_charges=1,
        )
        self.player_2 = AgentState(
            x=float(np.clip(p2x + self.np_random.uniform(-jitter, jitter), 0.8, self.cfg.arena_size - 0.8)),
            y=float(np.clip(p2y + self.np_random.uniform(-jitter, jitter), 0.8, self.cfg.arena_size - 0.8)),
            hp=self.cfg.max_hp,
            aim_dx=-1.0,
            aim_dy=0.0,
            shield_charges=1,
        )

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: Action):
        action = np.asarray(action, dtype=np.int64)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_p1_hp = self.player_1.hp
        prev_p2_hp = self.player_2.hp
        obs_before = self._get_obs()

        enemy_action = np.asarray(self.opponent_policy(self, obs_before), dtype=np.int64)
        enemy_action = np.clip(enemy_action, np.zeros_like(self.action_space.nvec), self.action_space.nvec - 1)

        total_damage_to_p2 = 0.0
        total_damage_to_p1 = 0.0
        terminated = False
        truncated = False

        for _ in range(max(1, self.cfg.action_repeat)):
            self.step_count += 1

            self._apply_action_pre_movement(self.player_1, action)
            self._apply_action_pre_movement(self.player_2, enemy_action)

            self._integrate_position(self.player_1)
            self._integrate_position(self.player_2)

            self._apply_action_post_movement(self.player_1, action, owner="player")
            self._apply_action_post_movement(self.player_2, enemy_action, owner="enemy")

            dmg_p1, dmg_p2 = self._update_projectiles()
            total_damage_to_p1 += dmg_p1
            total_damage_to_p2 += dmg_p2

            if dmg_p2 > 0.0:
                self.steps_since_last_damage_dealt = 0
            else:
                self.steps_since_last_damage_dealt += 1

            self._tick_agent_timers(self.player_1)
            self._tick_agent_timers(self.player_2)

            terminated = self.player_1.hp <= 0.0 or self.player_2.hp <= 0.0
            truncated = self.step_count >= self.cfg.max_steps
            if terminated or truncated:
                break

        obs = self._get_obs()
        reward = self._compute_reward(
            damage_dealt=total_damage_to_p2,
            damage_taken=total_damage_to_p1,
            terminated=terminated,
            truncated=truncated,
        )
        info = self._get_info(
            prev_p1_hp=prev_p1_hp,
            prev_p2_hp=prev_p2_hp,
            enemy_action=enemy_action,
            damage_to_p1=total_damage_to_p1,
            damage_to_p2=total_damage_to_p2,
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
            raise RuntimeError("pygame is required for rendering.") from exc

        width, height = self._surface_size
        scale = (width - 80) / self.cfg.arena_size

        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(self._surface_size)
            pygame.display.set_caption("Arena Shooter RL")

        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface(self._surface_size)
        canvas.fill((245, 245, 245))
        arena_rect = pygame.Rect(40, 40, width - 80, height - 80)
        pygame.draw.rect(canvas, (232, 232, 232), arena_rect)
        pygame.draw.rect(canvas, (45, 45, 45), arena_rect, width=4)

        self._draw_obstacles(canvas, scale)
        self._draw_projectiles(canvas, scale)
        self._draw_agent(canvas, self.player_1, scale, (70, 140, 255))
        self._draw_agent(canvas, self.player_2, scale, (255, 110, 110))

        self._draw_health_bar(canvas, 30, 10, 280, 20, self.player_1.hp / self.cfg.max_hp, (70, 140, 255))
        self._draw_health_bar(canvas, width - 310, 10, 280, 20, self.player_2.hp / self.cfg.max_hp, (255, 110, 110))

        font = pygame.font.SysFont("consolas", 20)
        time_left = max(0.0, (self.cfg.max_steps - self.step_count) / self.cfg.fps)
        hud = f"{time_left:05.1f}s | {self._episode_opponent_source}:{self._episode_opponent_style}"
        text = font.render(hud, True, (20, 20, 20))
        canvas.blit(text, (width // 2 - text.get_width() // 2, 10))

        if self.render_mode == "human":
            assert self._window is not None
            self._window.blit(canvas, (0, 0))
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

    # ------------------------------------------------------------------
    # Environment logic
    # ------------------------------------------------------------------
    def set_opponent_policy(self, opponent_policy: OpponentPolicy) -> None:
        self.opponent_policy = opponent_policy

    def set_scripted_difficulty(self, difficulty: float) -> None:
        self.scripted_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def _build_default_obstacles(self) -> list[Obstacle]:
        return [
            Obstacle(4.7, 4.7, 2.6, 2.6),
            Obstacle(2.0, 7.4, 1.6, 1.4),
            Obstacle(8.4, 3.2, 1.6, 1.4),
            Obstacle(2.0, 2.0, 1.2, 1.8),
            Obstacle(8.8, 8.2, 1.2, 1.8),
        ]

    def _apply_action_pre_movement(self, agent: AgentState, action: Action) -> None:
        move_x_code, move_y_code, aim_sector_code, _shoot, blink_code, shield_code = [int(v) for v in action]

        move_x = -1 if move_x_code == 0 else 0 if move_x_code == 1 else 1
        move_y = -1 if move_y_code == 0 else 0 if move_y_code == 1 else 1
        move_vec = np.array([move_x, move_y], dtype=np.float32)
        norm = float(np.linalg.norm(move_vec))
        if norm > 1e-6:
            move_vec /= norm
        agent.vx = float(move_vec[0] * self.cfg.move_speed)
        agent.vy = float(move_vec[1] * self.cfg.move_speed)

        if aim_sector_code < self.cfg.aim_sectors:
            agent.aim_dx, agent.aim_dy = self._sector_to_unit_vector(aim_sector_code)

        if blink_code == 1 and agent.blink_cooldown_remaining <= 0:
            blink_dir = np.array([agent.vx, agent.vy], dtype=np.float32)
            if float(np.linalg.norm(blink_dir)) < 1e-6:
                blink_dir = np.array([agent.aim_dx, agent.aim_dy], dtype=np.float32)
            blink_norm = float(np.linalg.norm(blink_dir))
            if blink_norm > 1e-6:
                blink_dir /= blink_norm
                dx = float(blink_dir[0] * self.cfg.blink_distance)
                dy = float(blink_dir[1] * self.cfg.blink_distance)
                self._move_circle(agent, dx, dy)
                agent.blink_cooldown_remaining = self.cfg.blink_cooldown

        if shield_code == 1 and agent.shield_charges > 0 and not agent.shield_active() and agent.shield_cooldown_remaining <= 0:
            agent.shield_active_timer = self.cfg.shield_duration
            agent.shield_cooldown_remaining = self.cfg.shield_cooldown
            agent.shield_charges -= 1

    def _apply_action_post_movement(self, agent: AgentState, action: Action, owner: str) -> None:
        _move_x, _move_y, _aim_sector, shoot_code, _blink, _shield = [int(v) for v in action]
        if shoot_code == 1 and agent.shot_cooldown_remaining <= 0:
            self._spawn_projectile(agent, owner)
            agent.shot_cooldown_remaining = self.cfg.shot_cooldown

    def _integrate_position(self, agent: AgentState) -> None:
        self._move_circle(agent, agent.vx, agent.vy)

    def _move_circle(self, agent: AgentState, dx: float, dy: float) -> None:
        dist = max(abs(dx), abs(dy))
        steps = max(1, int(np.ceil(dist / 0.06)))
        step_dx = dx / steps
        step_dy = dy / steps
        for _ in range(steps):
            nx = agent.x + step_dx
            if not self._circle_collides(nx, agent.y, self.cfg.agent_radius):
                agent.x = nx
            ny = agent.y + step_dy
            if not self._circle_collides(agent.x, ny, self.cfg.agent_radius):
                agent.y = ny

    def _spawn_projectile(self, agent: AgentState, owner: str) -> None:
        ox = agent.x + agent.aim_dx * (self.cfg.agent_radius + self.cfg.projectile_radius + 0.05)
        oy = agent.y + agent.aim_dy * (self.cfg.agent_radius + self.cfg.projectile_radius + 0.05)
        if self._circle_collides(ox, oy, self.cfg.projectile_radius):
            ox, oy = agent.x, agent.y
        self.projectiles.append(
            Projectile(
                x=float(ox),
                y=float(oy),
                vx=float(agent.aim_dx * self.cfg.projectile_speed),
                vy=float(agent.aim_dy * self.cfg.projectile_speed),
                owner=owner,
                ttl=self.cfg.projectile_lifetime_steps,
            )
        )

    def _update_projectiles(self) -> tuple[float, float]:
        damage_to_p1 = 0.0
        damage_to_p2 = 0.0
        survivors: list[Projectile] = []

        for proj in self.projectiles:
            alive = True
            substeps = max(1, int(np.ceil(max(abs(proj.vx), abs(proj.vy)) / 0.05)))
            step_vx = proj.vx / substeps
            step_vy = proj.vy / substeps

            for _ in range(substeps):
                nx = proj.x + step_vx
                ny = proj.y + step_vy
                if self._circle_collides(nx, ny, self.cfg.projectile_radius):
                    alive = False
                    break
                proj.x = nx
                proj.y = ny

                target = self.player_2 if proj.owner == "player" else self.player_1
                if target.shield_active() and self._distance_xy(proj.x, proj.y, target.x, target.y) <= self._shield_radius():
                    alive = False
                    break

                if self._distance_xy(proj.x, proj.y, target.x, target.y) <= self.cfg.agent_radius + self.cfg.projectile_radius:
                    if proj.owner == "player":
                        self.player_2.hp = float(max(0.0, self.player_2.hp - self.cfg.projectile_damage))
                        damage_to_p2 += self.cfg.projectile_damage
                    else:
                        self.player_1.hp = float(max(0.0, self.player_1.hp - self.cfg.projectile_damage))
                        damage_to_p1 += self.cfg.projectile_damage
                    alive = False
                    break

            proj.ttl -= 1
            if alive and proj.ttl > 0:
                survivors.append(proj)

        self.projectiles = survivors
        return float(damage_to_p1), float(damage_to_p2)

    def _tick_agent_timers(self, agent: AgentState) -> None:
        if agent.shot_cooldown_remaining > 0:
            agent.shot_cooldown_remaining -= 1
        if agent.blink_cooldown_remaining > 0:
            agent.blink_cooldown_remaining -= 1
        if agent.shield_cooldown_remaining > 0:
            agent.shield_cooldown_remaining -= 1
        if agent.shield_active_timer > 0:
            agent.shield_active_timer -= 1

    def _compute_reward(self, damage_dealt: float, damage_taken: float, terminated: bool, truncated: bool) -> float:
        reward = 0.0
        reward += self.cfg.reward_damage_scale * damage_dealt
        reward -= self.cfg.reward_damage_scale * damage_taken

        if self.steps_since_last_damage_dealt > 0 and self.cfg.no_damage_penalty_interval > 0:
            streak_bucket = self.steps_since_last_damage_dealt // self.cfg.no_damage_penalty_interval
            if streak_bucket > 0 and self.steps_since_last_damage_dealt % self.cfg.no_damage_penalty_interval == 0:
                reward += self.cfg.reward_no_damage_penalty * min(4.0, float(streak_bucket))

        if terminated:
            if self.player_2.hp <= 0.0 and self.player_1.hp > 0.0:
                reward += self.cfg.reward_win
            elif self.player_1.hp <= 0.0 and self.player_2.hp > 0.0:
                reward += self.cfg.reward_loss

        if truncated and not terminated:
            hp_delta = (self.player_1.hp - self.player_2.hp) / self.cfg.max_hp
            reward += self.cfg.reward_timeout_hp_scale * float(hp_delta)
            if self.steps_since_last_damage_dealt >= self.cfg.no_damage_penalty_interval:
                reward += self.cfg.reward_timeout_extra_no_damage

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        p1 = self.player_1
        p2 = self.player_2
        rel_x = p2.x - p1.x
        rel_y = p2.y - p1.y
        dist = float(np.hypot(rel_x, rel_y))
        max_dist = float(np.hypot(self.cfg.arena_size, self.cfg.arena_size))
        time_left = max(0.0, self.cfg.max_steps - self.step_count) / self.cfg.max_steps
        los = 0.0 if self._line_blocked(p1.x, p1.y, p2.x, p2.y) else 1.0
        hostile_to_p1 = self._nearest_projectile_features(target=self.player_1, hostile_owner="enemy")
        hostile_to_p2 = self._nearest_projectile_features(target=self.player_2, hostile_owner="player")

        return np.array(
            [
                self._norm_pos(p1.x), self._norm_pos(p1.y),
                self._norm_vel(p1.vx), self._norm_vel(p1.vy),
                p1.hp / self.cfg.max_hp,
                p1.aim_dx, p1.aim_dy,
                p1.shot_cd_ratio(self.cfg),
                p1.blink_cd_ratio(self.cfg),
                p1.shield_cd_ratio(self.cfg),
                float(p1.shield_active()),
                float(p1.shield_charges > 0),

                self._norm_pos(p2.x), self._norm_pos(p2.y),
                self._norm_vel(p2.vx), self._norm_vel(p2.vy),
                p2.hp / self.cfg.max_hp,
                p2.aim_dx, p2.aim_dy,
                p2.shot_cd_ratio(self.cfg),
                p2.blink_cd_ratio(self.cfg),
                p2.shield_cd_ratio(self.cfg),
                float(p2.shield_active()),
                float(p2.shield_charges > 0),

                *hostile_to_p1,
                *hostile_to_p2,

                np.clip(rel_x / self.cfg.arena_size, -1.0, 1.0),
                np.clip(rel_y / self.cfg.arena_size, -1.0, 1.0),
                np.clip(dist / max_dist, 0.0, 1.0),
                np.clip(time_left, 0.0, 1.0),
                los,
            ],
            dtype=np.float32,
        )

    def _nearest_projectile_features(self, target: AgentState, hostile_owner: str) -> list[float]:
        best_proj = None
        best_dist = float("inf")
        for proj in self.projectiles:
            if proj.owner != hostile_owner:
                continue
            d = self._distance_xy(proj.x, proj.y, target.x, target.y)
            if d < best_dist:
                best_dist = d
                best_proj = proj
        if best_proj is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            float(np.clip((best_proj.x - target.x) / self.cfg.arena_size, -1.0, 1.0)),
            float(np.clip((best_proj.y - target.y) / self.cfg.arena_size, -1.0, 1.0)),
            self._norm_vel(best_proj.vx),
            self._norm_vel(best_proj.vy),
            1.0,
        ]

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
            "damage_to_player": float(damage_to_p1),
            "damage_to_enemy": float(damage_to_p2),
            "player_hp_delta": None if prev_p1_hp is None else float(self.player_1.hp - prev_p1_hp),
            "enemy_hp_delta": None if prev_p2_hp is None else float(self.player_2.hp - prev_p2_hp),
            "enemy_action": None if enemy_action is None else np.asarray(enemy_action, dtype=np.int64),
            "steps_since_last_damage_dealt": int(self.steps_since_last_damage_dealt),
            "scripted_difficulty": float(self.scripted_difficulty),
            "opponent_source": self._episode_opponent_source,
            "opponent_style": self._episode_opponent_style,
            "active_projectiles": len(self.projectiles),
            "result": result,
        }

    # ------------------------------------------------------------------
    # Collision / geometry
    # ------------------------------------------------------------------
    def _circle_collides(self, x: float, y: float, radius: float) -> bool:
        if x - radius < 0.0 or x + radius > self.cfg.arena_size or y - radius < 0.0 or y + radius > self.cfg.arena_size:
            return True
        for obs in self.obstacles:
            cx = float(np.clip(x, obs.left, obs.right))
            cy = float(np.clip(y, obs.bottom, obs.top))
            if self._distance_xy(x, y, cx, cy) <= radius:
                return True
        return False

    def _line_blocked(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        dist = self._distance_xy(x1, y1, x2, y2)
        steps = max(2, int(np.ceil(dist / 0.12)))
        for i in range(1, steps):
            t = i / steps
            px = x1 + (x2 - x1) * t
            py = y1 + (y2 - y1) * t
            if self._circle_collides(px, py, self.cfg.projectile_radius):
                return True
        return False

    def _distance_xy(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return float(np.hypot(x2 - x1, y2 - y1))

    def _shield_radius(self) -> float:
        return self.cfg.agent_radius + self.cfg.shield_radius_bonus

    # ------------------------------------------------------------------
    # Scripted policies
    # ------------------------------------------------------------------
    def _default_opponent_policy(self, env: "ArenaShooterEnv", obs: np.ndarray) -> Action:
        del obs
        return env._scripted_opponent_action(style=env._episode_opponent_style, difficulty=env.scripted_difficulty)

    def _scripted_opponent_action(self, style: str = "balanced", difficulty: Optional[float] = None) -> Action:
        difficulty = self.scripted_difficulty if difficulty is None else float(np.clip(difficulty, 0.0, 1.0))
        p1 = self.player_1
        p2 = self.player_2
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dist = float(np.hypot(dx, dy))
        los = not self._line_blocked(p2.x, p2.y, p1.x, p1.y)

        if style == "aggressive":
            desired = np.array([dx, dy], dtype=np.float32)
            shoot_bias = 0.78
        elif style == "kiter":
            if dist < 3.6:
                desired = np.array([-dx, -dy], dtype=np.float32)
            elif los:
                desired = np.array([-dy, dx], dtype=np.float32)
            else:
                desired = np.array([dx, dy], dtype=np.float32)
            shoot_bias = 0.48
        elif style == "strafer":
            desired = np.array([-dy, dx], dtype=np.float32) if los else np.array([dx, dy], dtype=np.float32)
            shoot_bias = 0.60
        else:
            if not los or dist > self.cfg.projectile_lifetime_steps * self.cfg.projectile_speed * 0.7:
                desired = np.array([dx, dy], dtype=np.float32)
            elif dist < 2.0:
                desired = np.array([-dx, -dy], dtype=np.float32)
            else:
                desired = np.array([-dy, dx], dtype=np.float32)
            shoot_bias = 0.56

        if float(self.np_random.uniform()) < max(0.0, 0.40 - 0.30 * difficulty):
            desired *= 0.0

        move_x = 1
        move_y = 1
        if desired[0] > 0.25:
            move_x = 2
        elif desired[0] < -0.25:
            move_x = 0
        if desired[1] > 0.25:
            move_y = 2
        elif desired[1] < -0.25:
            move_y = 0

        aim_sector = self._vector_to_sector(dx, dy)
        sector_jitter = int(round((1.0 - difficulty) * 2.0))
        if sector_jitter > 0 and aim_sector < self.cfg.aim_sectors:
            aim_sector = int((aim_sector + int(self.np_random.integers(-sector_jitter, sector_jitter + 1))) % self.cfg.aim_sectors)

        blink = 0
        shield = 0
        incoming = self._nearest_hostile_projectile(target=p2, hostile_owner="player")
        if incoming is not None:
            proj, proj_dist = incoming
            if proj_dist < 1.5:
                if p2.shield_charges > 0 and not p2.shield_active() and p2.shield_cooldown_remaining <= 0:
                    shield = 1
                elif p2.blink_cooldown_remaining <= 0:
                    blink = 1
                    move_x = 0 if proj.vx > 0 else 2 if proj.vx < 0 else move_x
                    move_y = 0 if proj.vy > 0 else 2 if proj.vy < 0 else move_y

        alignment = self._aim_alignment_after_sector(p2, p1, aim_sector)
        shoot_prob = min(0.95, shoot_bias + 0.25 * difficulty)
        shoot = 1 if (los and dist <= self.cfg.projectile_lifetime_steps * self.cfg.projectile_speed and alignment > 0.92 and float(self.np_random.uniform()) < shoot_prob) else 0
        return np.array([move_x, move_y, aim_sector, shoot, blink, shield], dtype=np.int64)

    def _nearest_hostile_projectile(self, target: AgentState, hostile_owner: str) -> Optional[tuple[Projectile, float]]:
        best_proj = None
        best_dist = float("inf")
        for proj in self.projectiles:
            if proj.owner != hostile_owner:
                continue
            d = self._distance_xy(proj.x, proj.y, target.x, target.y)
            if d < best_dist:
                best_dist = d
                best_proj = proj
        if best_proj is None:
            return None
        return best_proj, float(best_dist)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_obstacles(self, canvas, scale: float) -> None:
        import pygame

        for obs in self.obstacles:
            left, top = self._world_to_screen(obs.left, obs.top, scale)
            right, bottom = self._world_to_screen(obs.right, obs.bottom, scale)
            rect = pygame.Rect(left, top, right - left, bottom - top)
            pygame.draw.rect(canvas, (120, 120, 120), rect, border_radius=6)
            pygame.draw.rect(canvas, (70, 70, 70), rect, width=2, border_radius=6)

    def _draw_projectiles(self, canvas, scale: float) -> None:
        import pygame

        for proj in self.projectiles:
            x, y = self._world_to_screen(proj.x, proj.y, scale)
            radius = max(2, int(self.cfg.projectile_radius * scale))
            color = (90, 180, 255) if proj.owner == "player" else (255, 140, 140)
            pygame.draw.circle(canvas, color, (x, y), radius)
            pygame.draw.circle(canvas, (30, 30, 30), (x, y), radius, width=1)

    def _draw_health_bar(self, canvas, x: int, y: int, w: int, h: int, ratio: float, color) -> None:
        import pygame

        ratio = float(np.clip(ratio, 0.0, 1.0))
        pygame.draw.rect(canvas, (40, 40, 40), (x, y, w, h), width=2)
        pygame.draw.rect(canvas, (220, 220, 220), (x + 2, y + 2, w - 4, h - 4))
        pygame.draw.rect(canvas, color, (x + 2, y + 2, int((w - 4) * ratio), h - 4))

    def _draw_agent(self, canvas, agent: AgentState, scale: float, color) -> None:
        import pygame

        x, y = self._world_to_screen(agent.x, agent.y, scale)
        radius_px = max(4, int(self.cfg.agent_radius * scale))
        pygame.draw.circle(canvas, color, (x, y), radius_px)
        pygame.draw.circle(canvas, (20, 20, 20), (x, y), radius_px, width=2)

        if agent.shield_active():
            shield_radius_px = max(radius_px + 2, int(self._shield_radius() * scale))
            pygame.draw.circle(canvas, (90, 230, 180), (x, y), shield_radius_px, width=3)

        aim_len = int(0.9 * scale)
        tip_x = int(x + agent.aim_dx * aim_len)
        tip_y = int(y - agent.aim_dy * aim_len)
        pygame.draw.line(canvas, (20, 20, 20), (x, y), (tip_x, tip_y), 2)

    def _world_to_screen(self, x: float, y: float, scale: float) -> tuple[int, int]:
        pad = 40
        sx = int(pad + x * scale)
        sy = int(self._surface_size[1] - pad - y * scale)
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        pad = 40
        scale = (self._surface_size[0] - 80) / self.cfg.arena_size
        world_x = (float(sx) - pad) / scale
        world_y = (self._surface_size[1] - pad - float(sy)) / scale
        return float(np.clip(world_x, 0.0, self.cfg.arena_size)), float(np.clip(world_y, 0.0, self.cfg.arena_size))

    # ------------------------------------------------------------------
    # Normalization / perspective helpers
    # ------------------------------------------------------------------
    def _sector_to_unit_vector(self, sector: int) -> tuple[float, float]:
        angle = (2.0 * np.pi * sector) / self.cfg.aim_sectors
        return float(np.cos(angle)), float(np.sin(angle))

    def _vector_to_sector(self, dx: float, dy: float) -> int:
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return self.cfg.aim_sectors
        angle = np.arctan2(dy, dx) % (2.0 * np.pi)
        return int(np.round(angle / (2.0 * np.pi / self.cfg.aim_sectors))) % self.cfg.aim_sectors

    def mouse_to_sector(self, mouse_x: float, mouse_y: float, shooter: Optional[AgentState] = None) -> int:
        shooter = shooter or self.player_1
        world_x, world_y = self._screen_to_world(mouse_x, mouse_y)
        return self._vector_to_sector(world_x - shooter.x, world_y - shooter.y)

    def _norm_pos(self, v: float) -> float:
        return float(np.clip((2.0 * v / self.cfg.arena_size) - 1.0, -1.0, 1.0))

    def _norm_vel(self, v: float) -> float:
        max_speed = max(self.cfg.move_speed, self.cfg.projectile_speed)
        return float(np.clip(v / max_speed, -1.0, 1.0))

    def _aim_alignment_after_sector(self, shooter: AgentState, target: AgentState, sector: int) -> float:
        if sector >= self.cfg.aim_sectors:
            aim_dx, aim_dy = shooter.aim_dx, shooter.aim_dy
        else:
            aim_dx, aim_dy = self._sector_to_unit_vector(sector)
        dx = target.x - shooter.x
        dy = target.y - shooter.y
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            return 1.0
        tx, ty = dx / dist, dy / dist
        return float(np.clip(tx * aim_dx + ty * aim_dy, -1.0, 1.0))


SELF_BLOCK = slice(0, 12)
OTHER_BLOCK = slice(12, 24)
PROJ_SELF_BLOCK = slice(24, 29)
PROJ_OTHER_BLOCK = slice(29, 34)
REL_X_IDX = 34
REL_Y_IDX = 35
DIST_IDX = 36
TIME_IDX = 37
LOS_IDX = 38


def swap_perspective(obs: np.ndarray) -> np.ndarray:
    swapped = np.array(obs, dtype=np.float32, copy=True)
    self_block = np.array(obs[SELF_BLOCK], dtype=np.float32, copy=True)
    other_block = np.array(obs[OTHER_BLOCK], dtype=np.float32, copy=True)
    proj_self = np.array(obs[PROJ_SELF_BLOCK], dtype=np.float32, copy=True)
    proj_other = np.array(obs[PROJ_OTHER_BLOCK], dtype=np.float32, copy=True)

    swapped[SELF_BLOCK] = other_block
    swapped[OTHER_BLOCK] = self_block
    swapped[PROJ_SELF_BLOCK] = proj_other
    swapped[PROJ_OTHER_BLOCK] = proj_self
    swapped[REL_X_IDX] *= -1.0
    swapped[REL_Y_IDX] *= -1.0
    return swapped


def make_env(render_mode: Optional[str] = None, scripted_difficulty: float = 0.5) -> ArenaShooterEnv:
    return ArenaShooterEnv(render_mode=render_mode, scripted_difficulty=scripted_difficulty)


def build_vec_env(n_envs: int = 1, scripted_difficulty: float = 0.5, normalize: bool = True):
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training.") from exc

    def _factory():
        return Monitor(ArenaShooterEnv(render_mode=None, scripted_difficulty=scripted_difficulty))

    vec_env = DummyVecEnv([_factory for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def linear_schedule(initial_value: float, final_value: float):
    def _schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return float(initial_value + (final_value - initial_value) * progress)

    return _schedule


def normalize_single_obs(obs: np.ndarray, vec_norm) -> np.ndarray:
    if vec_norm is None:
        return np.asarray(obs, dtype=np.float32)
    norm_obs = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    return np.asarray(norm_obs[0], dtype=np.float32)


class EpisodeStatsCallback:
    @staticmethod
    def build(verbose: int = 0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for training.") from exc

        class _EpisodeStatsCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose=verbose)
                self.episode_count = 0
                self.current_returns: list[float] = []
                self.current_lengths: list[int] = []

            def _on_training_start(self) -> None:
                n_envs = int(getattr(self.training_env, "num_envs", 1))
                self.current_returns = [0.0 for _ in range(n_envs)]
                self.current_lengths = [0 for _ in range(n_envs)]

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                rewards = self.locals.get("rewards", [])
                for idx in range(len(rewards)):
                    self.current_returns[idx] += float(rewards[idx])
                    self.current_lengths[idx] += 1
                for idx, done in enumerate(dones):
                    if not done:
                        continue
                    self.episode_count += 1
                    info = infos[idx] if idx < len(infos) else {}
                    if isinstance(info, dict) and "episode" in info:
                        ep_reward = float(info["episode"].get("r", self.current_returns[idx]))
                        ep_len = int(info["episode"].get("l", self.current_lengths[idx]))
                    else:
                        ep_reward = float(self.current_returns[idx])
                        ep_len = int(self.current_lengths[idx])
                    if self.verbose > 0 and self.episode_count % 10 == 0:
                        print(
                            f"[episode {self.episode_count}] src={info.get('opponent_source')} style={info.get('opponent_style')} "
                            f"result={info.get('result')} player_hp={info.get('player_hp', 0.0):.1f} enemy_hp={info.get('enemy_hp', 0.0):.1f} ep_reward={ep_reward:.3f} ep_len={ep_len}"
                        )
                    self.current_returns[idx] = 0.0
                    self.current_lengths[idx] = 0
                return True

        return _EpisodeStatsCallback(verbose=verbose)


class SelfPlayManager:
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        max_snapshots: int = 8,
        snapshot_interval: int = 100_000,
        deterministic: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) / "selfplay_pool"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.deterministic = deterministic

        self.progress = 0.0
        self.current_model = None
        self.vec_norm = None
        self.snapshots: list[dict[str, Any]] = []
        self._policy = self._build_policy()

    def attach(self, model, vec_norm) -> None:
        self.current_model = model
        self.vec_norm = vec_norm

    def set_progress(self, progress: float) -> None:
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def opponent_mix(self) -> tuple[float, float]:
        """Pure self-play: current self vs older selves only."""
        if not self.snapshots:
            return 1.0, 0.0
        current = max(0.55, 1.0 - 0.45 * self.progress)
        snapshot = 1.0 - current
        total = current + snapshot
        return current / total, snapshot / total

    def sample_opponent_spec(self, env: ArenaShooterEnv) -> dict[str, Any]:
        current_w, snapshot_w = self.opponent_mix()
        r = float(env.np_random.uniform())
        if r < current_w or not self.snapshots:
            return {"kind": "current", "style": "self"}
        idx = int(env.np_random.integers(0, len(self.snapshots)))
        return {"kind": "snapshot", "snapshot_idx": idx, "style": f"snapshot_{idx}"}

    def _predict_with_model(self, model, obs: np.ndarray) -> np.ndarray:
        policy_obs = normalize_single_obs(swap_perspective(obs), self.vec_norm)
        action, _ = model.predict(policy_obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.int64)

    def _build_policy(self) -> OpponentPolicy:
        def _policy(env: ArenaShooterEnv, obs: np.ndarray) -> np.ndarray:
            if env._episode_opponent_spec is None:
                env._episode_opponent_spec = self.sample_opponent_spec(env)
                env._episode_opponent_source = str(env._episode_opponent_spec["kind"])
                env._episode_opponent_style = str(env._episode_opponent_spec.get("style", "unknown"))

            spec = env._episode_opponent_spec
            if spec["kind"] == "current" and self.current_model is not None:
                return self._predict_with_model(self.current_model, obs)
            if spec["kind"] == "snapshot" and self.snapshots:
                model = self.snapshots[int(spec["snapshot_idx"])] ["model"]
                return self._predict_with_model(model, obs)

            # Emergency fallback only. In normal training, current_model is attached before rollouts start.
            return env._scripted_opponent_action(style="balanced", difficulty=0.5)

        return _policy

    @property
    def policy(self) -> OpponentPolicy:
        return self._policy

    def add_snapshot(self) -> Optional[str]:
        if self.current_model is None:
            return None
        try:
            from stable_baselines3 import PPO
        except ImportError:
            return None

        snap_idx = len(self.snapshots)
        path = self.checkpoint_dir / f"{self.model_name}_snapshot_{snap_idx}_{int(self.progress * 1000):04d}.zip"
        self.current_model.save(str(path))
        snapshot_model = PPO.load(str(path))
        self.snapshots.append({"path": str(path), "model": snapshot_model})
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        return str(path)


class SelfPlayCallback:
    @staticmethod
    def build(manager: SelfPlayManager, total_timesteps: int, verbose: int = 0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for training.") from exc

        class _SelfPlayCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=verbose)
                self.last_snapshot_at = 0

            def _on_training_start(self) -> None:
                manager.attach(self.model, self.training_env if hasattr(self.training_env, "normalize_obs") else None)
                self.training_env.env_method("set_opponent_policy", manager.policy)

            def _on_step(self) -> bool:
                progress = min(1.0, self.num_timesteps / max(1, total_timesteps))
                manager.set_progress(progress)
                if self.num_timesteps - self.last_snapshot_at >= manager.snapshot_interval:
                    snap = manager.add_snapshot()
                    self.last_snapshot_at = self.num_timesteps
                    if self.verbose > 0 and snap is not None:
                        print(f"[self-play] added snapshot -> {snap}")
                return True

        return _SelfPlayCallback()


def train_ppo(
    total_timesteps: int = 1_500_000,
    n_envs: int = 8,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "ppo_arena_shooter_pure_selfplay",
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training.") from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(n_envs=n_envs, scripted_difficulty=0.5, normalize=True)

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
                save_path=str(checkpoint_path / "intermediate"),
                name_prefix=model_name,
            ),
        ]
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    save_path = checkpoint_path / f"{model_name}.zip"
    norm_path = checkpoint_path / f"{model_name}_vecnormalize.pkl"
    model.save(str(save_path))
    vec_env.save(str(norm_path))
    print(f"saved model -> {save_path}")
    print(f"saved vecnormalize -> {norm_path}")
    return model, vec_env


def load_model_bundle(model_path: str):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to load a trained model.") from exc

    model = PPO.load(model_path)
    vec_norm = None
    norm_path = Path(model_path).with_name(Path(model_path).stem + "_vecnormalize.pkl")
    if norm_path.exists():
        dummy_env = DummyVecEnv([lambda: Monitor(ArenaShooterEnv(render_mode=None))])
        vec_norm = VecNormalize.load(str(norm_path), dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False
    return model, vec_norm


def evaluate_policy_rollout(model_path: str, episodes: int = 5, render: bool = False) -> None:
    model, vec_norm = load_model_bundle(model_path)
    env = ArenaShooterEnv(render_mode="human" if render else None, scripted_difficulty=1.0)
    for ep in range(1, episodes + 1):
        obs, _info = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        info = {}
        while not done:
            policy_obs = normalize_single_obs(obs, vec_norm)
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        print(
            f"[eval episode {ep}] src={info.get('opponent_source')} style={info.get('opponent_style')} result={info['result']} player_hp={info['player_hp']:.1f} enemy_hp={info['enemy_hp']:.1f} total_reward={total_reward:.3f}"
        )
    env.close()
    if vec_norm is not None:
        vec_norm.close()


def build_model_opponent(model_path: str):
    model, vec_norm = load_model_bundle(model_path)

    def _policy(env: ArenaShooterEnv, obs: np.ndarray) -> np.ndarray:
        del env
        policy_obs = normalize_single_obs(swap_perspective(obs), vec_norm)
        action, _ = model.predict(policy_obs, deterministic=True)
        return np.asarray(action, dtype=np.int64)

    return _policy


def human_action_from_inputs(env: ArenaShooterEnv) -> np.ndarray:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for local play.") from exc

    keys = pygame.key.get_pressed()
    move_x = 1
    move_y = 1
    if keys[pygame.K_a] and not keys[pygame.K_d]:
        move_x = 0
    elif keys[pygame.K_d] and not keys[pygame.K_a]:
        move_x = 2
    if keys[pygame.K_s] and not keys[pygame.K_w]:
        move_y = 0
    elif keys[pygame.K_w] and not keys[pygame.K_s]:
        move_y = 2

    mx, my = pygame.mouse.get_pos()
    aim_sector = env.mouse_to_sector(mx, my, shooter=env.player_1)
    mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
    shoot = 1 if mouse_buttons[0] else 0
    blink = 1 if keys[pygame.K_SPACE] else 0
    shield = 1 if keys[pygame.K_q] or mouse_buttons[2] else 0
    return np.array([move_x, move_y, aim_sector, shoot, blink, shield], dtype=np.int64)


def run_local_match(model_path: str, seed: int = 42) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for local play.") from exc

    env = ArenaShooterEnv(render_mode="human")
    env.set_opponent_policy(build_model_opponent(model_path))

    obs, _info = env.reset(seed=seed)
    env._episode_opponent_source = "model"
    env._episode_opponent_style = "selfplay_best"
    print("Controls: WASD move, mouse aim, left click shoot, SPACE blink, Q/right click shield, R reset, ESC quit")

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
                    obs, _info = env.reset(seed=seed + round_idx)
                    env._episode_opponent_source = "model"
                    env._episode_opponent_style = "selfplay_best"
                    print(f"[round {round_idx}] manual reset")
        if not running:
            break

        action = human_action_from_inputs(env)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"[round {round_idx}] result={info['result']} player_hp={info['player_hp']:.1f} enemy_hp={info['enemy_hp']:.1f}")
            round_idx += 1
            obs, _info = env.reset(seed=seed + round_idx)
            env._episode_opponent_source = "model"
            env._episode_opponent_style = "selfplay_best"

    env.close()


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ArenaShooterEnv(render_mode=None)
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
