from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from arena_shooter_rl import ArenaShooterEnv, build_model_opponent

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "arena_basic_bot_400000_steps.zip"
USE_MODEL_BOT = False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class BrowserArenaSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seed = 42
        self.using_model = False
        self.status_message = ""
        self.env = self._make_env()
        self.last_snapshot = self._reset_locked()

    def _make_env(self) -> ArenaShooterEnv:
        env = ArenaShooterEnv(render_mode=None)
        env.set_scripted_difficulty(0.82)
        self.using_model = False
        self.status_message = "Scripted bot"

        if USE_MODEL_BOT and MODEL_PATH.exists():
            try:
                env.set_opponent_policy(build_model_opponent(str(MODEL_PATH)))
                self.using_model = True
                self.status_message = f"Model bot: {MODEL_PATH.name}"
            except Exception as exc:
                self.using_model = False
                self.status_message = f"Fallback scripted bot ({type(exc).__name__})"

        return env

    def _ensure_episode_meta(self) -> None:
        if self.using_model:
            self.env._episode_opponent_source = "model"
            self.env._episode_opponent_style = "selfplay_best"
        else:
            self.env._episode_opponent_source = "scripted"
            self.env._episode_opponent_style = "browser_balanced"

    def _serialize_agent(self, agent) -> dict[str, Any]:
        return {
            "x": float(agent.x),
            "y": float(agent.y),
            "vx": float(agent.vx),
            "vy": float(agent.vy),
            "hp": float(agent.hp),
            "aim_dx": float(agent.aim_dx),
            "aim_dy": float(agent.aim_dy),
            "shot_cd": int(agent.shot_cooldown_remaining),
            "blink_cd": int(agent.blink_cooldown_remaining),
            "shield_cd": int(agent.shield_cooldown_remaining),
            "shield_active": bool(agent.shield_active()),
            "shield_timer": int(agent.shield_active_timer),
            "shield_charges": int(agent.shield_charges),
        }

    def _serialize_snapshot(
        self,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self.env
        cfg = env.cfg
        if info is None:
            info = env._get_info()

        result = str(info.get("result", "ongoing"))
        time_left_steps = max(0, cfg.max_steps - env.step_count)
        time_left_seconds = float(time_left_steps / max(1, cfg.fps))

        return {
            "config": {
                "arena_size": float(cfg.arena_size),
                "max_hp": float(cfg.max_hp),
                "fps": int(cfg.fps),
                "max_steps": int(cfg.max_steps),
                "agent_radius": float(cfg.agent_radius),
                "projectile_radius": float(cfg.projectile_radius),
                "projectile_speed": float(cfg.projectile_speed),
                "projectile_damage": float(cfg.projectile_damage),
                "projectile_lifetime_steps": int(cfg.projectile_lifetime_steps),
                "blink_distance": float(cfg.blink_distance),
                "shield_duration": int(cfg.shield_duration),
                "shield_radius": float(env._shield_radius()),
            },
            "meta": {
                "using_model": bool(self.using_model),
                "status_message": self.status_message,
                "seed": int(self.seed),
            },
            "player": self._serialize_agent(env.player_1),
            "enemy": self._serialize_agent(env.player_2),
            "projectiles": [
                {
                    "x": float(proj.x),
                    "y": float(proj.y),
                    "vx": float(proj.vx),
                    "vy": float(proj.vy),
                    "owner": str(proj.owner),
                    "ttl": int(proj.ttl),
                }
                for proj in env.projectiles
            ],
            "obstacles": [
                {
                    "x": float(obs.x),
                    "y": float(obs.y),
                    "w": float(obs.w),
                    "h": float(obs.h),
                }
                for obs in env.obstacles
            ],
            "hud": {
                "step": int(env.step_count),
                "time_left_steps": int(time_left_steps),
                "time_left_seconds": time_left_seconds,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "result": result,
                "player_hp": float(env.player_1.hp),
                "enemy_hp": float(env.player_2.hp),
                "active_projectiles": int(len(env.projectiles)),
                "steps_since_last_damage_dealt": int(info.get("steps_since_last_damage_dealt", 0)),
                "opponent_source": str(info.get("opponent_source", "unknown")),
                "opponent_style": str(info.get("opponent_style", "unknown")),
            },
        }

    def _reset_locked(self) -> dict[str, Any]:
        _obs, info = self.env.reset(seed=self.seed)
        self._ensure_episode_meta()
        info = self.env._get_info()
        self.seed += 1
        self.last_snapshot = self._serialize_snapshot(info=info)
        return self.last_snapshot

    def reset(self) -> dict[str, Any]:
        with self.lock:
            return self._reset_locked()

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            self.last_snapshot = self._serialize_snapshot()
            return self.last_snapshot

    def _payload_to_action(self, payload: dict[str, Any]) -> list[int]:
        keys = payload.get("keys", {}) if isinstance(payload, dict) else {}
        aim = payload.get("aim", {}) if isinstance(payload, dict) else {}

        left = bool(keys.get("left", False))
        right = bool(keys.get("right", False))
        down = bool(keys.get("down", False))
        up = bool(keys.get("up", False))
        shoot = bool(keys.get("shoot", False))
        blink = bool(keys.get("blink", False))
        shield = bool(keys.get("shield", False))

        move_x = 1
        if left and not right:
            move_x = 0
        elif right and not left:
            move_x = 2

        move_y = 1
        if down and not up:
            move_y = 0
        elif up and not down:
            move_y = 2

        aim_x = float(aim.get("x", self.env.player_2.x))
        aim_y = float(aim.get("y", self.env.player_2.y))
        aim_x = _clamp(aim_x, 0.0, self.env.cfg.arena_size)
        aim_y = _clamp(aim_y, 0.0, self.env.cfg.arena_size)
        sector = int(self.env._vector_to_sector(aim_x - self.env.player_1.x, aim_y - self.env.player_1.y))

        return [
            move_x,
            move_y,
            sector,
            1 if shoot else 0,
            1 if blink else 0,
            1 if shield else 0,
        ]

    def step(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        with self.lock:
            self._ensure_episode_meta()
            action = self._payload_to_action(payload)
            _obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_snapshot = self._serialize_snapshot(
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
            return self.last_snapshot


app = Flask(__name__, template_folder="templates", static_folder="static")
SESSION = None


def get_session():
    global SESSION
    if SESSION is None:
        SESSION = BrowserArenaSession()
    return SESSION


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(get_session().get_state())


@app.post("/api/reset")
def api_reset():
    return jsonify(get_session().reset())


@app.post("/api/step")
def api_step():
    payload = request.get_json(silent=True) or {}
    return jsonify(get_session().step(payload))


@app.get("/api/ping")
def api_ping():
    return jsonify({"ok": True, "message": "browser arena server ready"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
