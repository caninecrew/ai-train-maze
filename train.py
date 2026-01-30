import argparse
import concurrent.futures
import multiprocessing as mp
import csv
import json
import os
import subprocess
import sys
import traceback
import random
import shutil
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict, cast

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from games.registry import get_game, list_games
from maze_game import find_latest_maze_id, load_maze_files
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dep
    yaml = None
_tensorboard_checked = False


def _tensorboard_available() -> bool:
    """Return True if tensorboard is installed; avoid hard dependency at runtime."""
    global _tensorboard_checked
    try:
        import tensorboard  # type: ignore  # noqa: F401
        _tensorboard_checked = True
        return True
    except Exception:
        if not _tensorboard_checked:
            print("TensorBoard not installed; run `python -m pip install tensorboard` to enable logging.")
            _tensorboard_checked = True
    return False


class GoalReachedStopCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._triggered = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            try:
                if float(info.get("goal_reached", 0.0)) >= 1.0:
                    self._triggered = True
                    break
            except Exception:
                continue
        if self._triggered:
            print("[train] Goal reached in rollout; stopping this model early.")
            return False
        return True


def _parse_resolution(res_str: str) -> Tuple[int, int]:
    if "x" not in res_str:
        raise ValueError("Resolution must be in <width>x<height> format.")
    w_str, h_str = res_str.lower().split("x", 1)
    return int(w_str), int(h_str)


def _resolve_affinity_list(cpu_affinity: Optional[str], n_envs: int) -> Optional[List[int]]:
    if not cpu_affinity:
        return None
    if cpu_affinity.lower() == "auto":
        try:
            cpus = list(range(os.cpu_count() or 1))
            stride = max(1, len(cpus) // max(1, n_envs))
            return cpus[::stride] or cpus
        except Exception:
            return None
    try:
        return [int(x) for x in cpu_affinity.split(",") if x.strip()]
    except Exception:
        return None


def _add_overlay(frame: np.ndarray, text: str, footer: str = "") -> np.ndarray:
    if not text and not footer:
        return frame
    img = Image.fromarray(frame)
    drawer = ImageDraw.Draw(img)
    combined = text
    if footer:
        combined = f"{text} | {footer}" if text else footer
    if combined:
        y0 = img.height - 24
        drawer.rectangle([(0, y0), (img.width, img.height)], fill=(0, 0, 0))
        drawer.text((6, y0 + 4), combined, fill=(255, 255, 255))
    return np.array(img)


def record_video_segment(
    game,
    model: PPO,
    steps: int = 400,
    overlay_text: str = "",
    resolution: Tuple[int, int] = (320, 192),
    variant: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Roll out a short episode with the trained model and return frames.
    """
    env = game.make_env(render_mode="rgb_array", seed=seed, variant=variant)
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    target_size = resolution  # divisible by 16 to keep codecs happy

    frame = env.render()
    if frame is not None:
        img = Image.fromarray(frame)
        if img.size != target_size:
            resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", None)
            if resample is None:
                resample = getattr(Image, "NEAREST", 0)
            img = img.resize(target_size, resample=resample)
        footer = ""
        try:
            if hasattr(env, "get_agent_cell"):
                r, c = env.get_agent_cell()
                footer = f"agent=({r},{c})"
        except Exception:
            footer = ""
        frames.append(_add_overlay(np.array(img), overlay_text, footer=footer))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            img = Image.fromarray(frame)
            if img.size != target_size:
                resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", None)
                if resample is None:
                    resample = getattr(Image, "NEAREST", 0)
                img = img.resize(target_size, resample=resample)
            footer = ""
            try:
                if hasattr(env, "get_agent_cell"):
                    r, c = env.get_agent_cell()
                    footer = f"agent=({r},{c})"
            except Exception:
                footer = ""
            frames.append(_add_overlay(np.array(img), overlay_text, footer=footer))
        if terminated or truncated:
            break

    env.close()
    return frames


def record_video_segment_with_goal(
    game,
    model: PPO,
    steps: int = 400,
    overlay_text: str = "",
    resolution: Tuple[int, int] = (320, 192),
    variant: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], bool]:
    """
    Roll out a short episode and report whether the goal was reached.
    """
    env = game.make_env(render_mode="rgb_array", seed=seed, variant=variant)
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    target_size = resolution
    goal_reached = False

    frame = env.render()
    if frame is not None:
        img = Image.fromarray(frame)
        if img.size != target_size:
            resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", None)
            if resample is None:
                resample = getattr(Image, "NEAREST", 0)
            img = img.resize(target_size, resample=resample)
        footer = ""
        try:
            if hasattr(env, "get_agent_cell"):
                r, c = env.get_agent_cell()
                footer = f"agent=({r},{c})"
        except Exception:
            footer = ""
        frames.append(_add_overlay(np.array(img), overlay_text, footer=footer))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            img = Image.fromarray(frame)
            if img.size != target_size:
                resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", None)
                if resample is None:
                    resample = getattr(Image, "NEAREST", 0)
                img = img.resize(target_size, resample=resample)
            footer = ""
            try:
                if hasattr(env, "get_agent_cell"):
                    r, c = env.get_agent_cell()
                    footer = f"agent=({r},{c})"
            except Exception:
                footer = ""
            frames.append(_add_overlay(np.array(img), overlay_text, footer=footer))
        if terminated or truncated:
            break
    try:
        if hasattr(env, "get_eval_stats"):
            stats = env.get_eval_stats()
            goal_reached = bool(stats.get("goal_reached", 0.0))
    except Exception:
        goal_reached = False
    env.close()
    return frames, goal_reached


def build_grid_frames(segments: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Arrange per-model segments into a grid per timestep.
    """
    if not segments or not any(segments):
        return []

    max_len = max(len(seg) for seg in segments)
    num_models = len(segments)
    cols = int(np.ceil(np.sqrt(num_models)))
    rows = int(np.ceil(num_models / cols))

    max_h = 0
    max_w = 0
    for seg in segments:
        for frame in seg:
            h, w = frame.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

    if max_h == 0 or max_w == 0:
        return []

    def _pad(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == max_h and w == max_w:
            return frame
        canvas = np.zeros((max_h, max_w, frame.shape[2]), dtype=frame.dtype)
        canvas[:h, :w, :] = frame
        return canvas

    grid_frames: List[np.ndarray] = []

    for i in range(max_len):
        row_images = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                if idx >= num_models:
                    row_tiles.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
                    continue
                seg = segments[idx]
                if seg:
                    if i < len(seg):
                        row_tiles.append(_pad(seg[i]))
                    else:
                        row_tiles.append(_pad(seg[-1]))  # hold last frame if shorter
                else:
                    row_tiles.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
            if row_tiles:
                row_images.append(np.concatenate(row_tiles, axis=1))
        if row_images:
            grid_frame = np.concatenate(row_images, axis=0)
            grid_frames.append(grid_frame)

    return grid_frames


def _safe_write_video(frames: List[np.ndarray], path: Path, fps: int, final_overlay: str = "") -> bool:
    """
    Write frames to mp4 using ffmpeg if available. Returns True on success.
    """
    if not frames:
        print("No frames to write; skipping video.")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as iio  # use v2 API for stable ffmpeg handling

        writer = cast(Any, iio.get_writer(path, format="ffmpeg", fps=fps))  # type: ignore[arg-type]
        with writer:
            for idx, frame in enumerate(frames):
                if final_overlay and idx == len(frames) - 1:
                    frame = _add_overlay(frame, final_overlay)
                writer.append_data(frame)
        return True
    except Exception as exc:
        print(f"Video write failed for {path.name}: {exc}")
        return False


@dataclass
class TrainConfig:
    game: str = "template"
    model_prefix: Optional[str] = None
    train_timesteps: int = 750_000
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    device: str = "auto"
    target_fps: int = 30
    max_video_seconds: int = 30  # total seconds per cycle video
    video_steps: int = 600  # total frames per cycle video
    max_cycles: int = 5
    checkpoint_interval: int = 1  # cycles between timestamped checkpoints
    iterations_per_set: int = 4  # how many parallel model lines to train each cycle
    n_envs: int = 12  # vectorized envs per PPO learner
    seed: int = 0
    deterministic: bool = False
    base_seed: int = 0
    early_stop_patience: int = 0
    improvement_threshold: float = 0.05
    eval_episodes: int = 16
    eval_video_steps: int = 0
    long_eval_video_steps: int = 0
    top_k_checkpoints: int = 3
    no_checkpoint: bool = False
    individual_videos: bool = False
    cpu_affinity: Optional[str] = None  # e.g. "0,1,2" or "auto"
    num_threads: Optional[int] = 2
    video_dir: str = "videos"
    model_dir: str = "models"
    log_dir: str = "logs"
    metrics_csv: str = "logs/metrics.csv"
    metrics_deltas: bool = True
    stream_tensorboard: bool = False
    status: bool = False
    resume_from: Optional[str] = None
    config_path: Optional[str] = None
    profile: Optional[str] = None
    dry_run: bool = False
    video_resolution: str = "320x192"
    eval_overlay: bool = True
    eval_deterministic: bool = False
    worker_watchdog: bool = True
    list_games: bool = False
    export_config: Optional[str] = None
    evo_first: bool = False
    evo_cycles: int = 20
    evo_population: int = 50
    evo_max_steps: int = 2400
    evo_sensor_range: int = 20
    evo_top_k: int = 10
    evo_batch_size: int = 100
    evo_epochs: int = 15
    evo_learning_rate: float = 1e-3
    evo_model_path: str = "models/evo_maze.pt"

    @property
    def max_video_frames(self) -> int:
        return self.target_fps * self.max_video_seconds


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot read YAML configs.")
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of keys to values.")
    defaults = data.get("defaults", data)
    profiles = data.get("profiles", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config defaults must be a mapping.")
    if profiles and not isinstance(profiles, dict):
        raise ValueError("Config profiles must be a mapping of profile -> overrides.")
    return {"defaults": defaults, "profiles": profiles}


def _merge_profile_config(file_cfg: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    defaults = file_cfg.get("defaults", {})
    profiles = file_cfg.get("profiles", {})
    merged = dict(defaults)
    if profile and isinstance(profiles, dict) and profile in profiles:
        overrides = profiles[profile]
        if isinstance(overrides, dict):
            merged.update(overrides)
    return merged


def _extract_env_config(file_cfg: Dict[str, Any], profile: Optional[str]) -> Dict[str, str]:
    defaults = file_cfg.get("defaults", {})
    profiles = file_cfg.get("profiles", {})
    env_cfg = {}
    if isinstance(defaults, dict) and isinstance(defaults.get("env"), dict):
        env_cfg.update(defaults.get("env", {}))
    if profile and isinstance(profiles, dict) and profile in profiles:
        overrides = profiles[profile]
        if isinstance(overrides, dict) and isinstance(overrides.get("env"), dict):
            env_cfg.update(overrides.get("env", {}))
    return {str(k): str(v) for k, v in env_cfg.items()}


def _apply_env_config(env_cfg: Dict[str, str]) -> None:
    for key, value in env_cfg.items():
        if key and (key not in os.environ or os.environ.get(key, "").strip() == ""):
            os.environ[key] = value


def _apply_profile(cfg: TrainConfig) -> None:
    """
    Apply preset overrides for common workflows.
    """
    if not cfg.profile:
        return
    profiles: Dict[str, Dict[str, Any]] = {
        "quick": {
            "train_timesteps": 10_000,
            "iterations_per_set": 1,
            "max_cycles": 1,
            "eval_episodes": 1,
            "video_steps": 300,
            "max_video_seconds": 20,
            "checkpoint_interval": 0,
            "no_checkpoint": True,
        },
        "single": {
            "iterations_per_set": 1,
            "n_envs": 4,
            "eval_episodes": max(1, cfg.eval_episodes),
        },
        "gpu": {
            "device": "cuda",
            "batch_size": max(cfg.batch_size, 1024),
            "n_envs": max(cfg.n_envs, 16),
            "train_timesteps": max(cfg.train_timesteps, 500_000),
        },
    }
    overrides = profiles.get(cfg.profile)
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)
        print(f"Applied profile '{cfg.profile}' overrides: {overrides}")


_progress_bar_checked = False
_progress_bar_available = False


def _progress_bar_ready(suppress_log: bool = False) -> bool:
    """
    Check if optional progress bar deps are present. Prints a hint only once.
    """
    global _progress_bar_checked, _progress_bar_available
    if _progress_bar_checked:
        return _progress_bar_available
    _progress_bar_checked = True
    try:
        import rich  # type: ignore  # noqa: F401
        import tqdm  # type: ignore  # noqa: F401

        _progress_bar_available = True
    except Exception:
        if not suppress_log:
            print("Progress bar disabled (install 'rich' and 'tqdm' or stable-baselines3[extra] to enable).")
    return _progress_bar_available


def _auto_goal_enabled() -> bool:
    raw = os.getenv("MAZE_TRAIN_AUTO", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _latest_run_timestamp_from_logs(log_dir: str) -> Optional[str]:
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    jsonl_candidates = sorted(log_path.glob("train_run_*.jsonl"), reverse=True)
    for path in jsonl_candidates:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(first_line)
            run_ts = payload.get("resolved_at") or payload.get("run_timestamp")
            if run_ts:
                return str(run_ts)
        except Exception:
            continue

    json_candidates = sorted(log_path.glob("run_report_*.json"), reverse=True)
    for path in json_candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_ts = payload.get("summary", {}).get("run_timestamp")
            if run_ts:
                return str(run_ts)
        except Exception:
            continue

    html_candidates = sorted(log_path.glob("run_report_*.html"), reverse=True)
    for path in html_candidates:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"run_timestamp\\\":\\s*\\\"(\\d{8}-\\d{6})\\\"", content)
            if match:
                return match.group(1)
        except Exception:
            continue
    return None


def _metrics_fieldnames_for_game(game_name: str, extra_metrics: List[str]) -> List[str]:
    return [
        "cycle",
        "model_id",
        "model_index",
        "avg_reward",
        "avg_reward_ci",
        "avg_ep_len",
        "avg_ep_len_ci",
        *extra_metrics,
        "rank_avg_reward",
        "rank_best_dist",
        "rank_goal_rate",
        "rank_best_progress",
        "delta_reward",
        "train_goal",
        "train_goal_fraction",
        "train_timesteps",
        "iterations_per_set",
        "eval_episodes",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "video_steps",
        "max_video_seconds",
        "target_fps",
        "cycle_start",
        "cycle_duration_s",
        "eta_end",
        "timestamp",
        "run_timestamp",
        "game",
    ]


def parse_args() -> TrainConfig:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, help="Path to YAML/JSON config to merge.")
    base_parser.add_argument("--profile", choices=["quick", "single", "gpu"], help="Preset overrides for common workflows.")
    known, remaining = base_parser.parse_known_args()
    file_cfg = _load_config_file(known.config)
    defaults_from_file = _merge_profile_config(file_cfg, known.profile)
    env_from_file = _extract_env_config(file_cfg, known.profile)

    parser = argparse.ArgumentParser(description="Train PPO agents for a registered game.", parents=[base_parser])
    parser.add_argument("--game", type=str, default=defaults_from_file.get("game", TrainConfig.game), help="Game key to train.")
    parser.add_argument("--model-prefix", type=str, default=defaults_from_file.get("model_prefix", TrainConfig.model_prefix), help="Prefix for model ids.")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit.")
    parser.add_argument("--export-config", type=str, default=defaults_from_file.get("export_config", TrainConfig.export_config), help="Write resolved config to JSON and exit.")
    parser.add_argument("--status", action="store_true", default=defaults_from_file.get("status", False), help="Show current best status and exit.")
    parser.add_argument("--train-timesteps", type=int, default=defaults_from_file.get("train_timesteps", TrainConfig.train_timesteps))
    parser.add_argument("--n-steps", type=int, default=defaults_from_file.get("n_steps", TrainConfig.n_steps))
    parser.add_argument("--batch-size", type=int, default=defaults_from_file.get("batch_size", TrainConfig.batch_size))
    parser.add_argument("--n-epochs", type=int, default=defaults_from_file.get("n_epochs", TrainConfig.n_epochs))
    parser.add_argument("--gamma", type=float, default=defaults_from_file.get("gamma", TrainConfig.gamma))
    parser.add_argument("--learning-rate", type=float, default=defaults_from_file.get("learning_rate", TrainConfig.learning_rate))
    parser.add_argument("--device", type=str, default=defaults_from_file.get("device", TrainConfig.device))
    parser.add_argument("--target-fps", type=int, default=defaults_from_file.get("target_fps", TrainConfig.target_fps))
    parser.add_argument("--max-video-seconds", type=int, default=defaults_from_file.get("max_video_seconds", TrainConfig.max_video_seconds))
    parser.add_argument("--video-steps", type=int, default=defaults_from_file.get("video_steps", TrainConfig.video_steps), help="Frames captured per clip; e.g., 1800 ~= 60s at 30 fps, 3600 ~= 2min.")
    parser.add_argument("--max-cycles", type=int, default=defaults_from_file.get("max_cycles", TrainConfig.max_cycles))
    parser.add_argument("--checkpoint-interval", type=int, default=defaults_from_file.get("checkpoint_interval", TrainConfig.checkpoint_interval))
    parser.add_argument("--iterations-per-set", type=int, default=defaults_from_file.get("iterations_per_set", TrainConfig.iterations_per_set))
    parser.add_argument("--n-envs", type=int, default=defaults_from_file.get("n_envs", TrainConfig.n_envs))
    parser.add_argument("--seed", type=int, default=defaults_from_file.get("seed", TrainConfig.seed))
    parser.add_argument("--base-seed", type=int, default=defaults_from_file.get("base_seed", TrainConfig.base_seed))
    parser.add_argument("--deterministic", action="store_true", default=defaults_from_file.get("deterministic", TrainConfig.deterministic))
    parser.add_argument("--early-stop-patience", type=int, default=defaults_from_file.get("early_stop_patience", TrainConfig.early_stop_patience))
    parser.add_argument("--improvement-threshold", type=float, default=defaults_from_file.get("improvement_threshold", TrainConfig.improvement_threshold))
    parser.add_argument("--eval-episodes", type=int, default=defaults_from_file.get("eval_episodes", TrainConfig.eval_episodes))
    parser.add_argument("--eval-video-steps", type=int, default=defaults_from_file.get("eval_video_steps", TrainConfig.eval_video_steps), help="Optional extra eval video steps.")
    parser.add_argument("--long-eval-video-steps", type=int, default=defaults_from_file.get("long_eval_video_steps", TrainConfig.long_eval_video_steps), help="Capture a longer evaluation match separately.")
    parser.add_argument("--top-k-checkpoints", type=int, default=defaults_from_file.get("top_k_checkpoints", TrainConfig.top_k_checkpoints))
    parser.add_argument("--no-checkpoint", action="store_true", default=defaults_from_file.get("no_checkpoint", TrainConfig.no_checkpoint))
    parser.add_argument("--individual-videos", action="store_true", default=defaults_from_file.get("individual_videos", TrainConfig.individual_videos))
    parser.add_argument("--evo-first", action="store_true", default=defaults_from_file.get("evo_first", TrainConfig.evo_first), help="Run evolutionary maze pretraining before PPO (maze only).")
    parser.add_argument("--evo-cycles", type=int, default=defaults_from_file.get("evo_cycles", TrainConfig.evo_cycles))
    parser.add_argument("--evo-population", type=int, default=defaults_from_file.get("evo_population", TrainConfig.evo_population))
    parser.add_argument("--evo-max-steps", type=int, default=defaults_from_file.get("evo_max_steps", TrainConfig.evo_max_steps))
    parser.add_argument("--evo-sensor-range", type=int, default=defaults_from_file.get("evo_sensor_range", TrainConfig.evo_sensor_range))
    parser.add_argument("--evo-top-k", type=int, default=defaults_from_file.get("evo_top_k", TrainConfig.evo_top_k))
    parser.add_argument("--evo-batch-size", type=int, default=defaults_from_file.get("evo_batch_size", TrainConfig.evo_batch_size))
    parser.add_argument("--evo-epochs", type=int, default=defaults_from_file.get("evo_epochs", TrainConfig.evo_epochs))
    parser.add_argument("--evo-learning-rate", type=float, default=defaults_from_file.get("evo_learning_rate", TrainConfig.evo_learning_rate))
    parser.add_argument("--evo-model-path", type=str, default=defaults_from_file.get("evo_model_path", TrainConfig.evo_model_path))
    parser.add_argument("--cpu-affinity", type=str, default=defaults_from_file.get("cpu_affinity", TrainConfig.cpu_affinity))
    parser.add_argument("--num-threads", type=int, default=defaults_from_file.get("num_threads", TrainConfig.num_threads) or None, help="Override torch.num_threads.")
    parser.add_argument("--video-dir", type=str, default=defaults_from_file.get("video_dir", TrainConfig.video_dir))
    parser.add_argument("--model-dir", type=str, default=defaults_from_file.get("model_dir", TrainConfig.model_dir))
    parser.add_argument("--log-dir", type=str, default=defaults_from_file.get("log_dir", TrainConfig.log_dir))
    parser.add_argument("--metrics-csv", type=str, default=defaults_from_file.get("metrics_csv", TrainConfig.metrics_csv))
    parser.add_argument("--metrics-deltas", action="store_true", default=defaults_from_file.get("metrics_deltas", TrainConfig.metrics_deltas))
    parser.add_argument("--stream-tensorboard", action="store_true", default=defaults_from_file.get("stream_tensorboard", TrainConfig.stream_tensorboard))
    parser.add_argument("--resume-from", type=str, default=defaults_from_file.get("resume_from", TrainConfig.resume_from), help="Checkpoint to resume from instead of *_latest.")
    parser.add_argument("--video-resolution", type=str, default=defaults_from_file.get("video_resolution", TrainConfig.video_resolution))
    parser.add_argument("--eval-deterministic", action="store_true", default=defaults_from_file.get("eval_deterministic", TrainConfig.eval_deterministic), help="Deterministic eval to reduce variance.")
    parser.add_argument("--watchdog", dest="worker_watchdog", action="store_true", default=defaults_from_file.get("worker_watchdog", TrainConfig.worker_watchdog), help="Keep training when a worker fails.")
    parser.add_argument("--dry-run", action="store_true", help="Parse config, build envs, and exit without training.")
    args = parser.parse_args(remaining)
    cfg = TrainConfig(
        game=args.game,
        model_prefix=args.model_prefix,
        train_timesteps=args.train_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        device=args.device,
        target_fps=args.target_fps,
        max_video_seconds=args.max_video_seconds,
        video_steps=args.video_steps,
        max_cycles=args.max_cycles,
        checkpoint_interval=args.checkpoint_interval,
        iterations_per_set=args.iterations_per_set,
        n_envs=args.n_envs,
        seed=args.seed,
        base_seed=args.base_seed,
        deterministic=args.deterministic,
        early_stop_patience=args.early_stop_patience,
        improvement_threshold=args.improvement_threshold,
        eval_episodes=args.eval_episodes,
        eval_video_steps=args.eval_video_steps,
        long_eval_video_steps=args.long_eval_video_steps,
        top_k_checkpoints=args.top_k_checkpoints,
        no_checkpoint=args.no_checkpoint,
        individual_videos=args.individual_videos,
        evo_first=args.evo_first,
        evo_cycles=args.evo_cycles,
        evo_population=args.evo_population,
        evo_max_steps=args.evo_max_steps,
        evo_sensor_range=args.evo_sensor_range,
        evo_top_k=args.evo_top_k,
        evo_batch_size=args.evo_batch_size,
        evo_epochs=args.evo_epochs,
        evo_learning_rate=args.evo_learning_rate,
        evo_model_path=args.evo_model_path,
        cpu_affinity=args.cpu_affinity,
        num_threads=args.num_threads,
        video_dir=args.video_dir,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        metrics_csv=args.metrics_csv,
        metrics_deltas=args.metrics_deltas,
        stream_tensorboard=args.stream_tensorboard,
        status=args.status,
        resume_from=args.resume_from,
        config_path=known.config,
        profile=args.profile,
        dry_run=args.dry_run,
        video_resolution=args.video_resolution,
        eval_deterministic=args.eval_deterministic,
        worker_watchdog=args.worker_watchdog,
        list_games=args.list_games,
        export_config=args.export_config,
    )
    _apply_profile(cfg)
    _apply_env_config(env_from_file)
    if cfg.video_steps > cfg.max_video_frames:
        print(
            f"Warning: video_steps ({cfg.video_steps}) exceeds max frames from max_video_seconds*target_fps ({cfg.max_video_frames}). "
            "Consider lowering --video-steps or raising --max-video-seconds/--target-fps."
        )
    return cfg


def _maze_video_resolution() -> Optional[str]:
    maze_dir = os.getenv("MAZE_DIR", "data/mazes")
    maze_id = os.getenv("MAZE_ID", "").strip()
    if maze_id:
        raw = maze_id
        if raw.lower().startswith("maze_"):
            raw = raw.split("maze_", 1)[1]
        maze_id = raw.zfill(3)
    else:
        try:
            maze_id = str(find_latest_maze_id(maze_dir)).zfill(3)
        except Exception:
            return None

    try:
        grid, _ = load_maze_files(maze_dir, maze_id, prefer_npy=True)
    except Exception:
        return None
    cell_size_env = os.getenv("MAZE_CELL_SIZE", "").strip()
    try:
        cell_size = max(2, int(cell_size_env)) if cell_size_env else 8
    except ValueError:
        cell_size = 8
    rows, cols = grid.shape
    return f"{cols * cell_size}x{rows * cell_size}"


def _print_status(cfg: TrainConfig) -> None:
    metrics_path = Path(cfg.metrics_csv)
    if not metrics_path.exists():
        print(f"No metrics CSV found at {metrics_path}; run training first.")
        return
    import csv

    best_row = None
    with metrics_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                reward = float(row.get("avg_reward", 0.0))
            except Exception:
                reward = float("-inf")
            if best_row is None or reward > float(best_row.get("avg_reward", float("-inf"))):
                best_row = row
    if not best_row:
        print("Metrics file is empty; no status to report.")
        return
    print("=== Current Status ===")
    extra = []
    if best_row.get("win_rate"):
        extra.append(f"win_rate={best_row.get('win_rate')}")
    if best_row.get("avg_ep_len"):
        extra.append(f"avg_ep_len={best_row.get('avg_ep_len')}")
    extra_str = " | " + " | ".join(extra) if extra else ""
    print(f"Best model id: {best_row.get('model_id')} | avg_reward={best_row.get('avg_reward')}{extra_str}")
    print(f"Last recorded at cycle {best_row.get('cycle')} on {best_row.get('timestamp')}")


def _best_checkpoint_from_metrics(cfg: TrainConfig, model_prefix: str) -> Optional[str]:
    metrics_path = Path(cfg.metrics_csv)
    if not metrics_path.exists():
        return None
    best_row = None
    with metrics_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model_id = row.get("model_id", "")
            if not model_id or not model_id.startswith(model_prefix):
                continue
            try:
                goal_rate = float(row.get("goal_reached_rate") or 0.0)
            except Exception:
                goal_rate = 0.0
            try:
                best_dist = float(row.get("best_dist") or float("inf"))
            except Exception:
                best_dist = float("inf")
            if best_row is None:
                best_row = row
                continue
            try:
                best_goal_rate = float(best_row.get("goal_reached_rate") or 0.0)
            except Exception:
                best_goal_rate = 0.0
            try:
                best_best_dist = float(best_row.get("best_dist") or float("inf"))
            except Exception:
                best_best_dist = float("inf")
            if (goal_rate > best_goal_rate) or (
                goal_rate == best_goal_rate and best_dist < best_best_dist
            ):
                best_row = row
    if not best_row:
        return None
    model_id = best_row.get("model_id", "")
    if not model_id:
        return None
    candidate = Path(cfg.model_dir) / f"{model_id}_latest.zip"
    return str(candidate) if candidate.exists() else None


def _auto_training_goal_from_metrics(cfg: TrainConfig) -> Optional[Dict[str, str]]:
    metrics_path = Path(cfg.metrics_csv)
    if not metrics_path.exists():
        return None
    latest_by_run: Dict[str, Dict[str, Any]] = {}
    per_run_cycles: Dict[str, Dict[int, Dict[str, Any]]] = {}
    try:
        with metrics_path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline()
            fh.seek(0)
            has_header = "cycle" in first_line and "model_id" in first_line and "game" in first_line
            if has_header:
                reader = csv.DictReader(fh)
                rows = list(reader)
            else:
                game = get_game("maze")
                fieldnames = _metrics_fieldnames_for_game("maze", list(game.extra_metrics))
                raw_rows = csv.reader(fh)
                rows = []
                for raw in raw_rows:
                    padded = raw + [""] * max(0, len(fieldnames) - len(raw))
                    rows.append(dict(zip(fieldnames, padded[: len(fieldnames)])))
            for row in rows:
                if row.get("game") != "maze":
                    continue
                run_id = row.get("run_timestamp") or ""
                if not run_id:
                    continue
                try:
                    goal_rate = float(row.get("goal_reached_rate") or 0.0)
                except Exception:
                    goal_rate = 0.0
                try:
                    best_dist = float(row.get("best_dist") or 0.0)
                except Exception:
                    best_dist = 0.0
                train_goal = row.get("train_goal") or ""
                train_goal_fraction = row.get("train_goal_fraction") or ""
                try:
                    final_row = float(row.get("final_row")) if row.get("final_row") not in {None, ""} else None
                    final_col = float(row.get("final_col")) if row.get("final_col") not in {None, ""} else None
                except Exception:
                    final_row = None
                    final_col = None
                try:
                    final_dist = float(row.get("final_dist") or 0.0)
                except Exception:
                    final_dist = 0.0
                latest_by_run.setdefault(
                    run_id,
                    {
                        "goal_rate": 0.0,
                        "best_dist": best_dist,
                        "train_goal": train_goal,
                        "train_goal_fraction": train_goal_fraction,
                        "final_dist": final_dist,
                        "final_row": final_row,
                        "final_col": final_col,
                    },
                )
                if run_id not in per_run_cycles:
                    per_run_cycles[run_id] = {}
                try:
                    cycle_num = int(float(row.get("cycle") or 0))
                except Exception:
                    cycle_num = 0
                cycle_entry = per_run_cycles[run_id].setdefault(
                    cycle_num,
                    {
                        "goal_rate": 0.0,
                        "best_dist": best_dist,
                        "final_dist": final_dist,
                        "final_row": final_row,
                        "final_col": final_col,
                        "timestamp": row.get("timestamp") or "",
                    },
                )
                # keep best stats per cycle (closest final_dist, smallest best_dist)
                if final_dist < float(cycle_entry.get("final_dist", float("inf"))):
                    cycle_entry["final_dist"] = final_dist
                    cycle_entry["final_row"] = final_row
                    cycle_entry["final_col"] = final_col
                if best_dist < float(cycle_entry.get("best_dist", float("inf"))):
                    cycle_entry["best_dist"] = best_dist
                if goal_rate > float(cycle_entry.get("goal_rate", 0.0)):
                    cycle_entry["goal_rate"] = goal_rate
                ts = row.get("timestamp") or ""
                if ts and (not cycle_entry.get("timestamp") or ts > cycle_entry.get("timestamp")):
                    cycle_entry["timestamp"] = ts
                latest_by_run[run_id]["goal_rate"] = max(latest_by_run[run_id]["goal_rate"], goal_rate)
                latest_by_run[run_id]["best_dist"] = min(latest_by_run[run_id]["best_dist"], best_dist)
                if final_row is not None and final_col is not None:
                    if final_dist < float(latest_by_run[run_id].get("final_dist", float("inf"))):
                        latest_by_run[run_id]["final_dist"] = final_dist
                        latest_by_run[run_id]["final_row"] = final_row
                        latest_by_run[run_id]["final_col"] = final_col
                if train_goal:
                    latest_by_run[run_id]["train_goal"] = train_goal
                if train_goal_fraction:
                    latest_by_run[run_id]["train_goal_fraction"] = train_goal_fraction
    except Exception:
        return None
    if not latest_by_run:
        return None
    preferred_run = _latest_run_timestamp_from_logs(cfg.log_dir)
    last_run = preferred_run if preferred_run in latest_by_run else sorted(latest_by_run.keys())[-1]
    stats = latest_by_run[last_run]
    fraction_env = stats.get("train_goal_fraction") or os.getenv("MAZE_TRAIN_GOAL_FRACTION", "").strip()
    try:
        fraction = float(fraction_env) if fraction_env else 0.25
    except ValueError:
        fraction = 0.25
    train_goal = str(stats.get("train_goal") or "")
    final_row = stats.get("final_row")
    final_col = stats.get("final_col")
    if final_row is not None and final_col is not None:
        try:
            train_goal = f"{int(final_row)},{int(final_col)}"
        except Exception:
            pass
    goal_rate = stats.get("goal_rate", 0.0)
    best_dist = stats.get("best_dist", 1.0)
    reached = goal_rate > 0.0 or best_dist <= 1.0
    solved = goal_rate >= 0.9 and best_dist <= 1.0
    streak_env = os.getenv("MAZE_AUTO_GOAL_STREAK", "").strip()
    try:
        required_streak = max(1, int(streak_env)) if streak_env else 3
    except ValueError:
        required_streak = 2
    streak = 0
    cycles = sorted(per_run_cycles.get(last_run, {}).items(), key=lambda t: t[0])
    for _, cstats in cycles:
        c_reached = bool(float(cstats.get("goal_rate", 0.0)) > 0.0 or float(cstats.get("best_dist", 1e9)) <= 1.0)
        if c_reached:
            streak += 1
        else:
            streak = 0
    advance = streak >= required_streak
    if advance:
        if train_goal:
            train_goal = ""
        fraction = min(0.9, fraction + 0.03)
    elif solved:
        if train_goal:
            train_goal = ""
        fraction = min(0.9, fraction + 0.05)
    return {
        "train_goal": train_goal,
        "train_goal_fraction": str(max(0.05, min(0.9, fraction))),
    }


def _git_commit_artifacts(cfg: TrainConfig, message: str) -> None:
    if os.getenv("CI", "").lower() != "true":
        return
    if os.getenv("TRAIN_DISABLE_GIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    try:
        import subprocess

        def _run(cmd: list[str], timeout: int = 30) -> bool:
            try:
                result = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
                return result.returncode == 0
            except Exception:
                return False

        if not _run(["git", "rev-parse", "--is-inside-work-tree"]):
            return
        _run(["git", "config", "user.name", "github-actions[bot]"])
        _run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
        _run(["git", "pull", "--rebase", "--autostash"], timeout=60)
        _run(["git", "add", cfg.model_dir, cfg.video_dir, cfg.log_dir], timeout=30)
        if _run(["git", "diff", "--cached", "--quiet"], timeout=15):
            return
        _run(["git", "commit", "-m", message], timeout=30)
        _run(["git", "push"], timeout=60)
    except Exception:
        pass

def evaluate_model(game, model: PPO, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    return game.evaluate(model, episodes, deterministic)


def _train_single(
    model_id: str,
    game_name: str,
    cfg: TrainConfig,
    seed: int,
) -> Tuple[str, Dict[str, float], str, str, Optional[str]]:
    """Train one model line in isolation (separate process-friendly)."""
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if cfg.deterministic:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)

    affinity = _resolve_affinity_list(cfg.cpu_affinity, cfg.n_envs)
    if affinity:
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, affinity)  # type: ignore[attr-defined]
        except Exception:
            pass
    game = get_game(game_name)

    def _make_env_with_retry(attempts: int = 3):
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                env = make_vec_env(
                    lambda: game.make_env(render_mode=None),
                    n_envs=cfg.n_envs,
                    seed=seed,
                )
                # Prewarm to smooth jitters during initial rollout.
                env.reset()
                try:
                    if hasattr(env, "num_envs"):
                        actions = np.asarray([env.action_space.sample() for _ in range(env.num_envs)])
                    else:
                        actions = np.asarray(env.action_space.sample())
                    env.step(actions)
                except Exception:
                    env.close()
                    raise
                return env
            except Exception as exc:  # pragma: no cover - only on flaky init
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to create environments but no error captured.")

    env = _make_env_with_retry()
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    latest_path = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(cfg.log_dir, model_id) if _tensorboard_available() else None
    if tb_log_dir is None:
        print(f"[{model_id}] TensorBoard not installed; disabling tensorboard logging.")

    resume_path = cfg.resume_from or (latest_path if os.path.exists(latest_path) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"[{model_id}] Loading existing model from {resume_path} to continue training...")
        model = PPO.load(resume_path, env=env, device=cfg.device)
        if tb_log_dir is None:
            model.tensorboard_log = None
    else:
        print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    model.learn(
        total_timesteps=cfg.train_timesteps,
        reset_num_timesteps=False,
        progress_bar=_progress_bar_ready(suppress_log=True),
    )

    stamped_model_path: Optional[str] = None
    if cfg.checkpoint_interval > 0 and not cfg.no_checkpoint:
        stamped_model_path = os.path.join(cfg.model_dir, f"{model_id}_{timestamp}.zip")
        model.save(stamped_model_path)
        print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")

    model.save(latest_path)
    print(f"[{model_id}] Updated latest model: {latest_path}")

    os.environ["MAZE_EVAL_SEED_BASE"] = str(seed)
    metrics = evaluate_model(game, model, cfg.eval_episodes, deterministic=cfg.eval_deterministic or cfg.deterministic)
    print(f"[{model_id}] Avg reward over {cfg.eval_episodes} eval episodes: {metrics['avg_reward']:.3f}")

    env.close()
    return model_id, metrics, timestamp, latest_path, stamped_model_path


def _run_evo_pretrain(cfg: TrainConfig) -> None:
    if cfg.game != "maze":
        print("Evo pretraining is only supported for game=maze; skipping.")
        return
    maze_id = os.getenv("MAZE_ID", "")
    maze_dir = os.getenv("MAZE_DIR", "data/mazes")
    evo_cmd = [
        sys.executable,
        "scripts/mazes/evo_train.py",
        "--maze-dir",
        maze_dir,
        "--maze-id",
        maze_id,
        "--cycles",
        str(cfg.evo_cycles),
        "--population",
        str(cfg.evo_population),
        "--max-steps",
        str(cfg.evo_max_steps),
        "--sensor-range",
        str(cfg.evo_sensor_range),
        "--top-k",
        str(cfg.evo_top_k),
        "--batch-size",
        str(cfg.evo_batch_size),
        "--epochs",
        str(cfg.evo_epochs),
        "--learning-rate",
        str(cfg.evo_learning_rate),
        "--model-path",
        str(cfg.evo_model_path),
        "--move-start-on-success",
    ]
    print(f"Running evo pretrain: {' '.join(evo_cmd)}")
    subprocess.run(evo_cmd, check=False)


def main():
    cfg = parse_args()
    if cfg.game == "maze":
        maze_res = _maze_video_resolution()
        if maze_res:
            cfg.video_resolution = maze_res
        auto_goal = _auto_goal_enabled()
        if auto_goal:
            auto_settings = _auto_training_goal_from_metrics(cfg)
            if auto_settings:
                os.environ["MAZE_TRAIN_GOAL"] = auto_settings.get("train_goal", "")
                os.environ["MAZE_TRAIN_GOAL_FRACTION"] = auto_settings.get("train_goal_fraction", "")
                if auto_settings.get("train_goal"):
                    print(f"Auto training goal: {auto_settings.get('train_goal')}")
                if auto_settings.get("train_goal_fraction"):
                    print(f"Auto training goal fraction: {auto_settings.get('train_goal_fraction')}")
    if cfg.list_games:
        print("Available games:")
        for game in list_games():
            print(f"- {game.name}: {game.description}")
        return
    if cfg.export_config:
        export_path = Path(cfg.export_config)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(asdict(cfg), indent=2))
        print(f"Wrote resolved config to {export_path}")
        return
    game = get_game(cfg.game)
    model_prefix = cfg.model_prefix or game.model_prefix
    base_seed = cfg.base_seed or cfg.seed
    set_random_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    if not cfg.resume_from:
        best_from_metrics = _best_checkpoint_from_metrics(cfg, model_prefix)
        if best_from_metrics:
            cfg.resume_from = best_from_metrics
            print(f"Auto-resume: using best checkpoint from metrics {cfg.resume_from}")
        else:
            model_dir = Path(cfg.model_dir)
            latest_candidates = sorted(
                model_dir.glob(f"{model_prefix}_*_latest.zip"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if latest_candidates:
                cfg.resume_from = str(latest_candidates[0])
                print(f"Auto-resume: using latest checkpoint {cfg.resume_from}")

    print(
        f"Config: game={game.name}, profile={cfg.profile or 'none'}, train_steps={cfg.train_timesteps}, "
        f"iters_per_set={cfg.iterations_per_set}, max_cycles={cfg.max_cycles}, "
        f"video_steps={cfg.video_steps} (~{cfg.video_steps / cfg.target_fps:.1f}s @ {cfg.target_fps} fps)"
    )
    if cfg.status:
        _print_status(cfg)
        return
    _progress_bar_ready()

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, f"train_run_{run_timestamp}.jsonl")
    metrics_csv_path = Path(cfg.metrics_csv)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if cfg.stream_tensorboard and _tensorboard_available():
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "cycle_metrics", run_timestamp))
        except Exception:
            tb_writer = None

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", **asdict(cfg), "resolved_at": run_timestamp}) + "\n")
    if cfg.dry_run:
        env = game.make_env(render_mode=None)
        obs, info = env.reset()
        env.close()
        print("Dry run: environment initialized successfully. Exiting before training.")
        return
    if cfg.evo_first:
        _run_evo_pretrain(cfg)
    model_ids = [f"{model_prefix}_{i}" for i in range(cfg.iterations_per_set)]
    metric_fields = ["avg_reward", "avg_reward_ci", "avg_ep_len", "avg_ep_len_ci"] + list(game.extra_metrics)
    train_goal = os.getenv("MAZE_TRAIN_GOAL", "").strip()
    train_goal_fraction = os.getenv("MAZE_TRAIN_GOAL_FRACTION", "").strip()

    failure_detected = False
    best_score = float("-inf")
    best_overall_id: Optional[str] = None
    best_overall_score = float("-inf")
    last_combined_video: Optional[str] = None
    last_eval_video: Optional[str] = None
    no_improve_cycles = 0
    max_video_frames = cfg.max_video_frames
    top_checkpoints: List[Tuple[float, str]] = []
    stop_reason = "max_cycles"
    last_avg_by_model: Dict[str, float] = {}
    last_avg_steps_by_model: Dict[str, float] = {}
    cycle_reports: List[Dict[str, Any]] = []
    best_checkpoint_path: Optional[str] = None

    cycle = 0
    run_started_at = datetime.now()
    cycle_durations: List[float] = []
    initial_resume = cfg.resume_from if (cfg.resume_from and os.path.exists(cfg.resume_from)) else None
    while not failure_detected and cycle < cfg.max_cycles:
        combined_frames_per_model: List[List[np.ndarray]] = []
        segments_by_model: Dict[str, List[np.ndarray]] = {}
        all_grid_frames: List[np.ndarray] = []
        scores: List[Tuple[str, float]] = []
        metrics_list: List[Tuple[str, Dict[str, float], str]] = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cycle += 1
        print(f"\n=== Cycle {cycle} / {cfg.max_cycles} ===")
        cycle_started_at = datetime.now()
        print(f"[cycle {cycle}] start: {cycle_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        start_cycle = time.time()
        if cfg.game == "maze":
            auto_goal = _auto_goal_enabled()
            if auto_goal:
                auto_settings = _auto_training_goal_from_metrics(cfg)
                if auto_settings:
                    os.environ["MAZE_TRAIN_GOAL"] = auto_settings.get("train_goal", "")
                    os.environ["MAZE_TRAIN_GOAL_FRACTION"] = auto_settings.get("train_goal_fraction", "")
                    if auto_settings.get("train_goal"):
                        print(f"[cycle {cycle}] Auto training goal: {auto_settings.get('train_goal')}")
                    if auto_settings.get("train_goal_fraction"):
                        print(f"[cycle {cycle}] Auto training goal fraction: {auto_settings.get('train_goal_fraction')}")
                train_goal = os.getenv("MAZE_TRAIN_GOAL", "").strip()
                train_goal_fraction = os.getenv("MAZE_TRAIN_GOAL_FRACTION", "").strip()

        futures: List[concurrent.futures.Future] = []
        seed_by_future: Dict[concurrent.futures.Future, int] = {}
        seed_by_model: Dict[str, int] = {}
        try:
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                for model_id in model_ids:
                    target_latest = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
                    if best_checkpoint_path != target_latest:
                        try:
                            shutil.copy2(best_checkpoint_path, target_latest)
                        except OSError:
                            pass
            elif initial_resume:
                for model_id in model_ids:
                    target_latest = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
                    if initial_resume != target_latest:
                        try:
                            shutil.copy2(initial_resume, target_latest)
                        except OSError:
                            pass
            worker_timeout_env = os.getenv("TRAIN_WORKER_TIMEOUT", "").strip()
            try:
                worker_timeout = int(worker_timeout_env) if worker_timeout_env else 1800
            except ValueError:
                worker_timeout = 1800
            no_multiproc = os.getenv("TRAIN_NO_MULTIPROC", "").strip().lower() in {"1", "true", "yes", "on"}
            allow_win_multi_env = os.getenv("TRAIN_ALLOW_MULTIPROC_WIN", "").strip().lower()
            if allow_win_multi_env:
                allow_win_multi = allow_win_multi_env in {"1", "true", "yes", "on"}
            else:
                allow_win_multi = os.name == "nt"
            use_threads_env = os.getenv("TRAIN_USE_THREADS_WIN", "").strip().lower()
            use_threads_win = use_threads_env in {"1", "true", "yes", "on"} if use_threads_env else True
            if os.name == "nt" and allow_win_multi and not no_multiproc:
                if use_threads_win:
                    print("[cycle] Windows parallelism enabled via threads (set TRAIN_USE_THREADS_WIN=0 to use processes).")
                else:
                    print("[cycle] Windows multiprocessing enabled (set TRAIN_ALLOW_MULTIPROC_WIN=0 to disable).")
            if os.name == "nt" and not no_multiproc and not allow_win_multi:
                print("[cycle] Windows detected; forcing single-process training to avoid spawn/import crashes.")
                print("[cycle] Set TRAIN_ALLOW_MULTIPROC_WIN=1 to override.")
                no_multiproc = True
            if no_multiproc:
                for idx, model_id in enumerate(model_ids):
                    derived_seed = base_seed + idx + cycle
                    try:
                        model_id, metrics, stamp, latest_path, stamped_path = _train_single(
                            model_id=model_id,
                            game_name=game.name,
                            cfg=cfg,
                            seed=derived_seed,
                        )
                    except Exception as exc:
                        if cfg.worker_watchdog:
                            print(f"[watchdog] Worker failed: {exc}; continuing without this model.")
                            print(traceback.format_exc())
                            continue
                        raise
                    best_dist = metrics.get("best_dist")
                    goal_rate = float(metrics.get("goal_reached_rate") or 0.0)
                    if best_dist is None:
                        score = float("-inf")
                    else:
                        score = -float(best_dist)
                    scores.append((model_id, score))
                    metrics_list.append((model_id, metrics, latest_path))
                    if stamped_path:
                        metrics_list[-1] = (model_id, metrics, stamped_path)
                    timestamp = stamp  # use last reported for video naming
                    seed_by_model[model_id] = derived_seed
                    print(f"[{model_id}] Training done; seed={derived_seed}")
            else:
                if os.name == "nt" and allow_win_multi and use_threads_win:
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.iterations_per_set)
                else:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=cfg.iterations_per_set,
                        mp_context=mp.get_context("spawn"),
                    )
                force_shutdown = False
                try:
                    for idx, model_id in enumerate(model_ids):
                        derived_seed = base_seed + idx + cycle
                        future = executor.submit(
                            _train_single,
                            model_id=model_id,
                            game_name=game.name,
                            cfg=cfg,
                            seed=derived_seed,
                        )
                        futures.append(future)
                        seed_by_future[future] = derived_seed

                    try:
                        for future in concurrent.futures.as_completed(futures, timeout=worker_timeout):
                            try:
                                model_id, metrics, stamp, latest_path, stamped_path = future.result()
                            except Exception as exc:
                                if cfg.worker_watchdog:
                                    print(f"[watchdog] Worker failed: {exc}; continuing without this model.")
                                    tb = future.exception()
                                    if tb:
                                        print(traceback.format_exc())
                                    continue
                                raise
                            best_dist = metrics.get("best_dist")
                            goal_rate = float(metrics.get("goal_reached_rate") or 0.0)
                            if best_dist is None:
                                score = float("-inf")
                            else:
                                score = -float(best_dist)
                            scores.append((model_id, score))
                            metrics_list.append((model_id, metrics, latest_path))
                            if stamped_path:
                                metrics_list[-1] = (model_id, metrics, stamped_path)
                            timestamp = stamp  # use last reported for video naming
                            seed_used = seed_by_future.get(future, base_seed)
                            seed_by_model[model_id] = seed_used
                            print(f"[{model_id}] Training done; seed={seed_used}")
                    except concurrent.futures.TimeoutError:
                        print(f"[watchdog] Worker(s) timed out after {worker_timeout}s; canceling remaining.")
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        force_shutdown = True
                finally:
                    executor.shutdown(wait=not force_shutdown, cancel_futures=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received; stopping training and canceling workers.")
            for future in futures:
                future.cancel()
            failure_detected = True
            stop_reason = "interrupted"
            break

        if cfg.video_steps > 0 and metrics_list:
            scored_candidates = []
            for model_id, metrics, _ in metrics_list:
                best_dist = metrics.get("best_dist")
                goal_rate = float(metrics.get("goal_reached_rate") or 0.0)
                if best_dist is None:
                    score = float("-inf")
                else:
                    score = -float(best_dist)
                scored_candidates.append((model_id, score, goal_rate))
            best_id = max(scored_candidates, key=lambda t: t[1])[0] if scored_candidates else None
            capture_models: List[Tuple[str, str, str]] = []
            for model_id, metrics, model_path in metrics_list:
                best_dist = metrics.get("best_dist")
                dist_text = "--"
                if best_dist is not None:
                    try:
                        dist_text = f"{float(best_dist):.1f}"
                    except (TypeError, ValueError):
                        dist_text = str(best_dist)
                capture_models.append(
                    (
                        model_id,
                        model_path,
                        f"{model_id} | r {metrics.get('avg_reward', 0.0):.2f} | dist {dist_text}",
                    )
                )

            for label, model_path, overlay in capture_models:
                try:
                    model = PPO.load(model_path, device="cpu")
                    variant = None
                    seed_for_video = seed_by_model.get(label)
                    if label in model_ids:
                        idx = model_ids.index(label)
                        variant = idx
                        if seed_for_video is None:
                            seed_for_video = base_seed + idx + cycle
                    overlay_text = overlay
                    if seed_for_video is not None:
                        overlay_text = f"{overlay} | seed {seed_for_video}"
                    segment = record_video_segment(
                        game,
                        model,
                        steps=cfg.video_steps,
                        overlay_text=overlay_text,
                        resolution=_parse_resolution(cfg.video_resolution),
                        variant=variant,
                        seed=seed_for_video,
                    )
                    if segment:
                        combined_frames_per_model.append(segment)
                        segments_by_model[label] = segment
                    print(f"[{label}] Added {len(segment)} frames.")
                except Exception as exc:
                    print(f"[{label}] Video capture failed: {exc}")
        else:
            print("Video capture disabled or no models; skipping video accumulation.")

        if combined_frames_per_model and any(combined_frames_per_model):
            grid_frames = build_grid_frames(combined_frames_per_model)
            for frame in grid_frames:
                if len(all_grid_frames) >= max_video_frames:
                    break
                all_grid_frames.append(frame)
            print(f"Accumulated {len(all_grid_frames)} frames toward combined video (max {max_video_frames}).")
        else:
            print("No frames captured this cycle; skipping video accumulation.")

        if scores:
            best_id, best_score_cycle = max(scores, key=lambda t: t[1])
            best_latest = os.path.join(cfg.model_dir, f"{best_id}_latest.zip")
            best_checkpoint_path = next((p for mid, _, p in metrics_list if mid == best_id), best_latest)
            best_metrics = next((metrics for mid, metrics, _ in metrics_list if mid == best_id), {})
            for model_id in model_ids:
                target_latest = os.path.join(cfg.model_dir, f"{model_id}_latest.zip")
                if os.path.exists(best_latest) and best_latest != target_latest:
                    shutil.copy2(best_latest, target_latest)
            # checkpoint pruning
            top_checkpoints.append((best_score_cycle, best_checkpoint_path))
            top_checkpoints = sorted(top_checkpoints, key=lambda t: t[0], reverse=True)
            for idx, (_, path) in enumerate(top_checkpoints):
                if idx >= cfg.top_k_checkpoints and os.path.exists(path) and not path.endswith("_latest.zip"):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            top_checkpoints = top_checkpoints[: cfg.top_k_checkpoints]

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "cycle",
                            "cycle": cycle,
                            "scores": scores,
                            "frames": len(all_grid_frames),
                            "timestamp": timestamp,
                            "seeds": [base_seed + i + cycle for i in range(len(model_ids))],
                        }
                    )
                    + "\n"
                )
            _git_commit_artifacts(cfg, f"Update training outputs (cycle {cycle})")

            cycle_seconds = max(0.0, time.time() - start_cycle)
            cycle_durations.append(cycle_seconds)
            avg_cycle = sum(cycle_durations) / len(cycle_durations)
            remaining = max(0, cfg.max_cycles - cycle)
            eta_seconds = avg_cycle * remaining
            eta_at = datetime.now() + timedelta(seconds=eta_seconds)

            metrics_csv_exists = metrics_csv_path.exists()
            def _model_index(mid: str) -> int:
                try:
                    return int(mid.rsplit("_", 1)[-1])
                except (ValueError, AttributeError):
                    return -1

            def _rank_by(key: str, reverse: bool = False) -> Dict[str, int]:
                scored = []
                for mid, metrics, _ in metrics_list:
                    value = metrics.get(key)
                    if value is None or value == "":
                        continue
                    scored.append((mid, value))
                scored.sort(key=lambda item: item[1], reverse=reverse)
                ranks: Dict[str, int] = {}
                for idx, (mid, _) in enumerate(scored, start=1):
                    ranks[mid] = idx
                return ranks

            rank_avg_reward = _rank_by("avg_reward", reverse=True)
            rank_best_dist = _rank_by("best_dist", reverse=False)
            rank_goal_rate = _rank_by("goal_reached_rate", reverse=True)
            rank_best_progress = _rank_by("best_progress", reverse=True)
            rank_avg_steps = _rank_by("avg_steps", reverse=False)

            with metrics_csv_path.open("a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=[
                            "cycle",
                            "model_id",
                            "model_index",
                            *metric_fields,
                            "rank_avg_reward",
                            "rank_best_dist",
                            "rank_goal_rate",
                            "rank_best_progress",
                            "rank_avg_steps",
                            "delta_reward",
                            "delta_avg_steps",
                            "train_goal",
                            "train_goal_fraction",
                            "train_timesteps",
                            "iterations_per_set",
                            "eval_episodes",
                        "n_envs",
                        "n_steps",
                        "batch_size",
                        "n_epochs",
                        "video_steps",
                        "max_video_seconds",
                        "target_fps",
                        "cycle_start",
                        "cycle_duration_s",
                        "eta_end",
                        "timestamp",
                        "run_timestamp",
                        "game",
                    ],
                )
                if not metrics_csv_exists:
                    writer.writeheader()
                    for model_id, metrics, _ in metrics_list:
                        delta_reward = None
                        if cfg.metrics_deltas and model_id in last_avg_by_model:
                            delta_reward = metrics.get("avg_reward", 0.0) - last_avg_by_model[model_id]
                        last_avg_by_model[model_id] = metrics.get("avg_reward", 0.0)
                        delta_avg_steps = None
                        if cfg.metrics_deltas and model_id in last_avg_steps_by_model:
                            delta_avg_steps = metrics.get("avg_steps", 0.0) - last_avg_steps_by_model[model_id]
                        last_avg_steps_by_model[model_id] = metrics.get("avg_steps", 0.0)
                        row = {"cycle": cycle, "model_id": model_id}
                        row["model_index"] = _model_index(model_id)
                        for field in metric_fields:
                            row[field] = metrics.get(field, "")
                        row["rank_avg_reward"] = rank_avg_reward.get(model_id, "")
                        row["rank_best_dist"] = rank_best_dist.get(model_id, "")
                        row["rank_goal_rate"] = rank_goal_rate.get(model_id, "")
                        row["rank_best_progress"] = rank_best_progress.get(model_id, "")
                        row["rank_avg_steps"] = rank_avg_steps.get(model_id, "")
                        row["delta_reward"] = delta_reward if delta_reward is not None else ""
                        row["delta_avg_steps"] = delta_avg_steps if delta_avg_steps is not None else ""
                    row["train_goal"] = train_goal
                    row["train_goal_fraction"] = train_goal_fraction
                    row["train_timesteps"] = cfg.train_timesteps
                    row["iterations_per_set"] = cfg.iterations_per_set
                    row["eval_episodes"] = cfg.eval_episodes
                    row["n_envs"] = cfg.n_envs
                    row["n_steps"] = cfg.n_steps
                    row["batch_size"] = cfg.batch_size
                    row["n_epochs"] = cfg.n_epochs
                    row["video_steps"] = cfg.video_steps
                    row["max_video_seconds"] = cfg.max_video_seconds
                    row["target_fps"] = cfg.target_fps
                    row["cycle_start"] = cycle_started_at.strftime("%Y-%m-%d %H:%M:%S")
                    row["cycle_duration_s"] = f"{cycle_seconds:.2f}"
                    row["eta_end"] = eta_at.strftime("%Y-%m-%d %H:%M:%S")
                    row["timestamp"] = timestamp
                    row["run_timestamp"] = run_timestamp
                    row["game"] = game.name
                    writer.writerow(row)

            if best_score_cycle > best_score + cfg.improvement_threshold:
                best_score = best_score_cycle
                best_overall_id = best_id
                best_overall_score = best_score_cycle
                no_improve_cycles = 0
            else:
                no_improve_cycles += 1
            if all_grid_frames:
                annotated = _add_overlay(all_grid_frames[-1], f"Next base model: {best_id}")
                all_grid_frames.append(annotated)
            elapsed = time.time() - start_cycle
            print(
                f"Best model this cycle: {best_id} (avg reward {best_score_cycle:.3f}); "
                f"elapsed {elapsed:.1f}s; no_improve={no_improve_cycles}"
            )
            print(
                f"[cycle {cycle}] duration: {cycle_seconds/60:.1f} min | "
                f"avg: {avg_cycle/60:.1f} min | "
                f"eta: {eta_seconds/60:.1f} min (est end {eta_at.strftime('%Y-%m-%d %H:%M:%S')})"
            )
            cycle_reports.append(
                {
                    "cycle": cycle,
                    "scores": scores,
                    "best_id": best_id,
                    "best_score": best_score_cycle,
                    "timestamp": timestamp,
                    "seeds": [base_seed + i + cycle for i in range(len(model_ids))],
                }
            )
        else:
            print("No scores recorded; cannot propagate best model.")

        if cfg.early_stop_patience > 0 and no_improve_cycles >= cfg.early_stop_patience:
            print(f"No improvement for {no_improve_cycles} cycles; stopping early.")
            stop_reason = "early_stop"
            break
        if all_grid_frames:
            overlay_text = ""
            if scores:
                overlay_parts = [f"Cycle {cycle} best {best_id}"]
                if "win_rate" in best_metrics:
                    overlay_parts.append(f"win {best_metrics.get('win_rate', 0.0):.2f}")
                if "avg_ep_len" in best_metrics:
                    overlay_parts.append(f"len {best_metrics.get('avg_ep_len', 0.0):.1f}")
                overlay_text = " | ".join(overlay_parts)
            combined_video_path = Path(cfg.video_dir) / f"{model_prefix}_combined_{timestamp}_seed{base_seed}.mp4"
            if _safe_write_video(all_grid_frames, combined_video_path, cfg.target_fps, final_overlay=overlay_text):
                print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
                last_combined_video = str(combined_video_path)
        else:
            print("No frames captured; combined video not written.")

        if cfg.long_eval_video_steps and best_checkpoint_path and os.path.exists(best_checkpoint_path):
            try:
                best_model = PPO.load(best_checkpoint_path, device=cfg.device)
                frames = record_video_segment(
                    game,
                    best_model,
                    steps=cfg.long_eval_video_steps,
                    overlay_text=f"{best_id} long eval",
                    resolution=_parse_resolution(cfg.video_resolution),
                )
                if frames:
                    eval_path = Path(cfg.video_dir) / f"{best_id}_eval_{timestamp}_seed{base_seed}.mp4"
                    if _safe_write_video(frames, eval_path, cfg.target_fps, final_overlay="Long eval clip"):
                        print(f"[{best_id}] Saved extended eval video: {eval_path}")
                        last_eval_video = str(eval_path)
            except Exception as exc:  # pragma: no cover - non-critical
                print(f"Could not record extended eval video: {exc}")

        if cfg.individual_videos and segments_by_model:
            for model_id, frames in segments_by_model.items():
                if not frames:
                    continue
                indiv_path = Path(cfg.video_dir) / f"{model_id}_{timestamp}_seed{base_seed}.mp4"
                if _safe_write_video(frames, indiv_path, cfg.target_fps):
                    print(f"[{model_id}] Saved individual video: {indiv_path}")
        if segments_by_model:
            for model_id, frames in segments_by_model.items():
                if not frames or not (model_id.endswith("_best") or model_id.endswith("_solved")):
                    continue
                tagged_path = Path(cfg.video_dir) / f"{model_id}_{timestamp}_seed{base_seed}.mp4"
                if _safe_write_video(frames, tagged_path, cfg.target_fps):
                    print(f"[{model_id}] Saved tagged video: {tagged_path}")

        if tb_writer:
            for model_id, metrics, _ in metrics_list:
                tb_writer.add_scalar(f"{model_id}/avg_reward", metrics.get("avg_reward", 0.0), cycle)
                tb_writer.add_scalar(f"{model_id}/win_rate", metrics.get("win_rate", 0.0), cycle)

    if tb_writer:
        tb_writer.close()

    summary = {
        "event": "summary",
        "game": game.name,
        "model_prefix": model_prefix,
        "best_model": best_overall_id,
        "best_score": best_overall_score,
        "metrics_csv": cfg.metrics_csv,
        "video_dir": cfg.video_dir,
        "model_dir": cfg.model_dir,
        "last_combined_video": last_combined_video,
        "last_eval_video": last_eval_video,
        "run_timestamp": run_timestamp,
        "stop_reason": stop_reason,
        "run_started_at": run_started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "run_ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cycles": cycle_reports,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    report_json = Path(cfg.log_dir) / f"run_report_{run_timestamp}.json"
    report_json.write_text(json.dumps({"config": asdict(cfg), "summary": summary}, indent=2))
    report_html = Path(cfg.log_dir) / f"run_report_{run_timestamp}.html"
    report_html.write_text(
        f"<html><body><h1>Training Run {run_timestamp}</h1>"
        f"<p>Game: {game.name}</p>"
        f"<p>Best model: {best_overall_id or 'n/a'} (avg reward {best_overall_score:.3f})</p>"
        f"<p>Stop reason: {stop_reason}</p>"
        f"<p>Metrics CSV: {cfg.metrics_csv}</p>"
        f"<p>Last combined video: {last_combined_video or 'n/a'}</p>"
        f"<h2>Cycles</h2>"
        + "".join(
            f"<div><strong>Cycle {c['cycle']}</strong>: best {c['best_id']} @ {c['best_score']:.3f} (ts {c['timestamp']})</div>"
            for c in cycle_reports
        )
        + "</body></html>"
    )
    print("\n=== Run Summary ===")
    print(f"Best model: {best_overall_id or 'n/a'} (avg reward {best_overall_score:.3f})")
    print(f"Stop reason: {stop_reason}")
    print(f"Metrics CSV: {cfg.metrics_csv}")
    if last_combined_video:
        print(f"Last combined video: {last_combined_video}")
    print(f"Models dir: {cfg.model_dir} | Videos dir: {cfg.video_dir}")
    if _tensorboard_available():
        print(f"TensorBoard available: run `tensorboard --logdir {cfg.log_dir}`")
    else:
        print("TensorBoard not installed; install tensorboard to view training curves.")

if __name__ == "__main__":
    main()
