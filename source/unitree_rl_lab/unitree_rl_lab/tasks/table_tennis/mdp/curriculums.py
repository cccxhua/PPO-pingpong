from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _lerp_range(
    easy: tuple[float, float], hard: tuple[float, float], t: float
) -> tuple[float, float]:
    return (
        easy[0] + (hard[0] - easy[0]) * t,
        easy[1] + (hard[1] - easy[1]) * t,
    )


def ball_difficulty_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    ramp_steps: int = 50_000,
    easy_x: tuple[float, float] = (1.0, 1.4),
    easy_y: tuple[float, float] = (-0.1, 0.1),
    easy_z: tuple[float, float] = (0.85, 1.05),
    easy_vx: tuple[float, float] = (0.2, 0.8),
    easy_vy: tuple[float, float] = (-0.1, 0.1),
    easy_vz: tuple[float, float] = (-1.0, 0.0),
    hard_x: tuple[float, float] = (0.3, 1.0),
    hard_y: tuple[float, float] = (-0.5, 0.5),
    hard_z: tuple[float, float] = (0.9, 1.2),
    hard_vx: tuple[float, float] = (1.5, 3.0),
    hard_vy: tuple[float, float] = (-0.3, 0.3),
    hard_vz: tuple[float, float] = (-4.0, -2.0),
) -> torch.Tensor:
    t = min(env.common_step_counter / ramp_steps, 1.0)

    new_params = {
        "x_range": _lerp_range(easy_x, hard_x, t),
        "y_range": _lerp_range(easy_y, hard_y, t),
        "z_range": _lerp_range(easy_z, hard_z, t),
        "vx_range": _lerp_range(easy_vx, hard_vx, t),
        "vy_range": _lerp_range(easy_vy, hard_vy, t),
        "vz_range": _lerp_range(easy_vz, hard_vz, t),
    }

    for term_name in ("reset_ball", "relaunch_ball"):
        try:
            cfg = env.event_manager.get_term_cfg(term_name)
        except ValueError:
            continue
        for key, val in new_params.items():
            cfg.params[key] = val
        env.event_manager.set_term_cfg(term_name, cfg)

    return torch.tensor(t, device=env.device)
