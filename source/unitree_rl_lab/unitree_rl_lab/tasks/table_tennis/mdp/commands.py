from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UpperBodyMotionLoader:

    def __init__(self, motion_files: list[str], device: str = "cpu", axis_flip_indices: list[int] | None = None):
        self.motions = []
        self.device = device

        for path in motion_files:
            assert os.path.isfile(path), f"Motion file not found: {path}"
            data = np.load(path, allow_pickle=True)

            fps = float(data["fps"])
            dof = torch.tensor(data["upper_body_dof"], dtype=torch.float32, device=device)
            base_y = torch.tensor(data["base_y"], dtype=torch.float32, device=device)
            joint_names = list(data["joint_names"])
            num_frames = dof.shape[0]

            if axis_flip_indices:
                for idx in axis_flip_indices:
                    dof[:, idx] = -dof[:, idx]

            dt = 1.0 / fps
            dof_vel = torch.zeros_like(dof)
            dof_vel[:-1] = (dof[1:] - dof[:-1]) / dt
            dof_vel[-1] = dof_vel[-2]

            base_y_vel = torch.zeros_like(base_y)
            base_y_vel[:-1] = (base_y[1:] - base_y[:-1]) / dt
            base_y_vel[-1] = base_y_vel[-2]

            self.motions.append({
                "fps": fps,
                "dof": dof,
                "dof_vel": dof_vel,
                "base_y": base_y,
                "base_y_vel": base_y_vel,
                "joint_names": joint_names,
                "num_frames": num_frames,
                "duration": num_frames / fps,
            })

        self.num_motions = len(self.motions)
        self.num_dof = self.motions[0]["dof"].shape[1]

    def get_reference(self, phase: torch.Tensor, motion_ids: torch.Tensor):
        num_envs = phase.shape[0]
        ref_dof = torch.zeros(num_envs, self.num_dof, device=self.device)
        ref_dof_vel = torch.zeros(num_envs, self.num_dof, device=self.device)
        ref_base_y = torch.zeros(num_envs, device=self.device)
        ref_base_y_vel = torch.zeros(num_envs, device=self.device)

        for i in range(self.num_motions):
            mask = motion_ids == i
            if not mask.any():
                continue

            motion = self.motions[i]
            n = motion["num_frames"]

            frame_f = phase[mask] * (n - 1)
            frame_lo = frame_f.long().clamp(0, n - 2)
            frame_hi = frame_lo + 1
            alpha = (frame_f - frame_lo.float()).unsqueeze(-1)

            ref_dof[mask] = (1 - alpha) * motion["dof"][frame_lo] + alpha * motion["dof"][frame_hi]
            ref_dof_vel[mask] = (1 - alpha) * motion["dof_vel"][frame_lo] + alpha * motion["dof_vel"][frame_hi]
            ref_base_y[mask] = (1 - alpha.squeeze(-1)) * motion["base_y"][frame_lo] + alpha.squeeze(-1) * motion["base_y"][frame_hi]
            ref_base_y_vel[mask] = (1 - alpha.squeeze(-1)) * motion["base_y_vel"][frame_lo] + alpha.squeeze(-1) * motion["base_y_vel"][frame_hi]

        return ref_dof, ref_dof_vel, ref_base_y, ref_base_y_vel


class UpperBodyMotionCommand(CommandTerm):
    cfg: UpperBodyMotionCommandCfg

    def __init__(self, cfg: UpperBodyMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.motion = UpperBodyMotionLoader(cfg.motion_files, device=self.device, axis_flip_indices=cfg.axis_flip_indices)

        raw_names = list(self.motion.motions[0]["joint_names"])
        all_robot_joints = self.robot.joint_names
        if raw_names[0] in all_robot_joints:
            robot_joint_names = raw_names
        else:
            robot_joint_names = [n + "_joint" for n in raw_names]
        self.upper_body_joint_ids = torch.tensor(
            self.robot.find_joints(robot_joint_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device,
        )

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._ref_dof = torch.zeros(self.num_envs, self.motion.num_dof, device=self.device)
        self._ref_dof_vel = torch.zeros(self.num_envs, self.motion.num_dof, device=self.device)
        self._ref_base_y = torch.zeros(self.num_envs, device=self.device)
        self._ref_base_y_vel = torch.zeros(self.num_envs, device=self.device)

        max_bins = max(int(m["num_frames"] / (m["fps"] * env.step_dt)) + 1 for m in self.motion.motions)
        self.bin_count = max_bins
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda ** i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)

        self.phase_speed = torch.ones(self.num_envs, device=self.device)
        self.ball_was_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.swing_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self._ref_dof, self._ref_dof_vel, self._ref_base_y.unsqueeze(-1)], dim=-1)

    @property
    def ref_dof(self) -> torch.Tensor:
        return self._ref_dof

    @property
    def ref_dof_vel(self) -> torch.Tensor:
        return self._ref_dof_vel

    @property
    def ref_base_y(self) -> torch.Tensor:
        return self._ref_base_y

    @property
    def ref_base_y_vel(self) -> torch.Tensor:
        return self._ref_base_y_vel

    def robot_upper_body_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.upper_body_joint_ids]

    def robot_upper_body_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.upper_body_joint_ids]

    def _update_references(self):
        self._ref_dof, self._ref_dof_vel, self._ref_base_y, self._ref_base_y_vel = (
            self.motion.get_reference(self.phase, self.motion_ids)
        )
        self._ref_dof_vel = self._ref_dof_vel * self.phase_speed.unsqueeze(-1)
        self._ref_base_y_vel = self._ref_base_y_vel * self.phase_speed

    def _update_metrics(self):
        cur_dof = self.robot_upper_body_joint_pos()
        cur_vel = self.robot_upper_body_joint_vel()
        cur_base_y = self.robot.data.root_pos_w[:, 1] - self._env.scene.env_origins[:, 1]

        self.metrics["error_joint_pos"] = torch.norm(cur_dof - self._ref_dof, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(cur_vel - self._ref_dof_vel, dim=-1)
        self.metrics["error_base_y"] = torch.abs(cur_base_y - self._ref_base_y)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.phase[env_ids] * self.bin_count).long(), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count).float()

        sampling_prob = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_prob = torch.nn.functional.pad(
            sampling_prob.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_prob = torch.nn.functional.conv1d(sampling_prob, self.kernel.view(1, 1, -1)).view(-1)
        sampling_prob = sampling_prob / sampling_prob.sum()

        sampled_bins = torch.multinomial(sampling_prob, len(env_ids), replacement=True)
        self.phase[env_ids] = (
            (sampled_bins.float() + torch.rand(len(env_ids), device=self.device)) / self.bin_count
        ).clamp(0.0, 0.999)

        self.motion_ids[env_ids] = torch.randint(0, self.motion.num_motions, (len(env_ids),), device=self.device)

        H = -(sampling_prob * (sampling_prob + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 0.0
        self.metrics["sampling_entropy"][:] = H_norm

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if self.cfg.freeze_phase:
            self.phase[env_ids] = self.cfg.hit_phase
            if self.cfg.match_ball_direction and self.motion.num_motions >= 3:
                self._assign_motion_by_ball(env_ids)
            else:
                self.motion_ids[env_ids] = torch.randint(
                    0, self.motion.num_motions, (len(env_ids),), device=self.device
                )
        elif self.cfg.hit_phase >= 0:
            self._phase_aligned_init(env_ids)
        else:
            self._adaptive_sampling(env_ids)
        self.phase_speed[env_ids] = 1.0
        self.ball_was_hit[env_ids] = False
        self.swing_done[env_ids] = False
        self._update_references()

        ref_dof_for_reset = self._ref_dof[env_ids]
        full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_joint_pos[:, self.upper_body_joint_ids] = ref_dof_for_reset

        full_joint_vel = torch.zeros_like(full_joint_pos)
        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        if not self.cfg.fixed_base:
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self._env.scene.env_origins[env_ids]
            root_state[:, 1] += self._ref_base_y[env_ids]
            noise = sample_uniform(*self.cfg.base_y_noise_range, (len(env_ids),), device=self.device)
            root_state[:, 1] += noise
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _phase_aligned_init(self, env_ids: Sequence[int]):
        """Initialize phase so the hit_phase is reached when ball arrives.

        phase advances at step_dt/duration * phase_speed per step.
        We want: initial_phase + ball_arrive_time / duration = hit_phase (mod 1.0)
        So: initial_phase = hit_phase - ball_arrive_time / duration
        """
        num = len(env_ids)
        duration = self.motion.motions[0]["duration"]
        arrive_time = self.cfg.ball_arrive_time_est

        target_phase = self.cfg.hit_phase
        noise = torch.randn(num, device=self.device) * self.cfg.hit_phase_noise
        initial_phase = (target_phase - arrive_time / duration + noise) % 1.0

        self.phase[env_ids] = initial_phase.clamp(0.0, 0.999)

        if self.cfg.match_ball_direction and self.motion.num_motions >= 3:
            self._assign_motion_by_ball(env_ids)
        else:
            self.motion_ids[env_ids] = torch.randint(0, self.motion.num_motions, (num,), device=self.device)

    def _assign_motion_by_ball(self, env_ids: Sequence[int]):
        """Select motion based on predicted ball y at robot position."""
        ball: RigidObject = self._env.scene["ball"]
        ball_pos = ball.data.root_pos_w[env_ids, :3] - self._env.scene.env_origins[env_ids]
        ball_vel = ball.data.root_lin_vel_w[env_ids]

        bx, by = ball_pos[:, 0], ball_pos[:, 1]
        vx, vy = ball_vel[:, 0], ball_vel[:, 1]

        robot_x = 1.5
        t_arrive = torch.where(vx > 0.1, (robot_x - bx) / vx, torch.full_like(bx, 1.0))
        predicted_y = by + vy * t_arrive

        # motion order: 0=middle, 1=left, 2=right
        ids = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        ids[predicted_y > self.cfg.ball_y_threshold] = 1
        ids[predicted_y < -self.cfg.ball_y_threshold] = 2
        self.motion_ids[env_ids] = ids

    def _update_command(self):
        if not self.cfg.freeze_phase:
            hit_phase = self.cfg.hit_phase
            prev_phase = self.phase.clone()
            for i in range(self.motion.num_motions):
                mask = self.motion_ids == i
                if mask.any():
                    duration = self.motion.motions[i]["duration"]
                    self.phase[mask] += self._env.step_dt / duration * self.phase_speed[mask]

            # 标记本轮挥拍已通过 hit_phase (含 wrap 跨越的情况)
            if hit_phase >= 0:
                no_wrap = (prev_phase <= self.phase) & (prev_phase < hit_phase) & (self.phase >= hit_phase)
                with_wrap = (prev_phase > self.phase) & ((prev_phase < hit_phase) | (self.phase >= hit_phase))
                self.swing_done |= no_wrap | with_wrap

            # phase 到 1.0:
            #   - 若已击球 (swing_done): 停在 PREP, 等下一个球
            #   - 否则: 正常 wrap 继续推进 (e.g. aligned_phase=0.95 起始时)
            wrapped = self.phase >= 1.0
            freeze = wrapped & self.swing_done
            self.phase[freeze] = 0.0
            self.phase_speed[freeze] = 0.0
            cont = wrapped & ~self.swing_done
            self.phase[cont] %= 1.0

        self._update_references()
        self._update_metrics()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UpperBodyMotionCommandCfg(CommandTermCfg):
    class_type: type = UpperBodyMotionCommand

    asset_name: str = MISSING
    motion_files: list[str] = MISSING

    base_y_noise_range: tuple[float, float] = (-0.02, 0.02)
    fixed_base: bool = False

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    hit_phase: float = -1.0
    hit_phase_noise: float = 0.05
    ball_arrive_time_est: float = 0.9

    match_ball_direction: bool = False
    ball_y_threshold: float = 0.05

    axis_flip_indices: list[int] | None = None

    freeze_phase: bool = False
    """If True, lock phase at hit_phase and skip phase advancement.

    Used when policy controls swing entirely via residual instead of
    riding the reference trajectory. The hit_phase frame becomes a
    fixed "ready pose" anchor.
    """


class BallHitTrackerCommand(CommandTerm):
    """Minimal command that only tracks ball_was_hit state.

    Used in pure-RL environments (no motion reference) so that
    track_ball_hit / ball_return_reward / ball_land_on_own_table
    can access ball_was_hit via env.command_manager.get_term(...).
    """

    cfg: "BallHitTrackerCommandCfg"

    def __init__(self, cfg: "BallHitTrackerCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.ball_was_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.ball_was_hit.float().unsqueeze(-1)

    def _resample_command(self, env_ids: Sequence[int]):
        self.ball_was_hit[env_ids] = False

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class BallHitTrackerCommandCfg(CommandTermCfg):
    class_type: type = BallHitTrackerCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
