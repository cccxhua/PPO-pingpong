"""
A1 Table Tennis Deployment Inference Script.

Standalone inference loop for deploying the trained table tennis policy
on the Unitree A1 robot arm. Loads exported policy (JIT/ONNX), implements
observation assembly, phase-based motion state machine, and outputs
joint position targets at 50Hz.

Coordinate convention: +X points from robot toward opponent.
Robot is at -X side (ROBOT_POS.x = -1.7), opponent at +X side.
Ball approaches in -X direction (vx < 0).

Model: 57D obs → [512,256,128] ELU → 8D action
  action[0:7] = residual joint position (scaled by 0.05)
  action[7]   = phase speed (mapped to [0.85, 1.15])

Usage:
    python a1_table_tennis_deploy.py \
        --policy /path/to/exported/policy.pt \
        [--checkpoint /path/to/model_6000.pt]  # alternative: raw checkpoint
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants from training env_cfg (A1, ROBOT_SIDE = -1)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MOTION_DIR = os.path.join(
    PROJECT_ROOT,
    "unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/forehand",
)
DEFAULT_MOTION_FILES = [
    os.path.join(MOTION_DIR, "forehand_middle_a1_whip.npz"),
    os.path.join(MOTION_DIR, "forehand_left_a1.npz"),
    os.path.join(MOTION_DIR, "forehand_right_a1.npz"),
]

RIGHT_ARM_JOINT_NAMES = [
    "joint_yb_1", "joint_yb_2", "joint_yb_3", "joint_yb_4",
    "joint_yb_5", "joint_yb_6", "joint_yb_7",
]

DEFAULT_JOINT_POS = {
    "joint_yb_1": 1.769, "joint_yb_2": -0.762, "joint_yb_3": -1.863,
    "joint_yb_4": 1.445, "joint_yb_5": 0.206, "joint_yb_6": -0.827,
    "joint_yb_7": 1.043,
}

JOINT_LIMITS = {
    "joint_yb_1": (-1.04, 3.14),
    "joint_yb_2": (-3.14, 0.26),
    "joint_yb_3": (-2.758, 2.758),
    "joint_yb_4": (-1.92, 1.92),
    "joint_yb_5": (-2.758, 2.758),
    "joint_yb_6": (-1.57, 1.57),
    "joint_yb_7": (-2.758, 2.758),
}

RESIDUAL_SCALE = np.array([0.05] * 7, dtype=np.float32)
PHASE_SPEED_MIN = 0.85
PHASE_SPEED_MAX = 1.15

# PD gains (matching IsaacSim implicit actuator)
KP = {"joint_yb_1": 250.0, "joint_yb_2": 250.0, "joint_yb_3": 250.0,
      "joint_yb_4": 120.0, "joint_yb_5": 120.0, "joint_yb_6": 120.0, "joint_yb_7": 120.0}
KD = {"joint_yb_1": 1.0, "joint_yb_2": 1.0, "joint_yb_3": 1.0,
      "joint_yb_4": 0.5, "joint_yb_5": 0.5, "joint_yb_6": 0.5, "joint_yb_7": 0.5}

# A1: robot at -X side, facing +X (toward opponent)
ROBOT_POS = np.array([-1.7, 0.0, 0.0], dtype=np.float32)
ROBOT_X = -1.5  # effective x for ball prediction

STEP_DT = 0.02  # 50 Hz control
HIT_PHASE = 0.54
BALL_ARRIVE_TIME_EST = 0.51
BALL_Y_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# MotionLoader
# ---------------------------------------------------------------------------

class MotionLoader:
    def __init__(self, motion_files: list[str]):
        self.motions = []
        for path in motion_files:
            assert os.path.isfile(path), f"Motion file not found: {path}"
            data = np.load(path, allow_pickle=True)
            fps = float(data["fps"])
            dof = data["upper_body_dof"].astype(np.float32)
            base_y = data["base_y"].astype(np.float32)
            n = dof.shape[0]
            dt = 1.0 / fps

            dof_vel = np.zeros_like(dof)
            dof_vel[:-1] = (dof[1:] - dof[:-1]) / dt
            dof_vel[-1] = dof_vel[-2]

            self.motions.append({
                "fps": fps,
                "dof": dof,
                "dof_vel": dof_vel,
                "base_y": base_y,
                "num_frames": n,
                "duration": n / fps,
            })
        self.num_motions = len(self.motions)
        self.num_dof = self.motions[0]["dof"].shape[1]

    def get_reference(self, phase: float, motion_id: int):
        motion = self.motions[motion_id]
        n = motion["num_frames"]
        frame_f = phase * (n - 1)
        lo = int(np.clip(frame_f, 0, n - 2))
        hi = lo + 1
        alpha = frame_f - lo
        ref_dof = (1 - alpha) * motion["dof"][lo] + alpha * motion["dof"][hi]
        ref_dof_vel = (1 - alpha) * motion["dof_vel"][lo] + alpha * motion["dof_vel"][hi]
        ref_base_y = (1 - alpha) * motion["base_y"][lo] + alpha * motion["base_y"][hi]
        return ref_dof, ref_dof_vel, ref_base_y

    def duration(self, motion_id: int) -> float:
        return self.motions[motion_id]["duration"]


# ---------------------------------------------------------------------------
# PolicyRunner
# ---------------------------------------------------------------------------

class PolicyRunner:
    def __init__(self, policy_path: str = None, checkpoint_path: str = None, device: str = "cpu"):
        self.device = device
        if policy_path and os.path.exists(policy_path):
            if policy_path.endswith(".onnx"):
                import onnxruntime as ort
                self.session = ort.InferenceSession(policy_path)
                self.mode = "onnx"
                self.input_name = self.session.get_inputs()[0].name
            else:
                self.model = torch.jit.load(policy_path, map_location=device)
                self.model.eval()
                self.mode = "jit"
        elif checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)
            self.mode = "raw"
        else:
            raise FileNotFoundError("Must provide either --policy or --checkpoint")

    def _load_from_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        self.actor = nn.Sequential(
            nn.Linear(57, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 8),
        )
        actor_sd = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor_sd)
        self.actor.eval()
        self.actor.to(self.device)

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "jit":
            t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            out = self.model(t)
            return out.squeeze(0).cpu().numpy()
        elif self.mode == "onnx":
            out = self.session.run(None, {self.input_name: obs[np.newaxis].astype(np.float32)})
            return out[0].squeeze(0)
        else:
            t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            out = self.actor(t)
            return out.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Ball trajectory prediction (matches observations.py exactly)
# A1 coordinate: +X toward opponent, robot at -X, ball approaches in -X
# ---------------------------------------------------------------------------

def predict_ball_hit_point(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    robot_x: float = ROBOT_X,
    table_z: float = 0.76,
    restitution: float = 0.9,
    gravity: float = 9.81,
) -> tuple[float, float, float]:
    """Predict where the ball will be when reaching robot_x.

    Returns: (predicted_y, predicted_z, time_to_hit)
    """
    bx, by, bz = ball_pos
    vx, vy, vz = ball_vel

    # Ball not approaching: vx must be negative (toward -X where robot is)
    if vx >= -0.1:
        return 0.0, 0.0, 3.0

    dx = robot_x - bx
    # A1: on_own_side means ball is on our half (-X side of net)
    on_own_side = bx < 0.0
    is_rising = vz > 0.0
    already_bounced = on_own_side and is_rising and (bz < table_z + 0.3)

    if already_bounced:
        t_direct = max(dx / vx, 0.0)
        pred_y = by + vy * t_direct
        pred_z = bz + vz * t_direct - 0.5 * gravity * t_direct ** 2
        pred_t = min(t_direct, 3.0)
        pred_z = np.clip(pred_z, table_z, 2.0)
        return pred_y, pred_z, pred_t

    # Ball hasn't bounced yet — compute bounce time
    a = 0.5 * gravity
    b = -vz
    c = table_z - bz
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0, 0.0, 3.0

    t_bounce = (b + np.sqrt(disc)) / (2 * a)
    t_bounce = max(t_bounce, 0.0)

    x_bounce = bx + vx * t_bounce
    y_bounce = by + vy * t_bounce
    vz_after = abs(vz + (-gravity) * t_bounce) * restitution

    dx_after = robot_x - x_bounce
    if vx < -0.1:
        t_after = max(dx_after / vx, 0.0)
    else:
        t_after = 3.0

    pred_y = y_bounce + vy * t_after
    pred_z = table_z + vz_after * t_after - 0.5 * gravity * t_after ** 2
    pred_t = min(t_bounce + t_after, 3.0)
    pred_z = np.clip(pred_z, table_z, 2.0)
    return pred_y, pred_z, pred_t


def compute_ball_time_to_arrive(ball_pos: np.ndarray, ball_vel: np.ndarray, robot_x: float = ROBOT_X) -> float:
    """Estimated time for ball to reach robot_x."""
    bx = ball_pos[0]
    vx = ball_vel[0]
    # Ball approaching = vx < 0 (moving toward robot at -X)
    if vx < -0.1:
        return np.clip((robot_x - bx) / vx, 0.0, 3.0)
    return 3.0


def compute_ball_bounce_state(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    table_z: float = 0.76,
    robot_x: float = ROBOT_X,
) -> np.ndarray:
    """Ball bounce state: [has_bounced, is_rising, urgency]."""
    bx, _, bz = ball_pos
    vx, _, vz = ball_vel

    # A1: own side is -X (bx < 0)
    on_own_side = bx < 0.0
    is_rising = vz > 0.0
    has_bounced = float(on_own_side and is_rising and (bz < table_z + 0.3))
    rising = float(is_rising and on_own_side)

    dx = robot_x - bx
    if vx < -0.1:
        time_to_arrive = np.clip(dx / vx, 0.0, 3.0)
    else:
        time_to_arrive = 3.0
    urgency = np.clip(1.0 - time_to_arrive / 3.0, 0.0, 1.0)

    return np.array([has_bounced, rising, urgency], dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase State Machine (matches UpperBodyMotionCommand logic)
# ---------------------------------------------------------------------------

class PhaseStateMachine:
    def __init__(self, motion_loader: MotionLoader):
        self.motion = motion_loader
        self.phase = 0.0
        self.motion_id = 0
        self.phase_speed = 1.0
        self.ball_was_hit = False
        self.swing_done = False

    def reset(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Reset phase for new ball, aligned so hit_phase coincides with ball arrival."""
        duration = self.motion.duration(0)
        initial_phase = (HIT_PHASE - BALL_ARRIVE_TIME_EST / duration) % 1.0
        self.phase = np.clip(initial_phase, 0.0, 0.999)
        self.phase_speed = 1.0
        self.ball_was_hit = False
        self.swing_done = False
        self._select_motion(ball_pos, ball_vel)

    def _select_motion(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Select motion file based on predicted ball y."""
        # TODO: only middle motion is trained; force motion_id=0
        self.motion_id = 0

    def step(self, phase_speed_action: float):
        """Advance phase by one control step."""
        self.phase_speed = phase_speed_action
        prev_phase = self.phase
        duration = self.motion.duration(self.motion_id)
        self.phase += STEP_DT / duration * self.phase_speed

        # Check if we crossed hit_phase this step
        if HIT_PHASE >= 0:
            crossed = (prev_phase < HIT_PHASE <= self.phase) or \
                      (prev_phase > self.phase and (prev_phase < HIT_PHASE or self.phase >= HIT_PHASE))
            if crossed:
                self.swing_done = True

        # Phase wrapping/freeze logic
        if self.phase >= 1.0:
            if self.swing_done:
                self.phase = 0.0
                self.phase_speed = 0.0
            else:
                self.phase %= 1.0

    def get_reference(self):
        """Get current reference dof, dof_vel, base_y."""
        return self.motion.get_reference(self.phase, self.motion_id)

    def mark_hit(self):
        """Called when ball contact is detected."""
        self.ball_was_hit = True


# ---------------------------------------------------------------------------
# Observation Assembly (57D)
# ---------------------------------------------------------------------------

class ObservationAssembler:
    """Assembles the 57D observation vector.

    Layout:
        [0:15]   motion_command: ref_dof(7) + ref_dof_vel(7) + ref_base_y(1)
        [15:22]  joint_pos_rel: current_pos - default_pos (7)
        [22:29]  joint_vel: current joint velocities (7)
        [29:32]  ball_pos_relative: (ball_pos - robot_pos), clamp [-5, 5] (3)
        [32:35]  ball_vel_relative: (ball_vel - robot_vel), clamp [-10, 10] (3)
        [35:38]  ball_spin: ball angular velocity, clamp [-50, 50] (3)
        [38:41]  racket_pos: world position of paddle (3)
        [41:42]  motion_phase: current phase [0, 1] (1)
        [42:43]  ball_time_to_arrive: estimated time [0, 3] (1)
        [43:46]  ball_predicted_hit: (pred_y, pred_z, pred_t) (3)
        [46:49]  ball_bounce: (has_bounced, is_rising, urgency) (3)
        [49:57]  last_action: previous raw action output (8)
    """

    def __init__(self):
        self.default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )

    def compute(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        racket_pos: np.ndarray,
        phase_machine: PhaseStateMachine,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Assemble 57D observation.

        Args:
            joint_pos: Current right arm joint positions (7,)
            joint_vel: Current right arm joint velocities (7,)
            ball_pos: Ball world position (3,) [x, y, z]
            ball_vel: Ball linear velocity (3,) [vx, vy, vz]
            ball_ang_vel: Ball angular velocity (3,)
            racket_pos: Paddle world position (3,)
            phase_machine: Phase state machine instance
            last_action: Previous action output (8,)
        """
        # motion_command (15D)
        ref_dof, ref_dof_vel, ref_base_y = phase_machine.get_reference()
        ref_dof_vel_scaled = ref_dof_vel * phase_machine.phase_speed
        motion_cmd = np.concatenate([ref_dof, ref_dof_vel_scaled, [ref_base_y]])

        # joint_pos_rel (7D): relative to default
        joint_pos_rel = joint_pos - self.default_pos

        # joint_vel (7D)
        joint_vel_obs = joint_vel.copy()

        # ball_pos_relative (3D): relative to robot root
        ball_pos_rel = np.clip(ball_pos - ROBOT_POS, -5.0, 5.0)

        # ball_vel_relative (3D): robot is fixed, so robot_vel = 0
        ball_vel_rel = np.clip(ball_vel, -10.0, 10.0)

        # ball_spin (3D)
        ball_spin = np.clip(ball_ang_vel, -50.0, 50.0)

        # racket_pos (3D): world position minus env_origin (env_origin = 0 in deployment)
        racket_pos_obs = racket_pos.astype(np.float32)

        # motion_phase (1D)
        phase_obs = np.array([phase_machine.phase], dtype=np.float32)

        # ball_time_to_arrive (1D)
        time_arrive = compute_ball_time_to_arrive(ball_pos, ball_vel)
        time_arrive_obs = np.array([time_arrive], dtype=np.float32)

        # ball_predicted_hit (3D)
        pred_y, pred_z, pred_t = predict_ball_hit_point(ball_pos, ball_vel)
        ball_pred_obs = np.array([pred_y, pred_z, pred_t], dtype=np.float32)

        # ball_bounce (3D)
        ball_bounce_obs = compute_ball_bounce_state(ball_pos, ball_vel)

        # last_action (8D)
        last_action_obs = last_action.astype(np.float32)

        obs = np.concatenate([
            motion_cmd,         # 15
            joint_pos_rel,      # 7
            joint_vel_obs,      # 7
            ball_pos_rel,       # 3
            ball_vel_rel,       # 3
            ball_spin,          # 3
            racket_pos_obs,     # 3
            phase_obs,          # 1
            time_arrive_obs,    # 1
            ball_pred_obs,      # 3
            ball_bounce_obs,    # 3
            last_action_obs,    # 8
        ])
        assert obs.shape[0] == 57, f"Expected 57D obs, got {obs.shape[0]}"
        return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Action Processing
# ---------------------------------------------------------------------------

def process_action(
    action: np.ndarray,
    ref_dof: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Process raw policy output into joint targets and phase speed.

    Args:
        action: Raw 8D output from policy
        ref_dof: Current reference joint positions (7,)

    Returns:
        target_joint_pos: 7D joint position targets for PD controller
        phase_speed: Scalar phase speed value
    """
    residual = action[:7] * RESIDUAL_SCALE
    target_joint_pos = ref_dof + residual

    # Clamp to joint limits
    for i, name in enumerate(RIGHT_ARM_JOINT_NAMES):
        lo, hi = JOINT_LIMITS[name]
        target_joint_pos[i] = np.clip(target_joint_pos[i], lo, hi)

    # Phase speed: tanh-like mapping from action[7] to [speed_min, speed_max]
    raw_speed = action[7]
    phase_speed = PHASE_SPEED_MIN + (np.clip(raw_speed, -1, 1) + 1) * 0.5 * (PHASE_SPEED_MAX - PHASE_SPEED_MIN)

    return target_joint_pos, phase_speed


# ---------------------------------------------------------------------------
# Deployment Controller
# ---------------------------------------------------------------------------

class A1TableTennisController:
    """Main deployment controller. Subclass or modify get_sensor_data() for your hardware."""

    def __init__(self, policy_path: str = None, checkpoint_path: str = None, device: str = "cpu"):
        motion_files = DEFAULT_MOTION_FILES
        print(f"Loading {len(motion_files)} motion files...")
        self.motion = MotionLoader(motion_files)

        print("Loading policy...")
        self.policy = PolicyRunner(policy_path=policy_path, checkpoint_path=checkpoint_path, device=device)

        self.phase_machine = PhaseStateMachine(self.motion)
        self.obs_assembler = ObservationAssembler()
        self.last_action = np.zeros(8, dtype=np.float32)

        self._episode_active = False

    def reset(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Reset for new ball. Call when a new ball is launched."""
        self.phase_machine.reset(ball_pos, ball_vel)
        self.last_action[:] = 0.0
        self._episode_active = True

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        racket_pos: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Run one inference step at 50Hz.

        Args:
            joint_pos: Right arm joint positions (7,)
            joint_vel: Right arm joint velocities (7,)
            ball_pos: Ball world position (3,)
            ball_vel: Ball linear velocity (3,)
            ball_ang_vel: Ball angular velocity (3,)
            racket_pos: Paddle world position from FK or sensor (3,)

        Returns:
            target_joint_pos: Joint position commands for PD controller (7,)
            info: Dict with debug info (phase, motion_id, phase_speed, ref_dof)
        """
        # Assemble observation
        obs = self.obs_assembler.compute(
            joint_pos, joint_vel,
            ball_pos, ball_vel, ball_ang_vel,
            racket_pos, self.phase_machine, self.last_action,
        )

        # Run inference
        action = self.policy.infer(obs)
        action = np.clip(action, -1.0, 1.0)

        # Process action
        ref_dof, _, _ = self.phase_machine.get_reference()
        target_joint_pos, phase_speed = process_action(action, ref_dof)

        # Advance phase
        self.phase_machine.step(phase_speed)

        # Store action for next step
        self.last_action = action.copy()

        info = {
            "phase": self.phase_machine.phase,
            "motion_id": self.phase_machine.motion_id,
            "phase_speed": phase_speed,
            "ref_dof": ref_dof.copy(),
            "action_raw": action.copy(),
        }
        return target_joint_pos, info

    @property
    def is_swing_done(self) -> bool:
        return self.phase_machine.swing_done and self.phase_machine.phase_speed == 0.0


# ---------------------------------------------------------------------------
# Rally Controller (for continuous play with closed-loop phase)
# ---------------------------------------------------------------------------

class RallyState:
    IDLE = 0        # Ready position, waiting for ball
    TRACKING = 1    # Ball detected, phase running, preparing to swing
    SWINGING = 2    # In active swing (phase near HIT_PHASE)
    RECOVERING = 3  # Swing done, returning to ready position


class A1RallyController:
    """Wraps A1TableTennisController with rally logic and closed-loop phase.

    Usage (50Hz loop):
        controller = A1RallyController(policy_path=...)
        while True:
            ball_pos, ball_vel = perception.get_ball_state()
            targets = controller.update(joint_pos, joint_vel, ball_pos, ball_vel,
                                        ball_ang_vel, racket_pos)
            send_to_servo(targets)
    """

    RECOVER_DURATION = 0.5  # seconds to return to ready pose after swing

    def __init__(self, policy_path: str = None, checkpoint_path: str = None,
                 device: str = "cpu", known_latency: float = 0.04):
        self.inner = A1TableTennisController(
            policy_path=policy_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.known_latency = known_latency
        self.state = RallyState.IDLE
        self._recover_t = 0.0
        self._recover_start_pos = None
        self._default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._last_target = self._default_pos.copy()
        self._hit_count = 0

    def update(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        racket_pos: np.ndarray,
    ) -> np.ndarray:
        """Main update at 50Hz. Returns joint position targets (7,)."""

        if self.state == RallyState.IDLE:
            self._last_target = self._default_pos.copy()
            if self._ball_incoming(ball_pos, ball_vel):
                self._start_tracking(ball_pos, ball_vel)

        elif self.state == RallyState.TRACKING:
            target, info = self.inner.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            # Closed-loop phase correction
            self._correct_phase(ball_pos, ball_vel)
            self._last_target = target

            if info["phase"] > HIT_PHASE and self.inner.phase_machine.phase_speed > 0:
                self.state = RallyState.SWINGING

        elif self.state == RallyState.SWINGING:
            target, info = self.inner.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_target = target

            if self.inner.is_swing_done:
                self._hit_count += 1
                self._start_recovery(joint_pos)

        elif self.state == RallyState.RECOVERING:
            self._recover_t += STEP_DT
            alpha = min(1.0, self._recover_t / self.RECOVER_DURATION)
            # Smooth interpolation back to ready pose
            alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
            self._last_target = (
                (1 - alpha_smooth) * self._recover_start_pos +
                alpha_smooth * self._default_pos
            )
            if alpha >= 1.0:
                self.state = RallyState.IDLE

        return self._last_target

    def _ball_incoming(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
        """Detect if a ball is coming toward the robot (A1 at -X, ball from +X)."""
        return (
            ball_vel[0] < -1.0 and             # Moving toward robot (negative x)
            ball_pos[0] > ROBOT_X + 0.3 and    # Not already past us
            ball_pos[2] > 0.5                   # Above table level (not on ground)
        )

    def _start_tracking(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Start a new swing, compensating for known perception latency."""
        self.inner.reset(ball_pos, ball_vel)
        # Compensate for known latency: advance phase by the time already elapsed
        if self.known_latency > 0:
            duration = self.inner.phase_machine.motion.duration(
                self.inner.phase_machine.motion_id
            )
            self.inner.phase_machine.phase += self.known_latency / duration
        self.state = RallyState.TRACKING

    def _start_recovery(self, current_joint_pos: np.ndarray):
        """Begin smooth return to ready position."""
        self._recover_start_pos = current_joint_pos.copy()
        self._recover_t = 0.0
        self.state = RallyState.RECOVERING

    def _correct_phase(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Closed-loop phase correction based on real-time ball state."""
        # Ball must be approaching (vx < 0 for A1)
        if ball_vel[0] > -0.5:
            return

        pm = self.inner.phase_machine
        duration = pm.motion.duration(pm.motion_id)

        # Estimate remaining time to arrival
        t_remain = max(0.05, (ROBOT_X - ball_pos[0]) / ball_vel[0])

        # What phase SHOULD we be at, given the ball arrives in t_remain?
        ideal_phase = (HIT_PHASE - t_remain / duration) % 1.0

        # Compute phase error (wrapped to [-0.5, 0.5])
        error = ideal_phase - pm.phase
        if error > 0.5:
            error -= 1.0
        elif error < -0.5:
            error += 1.0

        # Soft correction: max ±0.003 per step (equivalent to ~±15% speed bias)
        correction = np.clip(error * 0.08, -0.003, 0.003)
        pm.phase += correction

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def current_state(self) -> str:
        names = {0: "IDLE", 1: "TRACKING", 2: "SWINGING", 3: "RECOVERING"}
        return names[self.state]

# ---------------------------------------------------------------------------
# Demo: offline test with synthetic ball trajectory
# ---------------------------------------------------------------------------

def demo_offline():
    """Demonstrate the controller with a synthetic ball trajectory (no hardware)."""
    parser = argparse.ArgumentParser(description="A1 Table Tennis Deploy")
    parser.add_argument("--policy", default=None, help="Path to exported policy.pt or policy.onnx")
    parser.add_argument("--checkpoint", default=None, help="Path to model_XXXX.pt checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to simulate")
    args = parser.parse_args()

    if args.policy is None and args.checkpoint is None:
        default_policy = os.path.join(
            PROJECT_ROOT,
            "logs/rsl_rl/a1_tabletennis/exported/policy.pt"
        )
        if os.path.exists(default_policy):
            args.policy = default_policy
        else:
            print("ERROR: No policy file specified. Use --policy or --checkpoint.")
            return

    controller = A1TableTennisController(
        policy_path=args.policy,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Synthetic ball: starts at opponent side (+X), moving toward robot (-X)
    ball_pos = np.array([0.35, 0.0, 1.10], dtype=np.float32)
    ball_vel = np.array([-3.0, 0.0, 0.2], dtype=np.float32)
    ball_ang_vel = np.zeros(3, dtype=np.float32)

    # Initial joint state at default
    joint_pos = np.array([DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32)
    joint_vel = np.zeros(7, dtype=np.float32)
    racket_pos = np.array([-1.5, 0.0, 1.1], dtype=np.float32)  # approximate

    controller.reset(ball_pos, ball_vel)

    print(f"\n{'Step':>4} | {'Phase':>5} | {'MID':>3} | {'PSpeed':>6} | Target Joints")
    print("-" * 80)

    gravity = 9.81
    for step in range(args.steps):
        target, info = controller.step(
            joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
        )

        if step % 5 == 0:
            tgt_str = " ".join(f"{t:+.3f}" for t in target)
            print(f"{step:4d} | {info['phase']:.3f} | {info['motion_id']:3d} | "
                  f"{info['phase_speed']:.3f} | {tgt_str}")

        # Simple ball physics (no bounce for demo)
        ball_pos += ball_vel * STEP_DT
        ball_vel[2] -= gravity * STEP_DT

        # Pretend joints track the target (simplified)
        joint_pos = 0.8 * joint_pos + 0.2 * target

        if controller.is_swing_done:
            print(f"\n[Step {step}] Swing complete. Resetting for next ball...")
            ball_pos = np.array([0.35, 0.02, 1.10], dtype=np.float32)
            ball_vel = np.array([-3.0, 0.1, 0.2], dtype=np.float32)
            controller.reset(ball_pos, ball_vel)

    print("\nDone.")


if __name__ == "__main__":
    demo_offline()
