#!/usr/bin/env python3
"""
ppo_inference_pingpong.py

A1 PPO 乒乓球策略推理节点，50Hz 控制循环。
订阅关节状态和球感知话题，运行策略推理，发布关节位置指令。

坐标约定: +X 由机器人指向对手（与训练一致），无需坐标变换。

订阅话题:
  /joint_states                (sensor_msgs/JointState)        — 右臂关节位置/速度
  /kalman/pingpong_pos         (geometry_msgs/PoseStamped)     — 球 3D 位置
  /kalman/pingpong_vel         (geometry_msgs/TwistStamped)    — 球 3D 速度 + 角速度
  /kalman/racket_pos           (geometry_msgs/PoseStamped)     — 球拍 3D 位置 (动捕)

发布话题:
  a1/sent_actions              (sensor_msgs/JointState)        — 推理输出的关节位置指令

注: ball_bounce 状态由 ball_pos + ball_vel 内部计算，无需外部话题。
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# PPO deploy modules (from a1_table_tennis_deploy.py)
# ---------------------------------------------------------------------------

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from a1_table_tennis_deploy import (
    MotionLoader,
    PolicyRunner,
    PhaseStateMachine,
    ObservationAssembler,
    A1TableTennisController,
    process_action,
    compute_ball_time_to_arrive,
    predict_ball_hit_point,
    compute_ball_bounce_state,
    DEFAULT_MOTION_FILES,
    RIGHT_ARM_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    ROBOT_POS,
    ROBOT_X,
    STEP_DT,
    HIT_PHASE,
)

# FK for racket position (same as sim2sim)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "unitree_rl_lab", "scripts", "sim2sim"))
try:
    from a1_play_mujoco import URDFForwardKinematics
except ImportError:
    # Fallback: inline minimal FK class if sim2sim not available
    URDFForwardKinematics = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 50.0  # 50Hz control loop
STEP_DT = 1.0 / CONTROL_HZ  # 覆盖 deploy 脚本中的 0.02，匹配实际控制频率

# /joint_states 中右臂关节名称 → 内部索引 (0~6)
RIGHT_ARM_SOURCE_NAMES = {
    "joint1-a1_r": 0,
    "joint2-a1_r": 1,
    "joint3-a1_r": 2,
    "joint4-a1_r": 3,
    "joint5-a1_r": 4,
    "joint6-a1_r": 5,
    "joint7-a1_r": 6,
}

# 推理输出中的关节名 (发布到 a1/sent_actions)
OUTPUT_JOINT_NAMES = [
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
]

# 默认策略路径
DEFAULT_POLICY_PATH = os.path.join(
    SCRIPT_DIR, "..", "..", "unitree_rl_lab", "logs", "rsl_rl",
    "a1_tabletennis", "exported", "policy.pt"
)
DEFAULT_POLICY_PATH = os.path.abspath(DEFAULT_POLICY_PATH)


# ---------------------------------------------------------------------------
# Rally State Machine
# ---------------------------------------------------------------------------

class RallyState:
    IDLE = 0        # 等待位姿，等球来
    BLENDING = 1    # 平滑过渡到参考动作起始帧
    TRACKING = 2    # 检测到球，phase 运行中
    SWINGING = 3    # 正在挥拍 (phase > HIT_PHASE)
    RECOVERING = 4  # 挥拍结束，回归等待位姿


class PPOPingPongNode(Node):
    """A1 PPO 乒乓球部署推理节点。"""

    RECOVER_DURATION = 0.5  # 回归等待位姿的时间(秒)
    BLEND_DURATION = 0.08   # IDLE→TRACKING 平滑过渡时间(秒)
    IDLE_RETURN_ALPHA = 0.02  # IDLE 状态下每步向 default_pos 指数平滑系数 (越小越慢)
    PHASE_CORRECTION_GAIN = 0.10  # 闭环 phase 校正比例增益
    PHASE_CORRECTION_MAX = 0.006  # 每步最大校正量 (50Hz 下约 ±30% speed bias)

    def __init__(self):
        super().__init__("ppo_inference_pingpong_a1")

        # ---- Parameters ----
        self.declare_parameter("policy_path", DEFAULT_POLICY_PATH)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("ball_incoming_vx_threshold", -1.0)
        self.declare_parameter("ball_incoming_x_offset", 0.3)
        self.declare_parameter("log_dir", os.path.join(SCRIPT_DIR, "logs"))

        policy_path = self.get_parameter("policy_path").value
        device = self.get_parameter("device").value
        self._ball_vx_thresh = self.get_parameter("ball_incoming_vx_threshold").value
        self._ball_x_offset = self.get_parameter("ball_incoming_x_offset").value
        log_dir = self.get_parameter("log_dir").value

        # ---- CSV Logger ----
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"a1_deploy_{timestamp}.csv")
        self._csv_file = open(log_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "time_s", "state",
            "jp_1", "jp_2", "jp_3", "jp_4", "jp_5", "jp_6", "jp_7",
            "jv_1", "jv_2", "jv_3", "jv_4", "jv_5", "jv_6", "jv_7",
            "je_1", "je_2", "je_3", "je_4", "je_5", "je_6", "je_7",
            "tgt_1", "tgt_2", "tgt_3", "tgt_4", "tgt_5", "tgt_6", "tgt_7",
            "ball_x", "ball_y", "ball_z",
            "ball_vx", "ball_vy", "ball_vz",
            "racket_x", "racket_y", "racket_z",
            "phase", "phase_speed", "motion_id",
            "ball_time_to_arrive", "pred_hit_y", "pred_hit_z", "pred_hit_t",
            "bounce_has", "bounce_rising", "bounce_urgency",
        ])
        # 观测值+动作 单独 CSV (方便 sim2real 对比)
        obs_log_path = os.path.join(log_dir, f"a1_obs_{timestamp}.csv")
        self._obs_csv_file = open(obs_log_path, "w", newline="")
        self._obs_csv_writer = csv.writer(self._obs_csv_file)
        self._obs_csv_writer.writerow([
            "time_s", "state",
            *[f"obs_{i:02d}" for i in range(57)],
            *[f"act_{i}" for i in range(8)],
        ])
        self._log_start_time = time.time()
        self.get_logger().info(f"CSV log: {log_path}")
        self.get_logger().info(f"Obs log: {obs_log_path}")

        # ---- PPO Controller ----
        self.get_logger().info(f"Loading A1 PPO policy from: {policy_path}")
        self.controller = A1TableTennisController(
            policy_path=policy_path, device=device
        )

        # 覆盖 phase machine 内部的 STEP_DT，匹配 50Hz 控制频率
        import a1_table_tennis_deploy
        a1_table_tennis_deploy.STEP_DT = STEP_DT

        # ---- State ----
        self._lock = Lock()
        self._joint_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._joint_vel = np.zeros(7, dtype=np.float32)
        self._joint_effort = np.zeros(7, dtype=np.float32)
        self._ball_pos = np.zeros(3, dtype=np.float32)
        self._ball_vel = np.zeros(3, dtype=np.float32)
        self._has_joint_data = False
        self._has_ball_data = False
        self._ball_last_time = 0.0

        # 差分计算关节速度
        self._prev_joint_pos = self._joint_pos.copy()
        self._prev_joint_time = 0.0

        # 球加速度估计 (用于闭环 phase 校正)
        self._prev_ball_vx = 0.0
        self._prev_ball_vel_time = 0.0
        self._ball_ax_est = 0.0  # 指数平滑后的 ax 估计值

        # FK for racket position (computed from joint angles, no external topic needed)
        self._lift_joint_pos = -0.28
        if URDFForwardKinematics is not None:
            self._fk = URDFForwardKinematics(ROBOT_POS)
        else:
            self._fk = None
            self.get_logger().warn("URDFForwardKinematics not available, using approximate racket pos")

        # Rally state
        self._state = RallyState.IDLE
        self._recover_t = 0.0
        self._recover_start_pos = None
        self._blend_t = 0.0
        self._blend_start_pos = None
        self._default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._last_target = self._default_pos.copy()
        self._hit_count = 0
        self._swing_count = 0

        # ---- ROS2 Subscriptions ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(
            JointState, "/right_joint_states", self._joint_states_cb, qos
        )
        self.create_subscription(
            PoseStamped, "/kalman/pingpong_pos", self._ball_pos_cb, qos
        )
        self.create_subscription(
            TwistStamped, "/kalman/pingpong_vel", self._ball_vel_cb, qos
        )

        # ---- ROS2 Publisher ----
        self._action_pub = self.create_publisher(
            JointState, "a1/sent_actions", qos
        )

        # ---- 50Hz Control Timer ----
        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self._last_log_time = 0.0

        self.get_logger().info(
            f"A1 PPO PingPong node started @ {CONTROL_HZ} Hz. "
            f"State: IDLE. Waiting for ball..."
        )

    # ------------------------------------------------------------------
    # ROS2 Callbacks
    # ------------------------------------------------------------------

    def _joint_states_cb(self, msg: JointState) -> None:
        now = time.time()
        with self._lock:
            # DEBUG: 打印一次实际收到的关节名
            if not hasattr(self, "_joint_names_logged"):
                self.get_logger().info(f"Joint names in /right_joint_states: {list(msg.name)}")
                self._joint_names_logged = True
            for name, pos in zip(msg.name, msg.position):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    idx = RIGHT_ARM_SOURCE_NAMES[name]
                    self._joint_pos[idx] = float(pos)
            # 差分计算关节速度（替代 topic velocity）
            dt = now - self._prev_joint_time if self._prev_joint_time > 0 else 0.0
            if dt > 1e-4:
                self._joint_vel = (self._joint_pos - self._prev_joint_pos) / dt
            self._prev_joint_pos = self._joint_pos.copy()
            self._prev_joint_time = now
            if msg.effort:
                for name, eff in zip(msg.name, msg.effort):
                    if name in RIGHT_ARM_SOURCE_NAMES:
                        idx = RIGHT_ARM_SOURCE_NAMES[name]
                        self._joint_effort[idx] = float(eff)
            self._has_joint_data = True

    def _ball_pos_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._ball_pos[0] = msg.pose.position.x
            self._ball_pos[1] = msg.pose.position.y
            self._ball_pos[2] = msg.pose.position.z
            self._has_ball_data = True
            self._ball_last_time = time.time()

    def _ball_vel_cb(self, msg: TwistStamped) -> None:
        now = time.time()
        with self._lock:
            new_vx = msg.twist.linear.x
            # 估计球 x 方向加速度
            dt = now - self._prev_ball_vel_time if self._prev_ball_vel_time > 0 else 0.0
            if dt > 1e-4 and dt < 0.5:
                ax_raw = (new_vx - self._prev_ball_vx) / dt
                # 指数平滑，alpha=0.3 平衡响应速度和噪声
                self._ball_ax_est = 0.7 * self._ball_ax_est + 0.3 * ax_raw
            self._prev_ball_vx = new_vx
            self._prev_ball_vel_time = now

            self._ball_vel[0] = new_vx
            self._ball_vel[1] = msg.twist.linear.y
            self._ball_vel[2] = msg.twist.linear.z

    def _racket_pos_from_fk(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute racket position via FK from current joint angles."""
        if self._fk is not None:
            return self._fk.compute(self._lift_joint_pos, joint_pos)
        # Fallback: approximate position based on robot base
        return ROBOT_POS + np.array([0.35, 0.0, 1.1], dtype=np.float32)

    # ------------------------------------------------------------------
    # Control Loop (50Hz)
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        if not self._has_joint_data:
            return

        # 首次收到关节数据时，将 _last_target 初始化为当前实际位置
        if not hasattr(self, "_target_initialized"):
            self._last_target = self._joint_pos.copy()
            self._target_initialized = True

        with self._lock:
            joint_pos = self._joint_pos.copy()
            joint_vel = self._joint_vel.copy()
            joint_effort = self._joint_effort.copy()
            ball_pos = self._ball_pos.copy()
            ball_vel = self._ball_vel.copy()
            has_ball = self._has_ball_data
            ball_age = time.time() - self._ball_last_time

        # 球角速度无法观测，设为零
        ball_ang_vel = np.zeros(3, dtype=np.float32)

        # 球拍位置通过 FK 从关节角度计算
        racket_pos = self._racket_pos_from_fk(joint_pos)
        # 球拍位置通过 FK 从关节角度计算
        racket_pos = self._racket_pos_from_fk(joint_pos)

        # 球数据超过 1 秒视为丢失
        if ball_age > 1.0:
            has_ball = False

        # A1 坐标系与训练一致 (+X 指向对手)，无需坐标变换

        # ---- Rally State Machine ----
        target = self._run_state_machine(
            joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos, has_ball
        )

        # ---- 发布关节指令 ----
        self._publish_action(target)

        # ---- CSV 记录 ----
        t = time.time() - self._log_start_time
        pm = self.controller.phase_machine
        # 计算球到达相关观测值
        ball_tta = compute_ball_time_to_arrive(ball_pos, ball_vel)
        pred_y, pred_z, pred_t = predict_ball_hit_point(ball_pos, ball_vel)
        bounce_state = compute_ball_bounce_state(ball_pos, ball_vel)
        self._csv_writer.writerow([
            f"{t:.4f}", self._state,
            *[f"{v:.5f}" for v in joint_pos],
            *[f"{v:.4f}" for v in joint_vel],
            *[f"{v:.3f}" for v in joint_effort],
            *[f"{v:.5f}" for v in target],
            *[f"{v:.4f}" for v in ball_pos],
            *[f"{v:.4f}" for v in ball_vel],
            *[f"{v:.4f}" for v in racket_pos],
            f"{pm.phase:.4f}", f"{pm.phase_speed:.4f}", pm.motion_id,
            f"{ball_tta:.4f}", f"{pred_y:.4f}", f"{pred_z:.4f}", f"{pred_t:.4f}",
            f"{bounce_state[0]:.1f}", f"{bounce_state[1]:.1f}", f"{bounce_state[2]:.4f}",
        ])

        # 观测值 CSV: 记录策略实际输入的 57D obs 和 8D action
        info = getattr(self, '_last_info', None)
        if info is not None and "obs" in info:
            obs = info["obs"]
            act = info["action_raw"]
        else:
            obs = np.zeros(57, dtype=np.float32)
            act = np.zeros(8, dtype=np.float32)
        self._obs_csv_writer.writerow([
            f"{t:.4f}", self._state,
            *[f"{v:.5f}" for v in obs],
            *[f"{v:.5f}" for v in act],
        ])

        # ---- 日志 ----
        now = time.time()
        if now - self._last_log_time >= 2.0:
            self._last_log_time = now
            state_names = ["IDLE", "BLENDING", "TRACKING", "SWINGING", "RECOVERING"]
            phase = self.controller.phase_machine.phase
            self.get_logger().info(
                f"State={state_names[self._state]} | phase={phase:.3f} | "
                f"swings={self._swing_count} | ball_pos={ball_pos} | "
                f"has_ball={has_ball}"
            )

    def _run_state_machine(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        racket_pos: np.ndarray,
        has_ball: bool,
    ) -> np.ndarray:
        """Rally 状态机，返回关节目标位置 (7,)."""

        if self._state == RallyState.IDLE:
            # Hold current position，缓慢指数平滑回归 default_pos（避免突变）
            self._last_target = (
                (1 - self.IDLE_RETURN_ALPHA) * self._last_target
                + self.IDLE_RETURN_ALPHA * self._default_pos
            )
            self._last_info = None
            if has_ball and self._ball_incoming(ball_pos, ball_vel):
                self._start_blending(ball_pos, ball_vel, joint_pos)

        elif self._state == RallyState.BLENDING:
            # 平滑过渡: 从当前位置插值到策略第一帧输出
            target, info = self.controller.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_info = info
            self._blend_t += STEP_DT
            alpha = min(1.0, self._blend_t / self.BLEND_DURATION)
            alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
            self._last_target = (
                (1 - alpha_smooth) * self._blend_start_pos
                + alpha_smooth * target
            )
            if alpha >= 1.0:
                self._state = RallyState.TRACKING

        elif self._state == RallyState.TRACKING:
            target, info = self.controller.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_info = info
            # 闭环 phase 校正: 根据球实时位置追赶理想 phase
            self._correct_phase(ball_pos, ball_vel)
            self._last_target = target

            if info["phase"] > HIT_PHASE:
                self._state = RallyState.SWINGING

        elif self._state == RallyState.SWINGING:
            target, info = self.controller.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_info = info
            self._last_target = target

            if self.controller.is_swing_done:
                self._swing_count += 1
                self._start_recovery(joint_pos)

        elif self._state == RallyState.RECOVERING:
            self._last_info = None
            self._recover_t += STEP_DT
            alpha = min(1.0, self._recover_t / self.RECOVER_DURATION)
            alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
            self._last_target = (
                (1 - alpha_smooth) * self._recover_start_pos
                + alpha_smooth * self._default_pos
            )
            if alpha >= 1.0:
                self._state = RallyState.IDLE

        # 速率限制: 防止任何阶段出现目标跳变，限制每步最大变化量
        # 50Hz 下 MAX_DELTA=0.08 rad/step ≈ 4 rad/s 最大关节速度
        MAX_DELTA = 0.08
        if hasattr(self, '_prev_output'):
            delta = self._last_target - self._prev_output
            self._last_target = self._prev_output + np.clip(delta, -MAX_DELTA, MAX_DELTA)
        self._prev_output = self._last_target.copy()

        return self._last_target

    def _ball_incoming(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
        """判断球是否正朝机器人飞来 (A1: vx < 0 表示球飞向 -X 机器人)。"""
        return (
            ball_vel[0] < self._ball_vx_thresh
            and ball_pos[0] > ROBOT_X + self._ball_x_offset
            and ball_pos[2] > 0.0
        )

    def _correct_phase(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """闭环 phase 校正: 根据球实时位置让 phase 追上理想值。
        使用匀加速模型估计 t_remain，补偿球的真实加速度。"""
        vx = ball_vel[0]
        if vx >= -0.1:
            return
        pm = self.controller.phase_machine
        duration = pm.motion.duration(pm.motion_id)

        # 匀加速模型: x = x0 + vx*t + 0.5*ax*t²
        # 解 t 使得 ball_x + vx*t + 0.5*ax*t² = ROBOT_X
        dx = ROBOT_X - ball_pos[0]  # 负值 (球在 ROBOT_X 前方)
        ax = self._ball_ax_est

        t_remain = None
        if abs(ax) > 0.1:
            # 二次方程: 0.5*ax*t² + vx*t - dx = 0
            disc = vx * vx + 2.0 * ax * dx
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                # 取正根: t = (-vx - sqrt(disc)) / ax
                t_remain = (-vx - sqrt_disc) / ax
                if t_remain <= 0:
                    t_remain = None

        # 退化到匀速模型
        if t_remain is None:
            t_remain = dx / vx

        t_remain = max(0.05, t_remain)

        # 理想 phase: 使得 t_remain 后恰好到 HIT_PHASE
        ideal_phase = (HIT_PHASE - t_remain / duration) % 1.0
        # phase 误差 (wrapped to [-0.5, 0.5])
        error = ideal_phase - pm.phase
        if error > 0.5:
            error -= 1.0
        elif error < -0.5:
            error += 1.0
        # 只向前校正 (加速)，不向后拉 (避免动作倒退)
        if error > 0:
            correction = min(error * self.PHASE_CORRECTION_GAIN, self.PHASE_CORRECTION_MAX)
            pm.phase += correction

    def _start_blending(self, ball_pos: np.ndarray, ball_vel: np.ndarray, current_pos: np.ndarray):
        """从 IDLE 进入 BLENDING，平滑过渡到参考动作。"""
        self.controller.reset(ball_pos, ball_vel)
        self._blend_start_pos = current_pos.copy()
        self._blend_t = 0.0
        self._state = RallyState.BLENDING
        self.get_logger().info(
            f"Ball incoming! vx={ball_vel[0]:.2f} pos={ball_pos} → BLENDING"
        )

    def _start_recovery(self, current_pos: np.ndarray):
        """挥拍完成，开始回归。"""
        self._state = RallyState.RECOVERING
        self._recover_t = 0.0
        self._recover_start_pos = current_pos.copy()

    def _publish_action(self, targets: np.ndarray):
        """发布 7 个关节目标位置到 a1/sent_actions。"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = OUTPUT_JOINT_NAMES
        msg.position = [float(t) for t in targets]
        msg.velocity = []
        msg.effort = []
        self._action_pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = PPOPingPongNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._csv_file.close()
        node._obs_csv_file.close()
        node.get_logger().info("CSV log saved and closed.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
