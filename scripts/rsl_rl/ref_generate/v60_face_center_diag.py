"""V60: 找出 paddle face center 真实世界位置, 然后看 V58 击球时刻 face 离球差多少.

问题背景:
  V58 跑下来球打在拍柄上 (用户视觉确认). V59 试图通过 wrist 旋转修正, 但基于
  "paddle body frame 轴 = STL 视觉轴 (face_width=x, face_normal=y, handle=z)"
  的假设是错的, 导致 V59 完全脱靶.

本脚本两步:
  PHASE A: 在 ready pose 下, 把每个 wrist 关节 (yb_5/6/7) 单独从 0 转到 +0.5 rad,
           记录 paddle quaternion 变化, 反推该关节的旋转轴在 body frame 中的方向.
           URDF 里 wrist_roll 一定绕 forearm 长轴 (= handle 方向),
           wrist_pitch 绕 face_width 方向, wrist_yaw 绕 face_normal 方向.
           这样就知道 body x_local / y_local / z_local 各对应 paddle 哪个语义方向.

  PHASE B: 跑 V58 完整轨迹. 在 hit window (t=0.40~0.55) 每 10ms 打印:
      - paddle 原点世界坐标
      - paddle body 三个轴在世界中的方向 (R 矩阵列)
      - 球世界坐标
      - 球 - paddle 原点 在 paddle body frame 三个轴上的投影 (mm)
      - 用 PHASE A 的轴标签 (handle/face_normal/face_width) 翻译投影
      - face_center 估计世界位置 (= origin + R @ face_offset_body)

  face_offset_body 计算: STL 显示 paddle 原点在手柄内部, face 中心在原点向 face 方向
      位移 ~120mm 处. 具体方向由 PHASE A 探测.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="X1-TableTennis")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if "--headless" not in sys.argv:
    args_cli.headless = True
    sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from scipy.interpolate import CubicSpline

import isaaclab_tasks  # noqa
import unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


V58 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def quat_to_R(q):
    """quat (w,x,y,z) -> 3x3 rotation matrix (col i = body axis i in world)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)


def quat_axis_angle(q):
    """quat (w,x,y,z) -> (axis_xyz, angle_rad). Axis is unit vector in world frame."""
    w = float(q[0])
    xyz = q[1:].cpu().numpy().astype(np.float64)
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / sin_half
    return axis, angle


def quat_mul(q1, q2):
    """Hamilton product of (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], device=q1.device, dtype=q1.dtype)


def quat_conj(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device, dtype=q.dtype)


def settle(robot, scene, env, full_target, ids, n=200):
    for _ in range(n):
        robot.set_joint_position_target(full_target, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(env.unwrapped.sim.get_physics_dt())


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]

    # ============================================================
    # PHASE A: 探测 wrist 关节的旋转轴 (= body local 各轴对应)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE A: probing wrist axes (yb_5=roll, yb_6=pitch, yb_7=yaw)")
    print("=" * 70)

    full = robot.data.default_joint_pos[0:1].clone()
    # 用 V58 t=0 的姿势 (ready pose)
    ready = V58[0][1]
    for k, jid in enumerate(yb_joint_ids):
        full[0, jid] = ready[k]
    v0 = torch.zeros_like(full)
    ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(full, v0, env_ids=ids)
    settle(robot, scene, env, full, ids, n=300)

    q_ref = robot.data.body_quat_w[0, paddle_idx].clone()
    R_ref = quat_to_R(q_ref).cpu().numpy()
    print(f"\nready pose paddle quat (wxyz): "
          f"({q_ref[0]:+.4f}, {q_ref[1]:+.4f}, {q_ref[2]:+.4f}, {q_ref[3]:+.4f})")
    print("R columns = body axes in world frame:")
    print(f"  body +x_local in world: ({R_ref[0,0]:+.3f}, {R_ref[1,0]:+.3f}, {R_ref[2,0]:+.3f})")
    print(f"  body +y_local in world: ({R_ref[0,1]:+.3f}, {R_ref[1,1]:+.3f}, {R_ref[2,1]:+.3f})")
    print(f"  body +z_local in world: ({R_ref[0,2]:+.3f}, {R_ref[1,2]:+.3f}, {R_ref[2,2]:+.3f})")

    axis_labels = {}  # joint_idx -> (body_axis_label, sign)
    for j_idx, name in [(4, "yb_5 wrist_roll"), (5, "yb_6 wrist_pitch"), (6, "yb_7 wrist_yaw")]:
        target = full.clone()
        target[0, yb_joint_ids[j_idx]] = ready[j_idx] + 0.5
        settle(robot, scene, env, target, ids, n=400)

        q_after = robot.data.body_quat_w[0, paddle_idx].clone()
        # 旋转增量: q_delta = q_after * q_ref^-1 (世界系下)
        q_delta = quat_mul(q_after, quat_conj(q_ref))
        axis_world, angle = quat_axis_angle(q_delta)
        # 把世界轴投影回 body frame: axis_body = R_ref^T @ axis_world
        axis_body = R_ref.T @ axis_world

        # 找最大分量 = 该关节绕的 body 轴
        idx_max = int(np.argmax(np.abs(axis_body)))
        sign = "+" if axis_body[idx_max] > 0 else "-"
        body_axis_str = f"{sign}{['x', 'y', 'z'][idx_max]}_local"
        axis_labels[j_idx] = (body_axis_str, axis_body)
        print(f"\n  {name}: rotated {np.degrees(angle):+.1f}°")
        print(f"    rotation axis in world: ({axis_world[0]:+.3f}, {axis_world[1]:+.3f}, {axis_world[2]:+.3f})")
        print(f"    rotation axis in body:  ({axis_body[0]:+.3f}, {axis_body[1]:+.3f}, {axis_body[2]:+.3f})")
        print(f"    -> joint {name} rotates around body {body_axis_str}")

        # 复位
        settle(robot, scene, env, full, ids, n=200)

    # 物理意义:
    #   wrist_roll  绕 forearm 长轴 = handle 方向 (paddle 拿住后, 沿手柄)
    #   wrist_pitch 绕 face_width 方向 (拍面平躺时左右轴)
    #   wrist_yaw   绕 face_normal 方向 (拍面法线)
    print("\n" + "-" * 70)
    print("Body axis interpretation:")
    print(f"  yb_5 (wrist_roll)  axis = {axis_labels[4][0]}  -> this is HANDLE direction")
    print(f"  yb_6 (wrist_pitch) axis = {axis_labels[5][0]}  -> this is FACE_WIDTH direction")
    print(f"  yb_7 (wrist_yaw)   axis = {axis_labels[6][0]}  -> this is FACE_NORMAL direction")

    # face_offset_body: paddle link 原点在手柄内部, face center 沿 -handle 方向偏 ~120mm
    handle_axis_label, handle_axis_body = axis_labels[4]
    handle_dir = handle_axis_body / np.linalg.norm(handle_axis_body)
    # 假设 paddle link 原点在 handle 中部, face center 离原点 ~0.10m 沿 -handle 方向
    # (handle 长度 ~92mm 从 +78 到 +170, 中部 +124; face center 在 +5.5; 差 ~120mm)
    face_offset_body = -handle_dir * 0.12
    print(f"\nface_center offset in body frame (assumed -0.12m along handle axis):")
    print(f"  ({face_offset_body[0]:+.4f}, {face_offset_body[1]:+.4f}, {face_offset_body[2]:+.4f})")

    # ============================================================
    # PHASE B: 跑 V58 完整轨迹, 在击球窗口打印 paddle face center 与球位置
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE B: V58 trajectory diagnostics")
    print("=" * 70)

    times = np.array([k[0] for k in V58])
    angs = np.array([k[1] for k in V58], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    # 重置到 ready 姿势
    q0 = spline(0.0)
    for k, jid in enumerate(yb_joint_ids):
        full[0, jid] = float(q0[k])
    robot.write_joint_state_to_sim(full, v0, env_ids=ids)
    settle(robot, scene, env, full, ids, n=200)

    # 发射球
    ball_state = ball.data.default_root_state.clone()
    ball_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
    ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
    ball_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
    ball.write_root_state_to_sim(ball_state, env_ids=ids)
    scene.write_data_to_sim()

    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    log = []
    for step in range(int(0.65 / sim_dt)):
        t = step * sim_dt
        target = spline(min(t, 1.0))
        full_target = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full_target[0, jid] = float(target[k])
        robot.set_joint_position_target(full_target, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)

        p = robot.data.body_pos_w[0, paddle_idx].clone()
        q = robot.data.body_quat_w[0, paddle_idx].clone()
        b = ball.data.root_pos_w[0].clone()
        log.append((t, p.cpu().numpy(), q.cpu().numpy(), b.cpu().numpy()))

    print(f"\n  axes labels: yb5=handle({axis_labels[4][0]}), "
          f"yb6=face_width({axis_labels[5][0]}), yb7=face_normal({axis_labels[6][0]})")
    print(f"  face_center = paddle_origin + R @ face_offset_body  (face_offset = -0.12m along handle)")
    print(f"\n  {'t':>5}  {'paddle_orig (xyz)':<24}  {'face_center (xyz)':<24}  "
          f"{'ball (xyz)':<24}  {'face->ball (mm)':<22}  {'gap'}")

    best_t, best_gap = None, 1e9
    for (t, p, q, b) in log:
        if not (0.35 <= t <= 0.60):
            continue
        q_t = torch.from_numpy(q).to(device)
        R = quat_to_R(q_t).cpu().numpy()
        face_world = p + R @ face_offset_body
        delta = b - face_world
        gap = float(np.linalg.norm(delta))
        if gap < best_gap:
            best_gap = gap
            best_t = t
            best_R = R
            best_p = p
            best_face = face_world
            best_b = b
        print(f"  {t:>5.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
              f"({face_world[0]:+.3f},{face_world[1]:+.3f},{face_world[2]:+.3f})  "
              f"({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})  "
              f"({delta[0]*1000:+5.0f},{delta[1]*1000:+5.0f},{delta[2]*1000:+5.0f})  "
              f"{gap*100:.2f}cm")

    print(f"\n=== Closest face-center approach ===")
    print(f"  t = {best_t:.3f}, gap = {best_gap*100:.2f}cm")
    print(f"  paddle_origin world = ({best_p[0]:+.3f}, {best_p[1]:+.3f}, {best_p[2]:+.3f})")
    print(f"  face_center  world = ({best_face[0]:+.3f}, {best_face[1]:+.3f}, {best_face[2]:+.3f})")
    print(f"  ball         world = ({best_b[0]:+.3f}, {best_b[1]:+.3f}, {best_b[2]:+.3f})")
    delta_w = best_b - best_face
    print(f"  Δ (ball - face) world = ({delta_w[0]*1000:+.0f}, {delta_w[1]*1000:+.0f}, {delta_w[2]*1000:+.0f}) mm")

    # 把 face->ball 的世界向量分解到 body frame, 看球落在 face 的哪个方向
    delta_body = best_R.T @ delta_w
    print(f"  Δ in body frame  = ({delta_body[0]*1000:+.0f}, {delta_body[1]*1000:+.0f}, {delta_body[2]*1000:+.0f}) mm")
    h_axis_idx = int(np.argmax(np.abs(handle_dir)))
    h_sign = np.sign(handle_dir[h_axis_idx])
    h_proj = float(delta_body[h_axis_idx] * h_sign)  # 球沿 +handle 方向的投影 (正 = 朝向 handle)
    print(f"  ball position along +handle axis: {h_proj*1000:+.0f}mm "
          f"({'toward handle' if h_proj>0 else 'toward face tip'})")

    # 给出修正方向: 想让 ball 落在 face center, 需要 paddle 在世界中沿何方向平移多少?
    # 平移量 = ball - face_center (世界向量), face_center 应朝 ball 移动这个向量.
    print(f"\n  -> 把 paddle 在世界中平移 ({delta_w[0]*1000:+.0f}, {delta_w[1]*1000:+.0f}, "
          f"{delta_w[2]*1000:+.0f}) mm 才能让 face center 接到球.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
