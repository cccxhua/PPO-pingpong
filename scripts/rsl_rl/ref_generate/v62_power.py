"""V62: 在 V61 击球位置基础上加力. 目标:
  - PIN 时刻 (t=0.475) paddle 位置 + face_normal 不变
  - paddle linear velocity 沿 -X 方向 (反弹球到 -X) 显著增大
  - vz 略 + (loft, 让球飞过网)

策略: 深化 windup (PIN 前 0.075s) + 加大 snap (PIN 后 0.075s).
- yb_6 (wrist_pitch): 主导 wrist snap, 给 -X+Z 速度
- yb_1 (shoulder_pitch): 主导 shoulder swing, 给 +X 但抬高
- yb_4 (elbow): 主导 elbow extension, 给 +X
- 注意: yb_1, yb_4 的 +X 速度 与 ball 反方向 (-X) 相反, 不能加多.
- 主要靠 yb_6 (wrist snap), 它给 -X+Z 速度.

V61 keyframes (hit window):
  (0.300, [+1.127, +0.198, -1.904, +1.227, -0.115, -1.045,  +1.000])  mid
  (0.400, [+1.187, +0.103, -1.979, +0.857, -0.115, -1.245,  +1.000])  windup
  (0.475, [+1.287, +0.103, -1.979, +0.807, -0.145, -1.020,  +1.000])  PIN
  (0.550, [+1.437, +0.103, -1.979, +0.757, -0.115, -0.495,  +1.000])  snap
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


V61 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.227, -0.115, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.857, -0.115, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.807, -0.145, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.757, -0.115, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

HIT_T = 0.475


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)


def edit_keyframe(keys, t, j, new_val):
    """替换 keyframe (t, j) 处的值."""
    out = []
    for kt, vals in keys:
        v = list(vals)
        if abs(kt - t) < 1e-6:
            v[j] = new_val
        out.append((kt, v))
    return out


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

    def run(keys, capture_window=0.06, with_ball=True):
        """跑轨迹, 在 hit 时刻 ±capture_window 内每个 sim step 抓 paddle/ball 状态."""
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(0.0)
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v_init = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v_init, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

        if with_ball:
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
            if abs(t - HIT_T) < capture_window:
                p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
                v = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
                w = robot.data.body_ang_vel_w[0, paddle_idx].cpu().numpy().copy()
                q = robot.data.body_quat_w[0, paddle_idx].clone()
                R = quat_to_R(q).cpu().numpy()
                if with_ball:
                    bp = ball.data.root_pos_w[0].cpu().numpy().copy()
                    bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
                else:
                    bp = np.zeros(3); bv = np.zeros(3)
                log.append((t, p, v, R, bp, bv, w))
        # find sample with min paddle-ball distance (= contact moment)
        if with_ball:
            hit_sample = min(log, key=lambda r: np.linalg.norm(r[1] - r[4]))
        else:
            hit_sample = min(log, key=lambda r: abs(r[0]-HIT_T))
        return hit_sample, log

    # =========================================================
    # 1) baseline V61 with ball: log full hit window 看 paddle 是否真够到球
    # =========================================================
    (t0, p0, v0, R0, bp0, bv0, w0), log0 = run(V61, with_ball=True)
    # PHASE A 实测 (v60_face_center_diag): handle_body = (-0.456, -0.707, +0.541), face_normal_body = (0, 0, -1)
    HANDLE_BODY = np.array([-0.456, -0.707, +0.541])
    HANDLE_BODY = HANDLE_BODY / np.linalg.norm(HANDLE_BODY)
    FACE_NORMAL_BODY = np.array([0.0, 0.0, -1.0])  # -z_local
    FACE_OFFSET_BODY = -0.12 * HANDLE_BODY  # face center 沿 -handle 方向 12cm

    n0 = R0 @ FACE_NORMAL_BODY  # face_normal in world
    handle_world = R0 @ HANDLE_BODY
    face_offset_world = R0 @ FACE_OFFSET_BODY
    face_p = p0 + face_offset_world
    face_v = v0 + np.cross(w0, face_offset_world)
    print(f"\n=== V61 baseline (closest paddle-ball moment) ===")
    print(f"  t = {t0:.3f}")
    print(f"  paddle origin pos = ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")
    print(f"  ball   pos        = ({bp0[0]:+.3f}, {bp0[1]:+.3f}, {bp0[2]:+.3f})")
    print(f"  Δ (ball - origin) = ({(bp0[0]-p0[0])*1000:+.0f},{(bp0[1]-p0[1])*1000:+.0f},{(bp0[2]-p0[2])*1000:+.0f}) mm,"
          f" gap={np.linalg.norm(bp0-p0)*100:.1f}cm")
    print(f"  face center est   = ({face_p[0]:+.3f}, {face_p[1]:+.3f}, {face_p[2]:+.3f})")
    print(f"  Δ (ball - face)   = ({(bp0[0]-face_p[0])*1000:+.0f},{(bp0[1]-face_p[1])*1000:+.0f},{(bp0[2]-face_p[2])*1000:+.0f}) mm,"
          f" gap={np.linalg.norm(bp0-face_p)*100:.1f}cm")
    print(f"  paddle origin lin v = ({v0[0]:+.2f}, {v0[1]:+.2f}, {v0[2]:+.2f}) |v|={np.linalg.norm(v0):.2f}")
    print(f"  paddle ang ω        = ({w0[0]:+.2f}, {w0[1]:+.2f}, {w0[2]:+.2f}) |ω|={np.linalg.norm(w0):.2f} rad/s")
    print(f"  face center lin v   = ({face_v[0]:+.2f}, {face_v[1]:+.2f}, {face_v[2]:+.2f}) |v|={np.linalg.norm(face_v):.2f}")
    print(f"  ball       v       = ({bv0[0]:+.2f}, {bv0[1]:+.2f}, {bv0[2]:+.2f}) m/s")
    print(f"  face_normal world = ({n0[0]:+.3f}, {n0[1]:+.3f}, {n0[2]:+.3f})")
    print(f"  handle    world = ({handle_world[0]:+.3f}, {handle_world[1]:+.3f}, {handle_world[2]:+.3f})")
    print(f"  face_v · n̂ = {float(np.dot(face_v, n0)):+.2f} m/s  (正=面推球远离 → 球反弹)")
    print(f"  origin_v · n̂ = {float(np.dot(v0, n0)):+.2f} m/s")

    print(f"\n  Hit-window timeline (paddle X vs ball X):")
    print(f"  {'t':>5}  {'paddle (xyz)':<24}  {'ball (xyz)':<24}  {'Δx mm':<8} {'Δy mm':<8} {'Δz mm':<8} {'gap cm':<6}")
    for entry in log0:
        t, p, v, R, bp, bv, w = entry
        gap = np.linalg.norm(bp - p)
        dx = (bp[0] - p[0]) * 1000
        dy = (bp[1] - p[1]) * 1000
        dz = (bp[2] - p[2]) * 1000
        marker = " <-min" if abs(t-t0) < 1e-6 else ""
        print(f"  {t:>5.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  ({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f})  {dx:+5.0f}    {dy:+5.0f}    {dz:+5.0f}    {gap*100:.1f}{marker}")

    # =========================================================
    # 2) 测试若干 variants 找最大反向速度
    #    - deeper windup at t=0.40 (yb_6 更负)
    #    - bigger snap at t=0.55 (yb_6 更正)
    #    - 同时保持 t=0.475 PIN 关节值不变 (等价 paddle pos+ori 不变)
    # =========================================================
    print(f"\n{'variant':<60}  {'paddle vel (m/s)':<28}  {'|v|':<6}  {'v·n':<7}")

    def make(yb6_windup=None, yb6_snap=None, yb1_windup=None, yb1_snap=None,
             yb4_windup=None, yb4_snap=None, yb5_snap=None):
        keys = V61
        if yb6_windup is not None: keys = edit_keyframe(keys, 0.400, 5, yb6_windup)
        if yb6_snap   is not None: keys = edit_keyframe(keys, 0.550, 5, yb6_snap)
        if yb1_windup is not None: keys = edit_keyframe(keys, 0.400, 0, yb1_windup)
        if yb1_snap   is not None: keys = edit_keyframe(keys, 0.550, 0, yb1_snap)
        if yb4_windup is not None: keys = edit_keyframe(keys, 0.400, 3, yb4_windup)
        if yb4_snap   is not None: keys = edit_keyframe(keys, 0.550, 3, yb4_snap)
        if yb5_snap   is not None: keys = edit_keyframe(keys, 0.550, 4, yb5_snap)
        return keys

    variants = [
        ("V61 baseline", V61),
        # 1) wrist snap 加深 windup + 加大 snap (yb_6 是主要 -X+Z 速度源)
        ("yb6 windup -1.45 + snap -0.20",  make(yb6_windup=-1.45, yb6_snap=-0.20)),
        ("yb6 windup -1.55 + snap +0.00",  make(yb6_windup=-1.55, yb6_snap=+0.00)),
        ("yb6 windup -1.65 + snap +0.20",  make(yb6_windup=-1.65, yb6_snap=+0.20)),
        # 2) elbow snap (yb_4 减小 = 伸肘)
        ("yb4 windup +0.95 + snap +0.55",  make(yb4_windup=+0.95, yb4_snap=+0.55)),
        # 3) shoulder swing (yb_1 增大 = 前送, 给 +X 但与球反向)
        ("yb1 windup +1.10 + snap +1.55",  make(yb1_windup=+1.10, yb1_snap=+1.55)),
        # 4) 全套: wrist + elbow + shoulder
        ("FULL: yb6+yb4 (no shoulder push)",
            make(yb6_windup=-1.55, yb6_snap=+0.00,
                 yb4_windup=+0.95, yb4_snap=+0.55)),
        ("FULL: yb6+yb1+yb4",
            make(yb6_windup=-1.55, yb6_snap=+0.00,
                 yb1_windup=+1.10, yb1_snap=+1.55,
                 yb4_windup=+0.95, yb4_snap=+0.55)),
        # 5) 极端 wrist snap
        ("EXTREME yb6 only -1.65 -> +0.40",
            make(yb6_windup=-1.65, yb6_snap=+0.40)),
    ]

    for label, keys in variants:
        (t, p, v, R, bp, bv, w), log = run(keys, with_ball=True)
        n = R[:, 2]
        speed = float(np.linalg.norm(v))
        v_n = float(np.dot(v, n))
        gap = float(np.linalg.norm(bp - p))
        post_vx = min(r[5][0] for r in log)
        post_vz = max(r[5][2] for r in log)
        ang = float(np.linalg.norm(w))
        print(f"  {label:<60}  ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) ω={ang:.1f}  {speed:.2f}    {v_n:+.2f}  "
              f"gap={gap*100:.1f}  bx={post_vx:+.2f}  bz={post_vz:+.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
