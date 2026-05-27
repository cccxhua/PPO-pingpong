"""Probe RIGHT forehand motion: 检查 paddle-ball 几何 + 击球效果.

发球: 球从 (x=-0.35, y=-0.3, z=1.3) 以 (vx=3.5, vy=-0.4, vz=0.5) 飞向 robot -y 侧.
predicted_y @ robot_x=1.5: -0.3 + (-0.4)*0.529 = -0.51 → 选 motion_id=2 (right).

扫描 hit_phase ∈ [0.425, 0.525], 报告:
  - paddle-ball 最小距离 + 时刻
  - paddle 击球瞬间线速度 (vx, vz)
  - ball post-hit 速度 (vx, vz)
  - z_at_net, x_bounce (解析弹道)
  - 是否过网 + 是否落对方台

用法:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_motion.py --task X1-TableTennis
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

# RIGHT 关键帧 v15: PIN plateau + yb_5=-1.10 (best face for -X reflection)
RIGHT = [
    (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # hold
    (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),  # windup
    (0.475, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),  # PIN start (yb5=-1.10)
    (0.540, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),  # PIN hold (plateau)
    (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),  # follow
    (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # return hold
]

# 发球参数 (ball to robot's -y side, predicted_y ≈ -0.08)
BALL_POS = np.array([-0.35, -0.03, 1.3])
BALL_VEL = np.array([3.5, -0.10, 0.5])

BALL_ARRIVE_TIME_EST = 0.55
DURATION = 1.0
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def analyze_trajectory(x0, z0, vx, vz):
    """解析弹道: 球从 (x0, z0) 以 (vx, vz) 飞, 算 z_at_net 和 x_bounce."""
    if vx >= 0:
        return None, None, False, False
    t_net = x0 / (-vx)
    z_at_net = z0 + vz * t_net - 0.5 * G * t_net * t_net
    a = 0.5 * G
    b = -vz
    c = TABLE_Z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        return z_at_net, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2 * a)
    x_bounce = x0 + vx * t_bounce
    clears_net = z_at_net > NET_Z and t_net < t_bounce
    valid = x_bounce < NET_X
    return z_at_net, x_bounce, clears_net, valid


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    bad = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo or y[:, i].max() > hi:
            bad.append(f"yb_{i+1}=[{y[:, i].min():+.3f},{y[:, i].max():+.3f}]")
    return bad


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
    env_origin = scene.env_origins[0].cpu().numpy()

    print(f"\nenv_origin = {env_origin}")
    print(f"paddle body idx = {paddle_idx}")
    print(f"ball preset: pos={BALL_POS}, vel={BALL_VEL}")
    predicted_y = BALL_POS[1] + BALL_VEL[1] * (1.85 / BALL_VEL[0])
    print(f"predicted_y @ robot = {predicted_y:.3f} (should trigger motion_id=2 'right')")

    bad = check_limits(RIGHT)
    if bad:
        print(f"WARNING: 关节超限! {bad}")

    times = np.array([k[0] for k in RIGHT])
    angs = np.array([k[1] for k in RIGHT], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    def run(hit_phase):
        initial_phase = (hit_phase - BALL_ARRIVE_TIME_EST / DURATION) % 1.0
        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(initial_phase)
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

        # 发球
        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device)
        ball_state[0, 0:3] += scene.env_origins[0]
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
        ball_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        sim_dt = float(env.unwrapped.sim.get_physics_dt())
        n_steps = int(1.5 / sim_dt)

        min_gap = 1e9
        min_gap_t = -1
        paddle_vel_at_min = np.zeros(3)
        ball_vel_post = np.zeros(3)
        ball_pos_post = np.zeros(3)
        hit_detected = False

        for step in range(n_steps):
            t = step * sim_dt
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy()
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            if gap < min_gap:
                min_gap = gap
                min_gap_t = t
                paddle_vel_at_min = pv.copy()

            # 检测 hit: ball vx 反转
            if not hit_detected and bv[0] < -0.5 and t > 0.3:
                hit_detected = True
                ball_vel_post = bv.copy()
                ball_pos_post = bp.copy() - env_origin

        if not hit_detected:
            # 用最近接时刻后的状态作参考
            ball_vel_post = bv.copy()
            ball_pos_post = bp.copy() - env_origin

        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            ball_pos_post[0], ball_pos_post[2], ball_vel_post[0], ball_vel_post[2])

        return dict(
            hit_phase=hit_phase,
            min_gap=min_gap, min_gap_t=min_gap_t,
            paddle_vel=paddle_vel_at_min,
            ball_vel_post=ball_vel_post,
            hit_detected=hit_detected,
            z_at_net=z_at_net, x_bounce=x_bounce,
            clears=clears, valid=valid,
        )

    phases = [0.425, 0.450, 0.475, 0.500, 0.525]

    print(f"\n{'phase':<8} {'gap':>5} {'gap_t':>5} "
          f"{'pvx':>6} {'pvy':>6} {'pvz':>6} "
          f"{'bvx':>6} {'bvy':>6} {'bvz':>6} "
          f"{'zn':>6} {'xb':>6} {'CLR':>4} {'HIT':>4}")
    print("=" * 100)

    results = []
    for hp in phases:
        try:
            r = run(hp)
            results.append(r)
        except Exception as e:
            print(f"{hp:<8.3f} ERROR: {e}")
            continue

        pv = r['paddle_vel']
        bv = r['ball_vel_post']
        zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
        xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
        clr = "Y" if r['clears'] and r['valid'] else "N"
        hit = "Y" if r['hit_detected'] else "N"

        print(f"{hp:<8.3f} {r['min_gap']:>5.3f} {r['min_gap_t']:>5.3f} "
              f"{pv[0]:>+6.2f} {pv[1]:>+6.2f} {pv[2]:>+6.2f} "
              f"{bv[0]:>+6.2f} {bv[1]:>+6.2f} {bv[2]:>+6.2f} "
              f"{zn:>6} {xb:>6} {clr:>4} {hit:>4}")

    # Summary
    n_hit = sum(1 for r in results if r['hit_detected'])
    n_clr = sum(1 for r in results if r['clears'] and r['valid'])
    mean_gap = np.mean([r['min_gap'] for r in results]) if results else 0
    mean_bvx = np.mean([r['ball_vel_post'][0] for r in results]) if results else 0

    print(f"\n=== Summary ===")
    print(f"  Hit: {n_hit}/{len(results)}, Clear net: {n_clr}/{len(results)}")
    print(f"  Mean min_gap: {mean_gap:.3f}m, Mean ball_vx: {mean_bvx:+.2f}")
    if n_hit == 0:
        print(f"\n  *** paddle 碰不到球! 需要调整 RIGHT keyframes.")
        print(f"  *** 最近接距离 {mean_gap:.3f}m @ hit_phase=0.475 (中心)")
        if results:
            center = [r for r in results if abs(r['hit_phase'] - 0.475) < 0.01]
            if center:
                c = center[0]
                pv = c['paddle_vel']
                print(f"  *** paddle_vel @ closest: ({pv[0]:+.2f}, {pv[1]:+.2f}, {pv[2]:+.2f})")
                print(f"  *** 需要让 paddle 在 t={c['min_gap_t']:.3f}s 时更靠近球")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
