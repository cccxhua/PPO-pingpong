"""对比 MIDDLE vs RIGHT 在击球时刻的 paddle 状态 (位置/速度/面法线).

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/compare_middle_right.py --task X1-TableTennis
"""

import argparse, sys
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
from scipy.spatial.transform import Rotation
import isaaclab_tasks, unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

MIDDLE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
]

RIGHT = [
    (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
    (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),
    (0.475, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),
    (0.540, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),
    (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
    (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
]

# 对应的球参数
BALL_MIDDLE = {"pos": np.array([-0.35, 0.0, 1.3]), "vel": np.array([3.5, 0.0, 0.5])}
BALL_RIGHT = {"pos": np.array([-0.35, -0.03, 1.3]), "vel": np.array([3.5, -0.10, 0.5])}

HIT_PHASE = 0.475
BAT = 0.55
DURATION = 1.0


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
    yb_joint_ids = [robot.find_joints(f"joint_yb_{i}")[0][0] for i in range(1, 8)]
    env_origin = scene.env_origins[0].cpu().numpy()
    sim_dt = float(env.unwrapped.sim.get_physics_dt())

    def run_motion(name, keyframes, ball_params):
        times = np.array([k[0] for k in keyframes])
        angs = np.array([k[1] for k in keyframes], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        initial_phase = (HIT_PHASE - BAT / DURATION) % 1.0
        q0 = spline(initial_phase)
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor(ball_params["pos"], dtype=torch.float32, device=device) + scene.env_origins[0]
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor(ball_params["vel"], dtype=torch.float32, device=device)
        ball_state[0, 10:13] = torch.zeros(3, device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        n_steps = int(1.2 / sim_dt)
        min_gap, min_t = 1e9, -1
        hit = False
        pv_at_hit = np.zeros(3)
        pp_at_hit = np.zeros(3)
        bp_at_hit = np.zeros(3)
        bv_post = np.zeros(3)
        face_at_hit = np.zeros(3)

        # 记录 paddle 覆盖范围 (挥拍期间)
        paddle_trajectory = []

        for step in range(n_steps):
            t = step * sim_dt
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            ft = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                ft[0, jid] = float(target[k])
            robot.set_joint_position_target(ft, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy() - env_origin
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            # 记录挥拍期间 (phase 0.35-0.60) 的 paddle 位置
            if 0.35 < phase < 0.60 or (phase > 0.90 and t < 0.3):
                paddle_trajectory.append(p.copy())

            if gap < min_gap:
                min_gap, min_t = gap, t
                pp_at_hit = p.copy()
                bp_at_hit = bp.copy()
                pv_at_hit = pv.copy()
                # face normal
                pq = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()
                rot = Rotation.from_quat([pq[1], pq[2], pq[3], pq[0]])
                face_at_hit = rot.apply([1, 0, 0])  # local +X = face normal

            if not hit and bv[0] < -0.5 and t > 0.3:
                hit = True
                bv_post = bv.copy()

        if not hit:
            bv_post = bv.copy()

        paddle_traj = np.array(paddle_trajectory) if paddle_trajectory else np.zeros((1, 3))

        return {
            "name": name,
            "min_gap": min_gap, "min_t": min_t, "hit": hit,
            "paddle_pos": pp_at_hit, "ball_pos": bp_at_hit,
            "paddle_vel": pv_at_hit, "ball_vel_post": bv_post,
            "face_normal": face_at_hit,
            "paddle_x_range": (paddle_traj[:, 0].min(), paddle_traj[:, 0].max()),
            "paddle_y_range": (paddle_traj[:, 1].min(), paddle_traj[:, 1].max()),
            "paddle_z_range": (paddle_traj[:, 2].min(), paddle_traj[:, 2].max()),
        }

    print(f"\n{'='*80}")
    print(f"  MIDDLE vs RIGHT 对比分析")
    print(f"{'='*80}")

    results = []
    results.append(run_motion("MIDDLE", MIDDLE, BALL_MIDDLE))
    results.append(run_motion("RIGHT", RIGHT, BALL_RIGHT))

    for r in results:
        pp, bp = r["paddle_pos"], r["ball_pos"]
        pv, bv = r["paddle_vel"], r["ball_vel_post"]
        fn = r["face_normal"]
        print(f"\n--- {r['name']} ---")
        print(f"  Ball: pos={BALL_MIDDLE['pos'] if r['name']=='MIDDLE' else BALL_RIGHT['pos']}, "
              f"vel={BALL_MIDDLE['vel'] if r['name']=='MIDDLE' else BALL_RIGHT['vel']}")
        print(f"  Hit: {'YES' if r['hit'] else 'NO'}, gap={r['min_gap']:.3f}m @ t={r['min_t']:.3f}s")
        print(f"  Paddle pos @ hit: ({pp[0]:+.3f}, {pp[1]:+.3f}, {pp[2]:+.3f})")
        print(f"  Ball pos @ hit:   ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")
        print(f"  Paddle vel @ hit: (vx={pv[0]:+.2f}, vy={pv[1]:+.2f}, vz={pv[2]:+.2f})")
        print(f"    |v|={np.linalg.norm(pv):.2f} m/s")
        print(f"  Face normal (local+X): ({fn[0]:+.3f}, {fn[1]:+.3f}, {fn[2]:+.3f})")
        print(f"    face_X (toward opp): {fn[0]:+.3f}")
        print(f"    face_Z (loft):       {fn[2]:+.3f}")
        if r['hit']:
            print(f"  Ball post-hit: (vx={bv[0]:+.2f}, vy={bv[1]:+.2f}, vz={bv[2]:+.2f})")
            print(f"    |v|={np.linalg.norm(bv):.2f} m/s")
        print(f"\n  挥拍覆盖范围 (phase 0.35-0.60):")
        xr, yr, zr = r["paddle_x_range"], r["paddle_y_range"], r["paddle_z_range"]
        print(f"    X: [{xr[0]:+.3f}, {xr[1]:+.3f}] (width={xr[1]-xr[0]:.3f}m)")
        print(f"    Y: [{yr[0]:+.3f}, {yr[1]:+.3f}] (width={yr[1]-yr[0]:.3f}m)")
        print(f"    Z: [{zr[0]:+.3f}, {zr[1]:+.3f}] (width={zr[1]-zr[0]:.3f}m)")

    # 对比总结
    m, r = results[0], results[1]
    print(f"\n{'='*80}")
    print(f"  关键差异总结")
    print(f"{'='*80}")
    print(f"\n  1. Paddle vz (击球时向上力):")
    print(f"     MIDDLE: pvz={m['paddle_vel'][2]:+.3f} m/s {'↑' if m['paddle_vel'][2]>0 else '↓'}")
    print(f"     RIGHT:  pvz={r['paddle_vel'][2]:+.3f} m/s {'↑' if r['paddle_vel'][2]>0 else '↓'}")
    print(f"     差异: RIGHT pvz 比 MIDDLE {'低' if r['paddle_vel'][2]<m['paddle_vel'][2] else '高'} "
          f"{abs(r['paddle_vel'][2]-m['paddle_vel'][2]):.3f} m/s")

    print(f"\n  2. Face loft (面法线 Z 分量, 决定反射球的上仰角度):")
    print(f"     MIDDLE: face_z={m['face_normal'][2]:+.3f}")
    print(f"     RIGHT:  face_z={r['face_normal'][2]:+.3f}")

    print(f"\n  3. 接球 X 覆盖范围:")
    mx = m["paddle_x_range"]
    rx = r["paddle_x_range"]
    print(f"     MIDDLE: X ∈ [{mx[0]:.3f}, {mx[1]:.3f}], 宽度={mx[1]-mx[0]:.3f}m")
    print(f"     RIGHT:  X ∈ [{rx[0]:.3f}, {rx[1]:.3f}], 宽度={rx[1]-rx[0]:.3f}m")

    print(f"\n  4. 接球 Y 覆盖范围:")
    my = m["paddle_y_range"]
    ry = r["paddle_y_range"]
    print(f"     MIDDLE: Y ∈ [{my[0]:.3f}, {my[1]:.3f}], 宽度={my[1]-my[0]:.3f}m")
    print(f"     RIGHT:  Y ∈ [{ry[0]:.3f}, {ry[1]:.3f}], 宽度={ry[1]-ry[0]:.3f}m")

    print(f"\n  5. 接球 Z 覆盖范围:")
    mz = m["paddle_z_range"]
    rz = r["paddle_z_range"]
    print(f"     MIDDLE: Z ∈ [{mz[0]:.3f}, {mz[1]:.3f}], 宽度={mz[1]-mz[0]:.3f}m")
    print(f"     RIGHT:  Z ∈ [{rz[0]:.3f}, {rz[1]:.3f}], 宽度={rz[1]-rz[0]:.3f}m")

    print(f"\n  6. 总球速对比:")
    if m['hit']:
        print(f"     MIDDLE post-hit |v|={np.linalg.norm(m['ball_vel_post']):.2f} m/s")
    if r['hit']:
        print(f"     RIGHT  post-hit |v|={np.linalg.norm(r['ball_vel_post']):.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
