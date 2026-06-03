"""V83: 排查 training Episode_Reward/ball_return = 0 bug.

V74 ref play 2/5 过网, 但 training 4096 envs 全部 ball_return=0.
ball_land_opponent=0.1171 (无需 ball_was_hit), ball_hit_speed=0.0077 (需 force>0.1+dist<0.25).
按理 ball_return=(ball_x<0) & (vx<-0.5) & ball_was_hit 应能触发.

本探针在 5 个 hit_phase ∈ [0.425, 0.525] (训练 noise 范围) 各跑一次, 完整 trace:
  - force_magnitude (contact sensor) 每 sim step
  - paddle-ball dist
  - hit = (force>0.1) & (dist<0.25)
  - ball_was_hit (OR sticky)
  - ball_x_local, vx_world
  - relaunch trigger (z<0.5 | |x|>3 | (z<0.745 & |v|<0.5))
  - ball_return condition: (ball_x<0) & (vx<-0.5) & ball_was_hit

报告:
  - 球被击中后 force 峰值 (检测 force>0.1 是否常 miss)
  - ball_was_hit 何时变 True / 何时被 reset
  - ball_return 触发时长 (timesteps)
  - 哪一步出 bug (force never > 0.1? proximity miss? premature relaunch?)
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


# V74 (current npz)
V74 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150,  +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

BALL_ARRIVE_TIME_EST = 0.5205
DURATION = 1.0
NET_X = 0.0  # env-relative


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    contact_sensor = scene.sensors["contact_forces"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]

    # contact sensor body id for paddle
    sensor_body_ids = contact_sensor.find_bodies("Link_yb_paddle")[0]
    sensor_paddle_idx = sensor_body_ids[0]

    env_origin = scene.env_origins[0].cpu().numpy()
    print(f"\nenv_origin = {env_origin}")
    print(f"sensor body 'Link_yb_paddle' idx in sensor = {sensor_paddle_idx}")

    times = np.array([k[0] for k in V74])
    angs = np.array([k[1] for k in V74], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    def trace(hit_phase, label):
        initial_phase = (hit_phase - BALL_ARRIVE_TIME_EST) % 1.0
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

        # 复制训练时 ball 发射 (匹配 v82/v81b)
        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
        ball_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        sim_dt = float(env.unwrapped.sim.get_physics_dt())
        n_steps = int(2.0 / sim_dt)

        ball_was_hit = False
        max_force = 0.0
        force_peak_t = -1.0
        force_peak_dist = -1.0

        # 关键时刻
        first_hit_t = -1.0  # ball_was_hit 第一次变 True
        ball_return_steps = 0  # ball_return 触发的 step 数
        relaunch_reset_t = -1.0  # ball_was_hit 被 relaunch reset 的时间
        first_cross_t = -1.0  # ball_x 第一次 < 0
        cross_with_hit = False  # 过网时 ball_was_hit 是否还 True

        # 简化的 relaunch_ball_if_out 检查
        z_min, x_limit, table_z, slow_thresh = 0.5, 3.0, 0.745, 0.5
        relaunch_check_dt = 0.02
        last_relaunch_check = 0.0

        log = []

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

            # paddle / ball state
            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy()
            bp_w = ball.data.root_pos_w[0].cpu().numpy()
            bv_w = ball.data.root_lin_vel_w[0].cpu().numpy()
            bp_local = bp_w - env_origin
            dist = float(np.linalg.norm(p - bp_w))

            # contact sensor force on paddle (history max, like track_ball_hit)
            net_forces = contact_sensor.data.net_forces_w_history  # [num_envs, history, num_bodies, 3]
            if net_forces is not None:
                force_paddle = net_forces[0, :, sensor_paddle_idx, :]  # [history, 3]
                fmag = float(torch.norm(force_paddle, dim=-1).max().item())
            else:
                fmag = 0.0

            if fmag > max_force:
                max_force = fmag
                force_peak_t = t
                force_peak_dist = dist

            hit = (fmag > 0.1) and (dist < 0.25)
            if hit and not ball_was_hit:
                first_hit_t = t
                ball_was_hit = True

            # relaunch_ball_if_out check (simulated, NOT actual)
            if t - last_relaunch_check >= relaunch_check_dt - 1e-6:
                last_relaunch_check = t
                z_bad = bp_local[2] < z_min
                x_bad = abs(bp_local[0]) > x_limit
                v_mag = float(np.linalg.norm(bv_w))
                slow_low = (bp_local[2] < table_z) and (v_mag < slow_thresh)
                out = z_bad or x_bad or slow_low
                if out and ball_was_hit:
                    if relaunch_reset_t < 0:
                        relaunch_reset_t = t
                    ball_was_hit = False
                    log.append((t, "RELAUNCH_RESET", bp_local.copy(), bv_w.copy(), fmag, dist))
                    # 不实际重发球, 只记录 reset 时刻
                    break  # 一旦 reset 就停止 (训练时会重发球)

            # ball_return condition
            if bp_local[0] < NET_X and bv_w[0] < -0.5 and ball_was_hit:
                ball_return_steps += 1

            if first_cross_t < 0 and bp_local[0] < NET_X:
                first_cross_t = t
                cross_with_hit = ball_was_hit

            log.append((t, "step", bp_local.copy(), bv_w.copy(), fmag, dist))

        return dict(
            label=label, hit_phase=hit_phase,
            max_force=max_force, force_peak_t=force_peak_t, force_peak_dist=force_peak_dist,
            first_hit_t=first_hit_t,
            first_cross_t=first_cross_t, cross_with_hit=cross_with_hit,
            ball_return_steps=ball_return_steps,
            relaunch_reset_t=relaunch_reset_t,
            final_was_hit=ball_was_hit,
        )

    print("\n" + "="*120)
    print(f"{'phase':<8} {'fmax':>6} {'fpeak_t':>7} {'fpeak_d':>7} "
          f"{'1st_hit_t':>9} {'1st_cr_t':>9} {'cr_hit':>6} {'rtn_steps':>9} {'reset_t':>8} {'end_hit':>7}")
    print("="*120)

    for hp in [0.425, 0.450, 0.475, 0.500, 0.525]:
        try:
            r = trace(hp, f"hp={hp:.3f}")
        except Exception as e:
            print(f"phase={hp:.3f}  ERROR: {e}")
            continue
        fp_t = f"{r['force_peak_t']:.3f}" if r['force_peak_t'] >= 0 else "  -  "
        fp_d = f"{r['force_peak_dist']:.3f}" if r['force_peak_dist'] >= 0 else "  -  "
        h_t = f"{r['first_hit_t']:.3f}" if r['first_hit_t'] >= 0 else "  -  "
        c_t = f"{r['first_cross_t']:.3f}" if r['first_cross_t'] >= 0 else "  -  "
        rl_t = f"{r['relaunch_reset_t']:.3f}" if r['relaunch_reset_t'] >= 0 else "  -  "
        print(f"{hp:<8.3f} {r['max_force']:>6.2f} {fp_t:>7} {fp_d:>7} "
              f"{h_t:>9} {c_t:>9} {str(r['cross_with_hit']):>6} "
              f"{r['ball_return_steps']:>9} {rl_t:>8} {str(r['final_was_hit']):>7}")

    print("\n=== 解读 ===")
    print("  fmax: paddle 接触力峰值 (>0.1 才算 hit)")
    print("  fpeak_d: 力峰值时刻的 paddle-ball 距离 (<0.25 才在 proximity)")
    print("  1st_hit_t: ball_was_hit 第一次变 True 的时刻 (-表示 force>0.1 + dist<0.25 从未同时满足)")
    print("  1st_cr_t: ball 第一次 ball_x<0 (env-relative) 的时刻")
    print("  cr_hit: 过网时 ball_was_hit 是否 True (False = bug! ball_was_hit 已被 reset 或没 set)")
    print("  rtn_steps: ball_return 条件 (cross + vx<-0.5 + ball_was_hit) 总触发 step 数")
    print("  reset_t: 模拟 relaunch_ball_if_out 重置 ball_was_hit 的时刻")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
