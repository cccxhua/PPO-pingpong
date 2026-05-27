"""FK probe — for each joint yb_1..7, perturb ±0.5 rad and report paddle delta.

Reveals which joints move the paddle in which world direction at the
current HIT pose, so we know which joints to edit in the keyframe.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
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
import torch

import isaaclab_tasks  # noqa
import unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


HIT_POSE = {
    "joint_yb_1": 1.40,
    "joint_yb_2": 0.25,
    "joint_yb_3": -1.75,
    "joint_yb_4": 1.00,
    "joint_yb_5": 1.10,
    "joint_yb_6": 0.50,
    "joint_yb_7": 1.40,
}


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]

    # find shoulder body for reference
    candidate_shoulder = None
    for name in ["Link_yb_1", "Link_yb_shoulder", "Link_yb_2"]:
        try:
            candidate_shoulder = robot.find_bodies(name)[0][0]
            print(f"[INFO] shoulder = {name} idx={candidate_shoulder}")
            break
        except Exception:
            pass

    env_origin = scene.env_origins[0]

    def set_pose_and_step(joint_pos_dict):
        full_joint_pos = robot.data.default_joint_pos[0:1].clone()
        for jn, val in joint_pos_dict.items():
            jid = robot.find_joints(jn)[0][0]
            full_joint_pos[0, jid] = val
        full_joint_vel = torch.zeros_like(full_joint_pos)
        # write state AND set as target so PD holds it
        robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=torch.tensor([0], device=device))
        robot.set_joint_position_target(full_joint_pos, env_ids=torch.tensor([0], device=device))
        # Step physics to settle - need many steps with low damping
        for _ in range(800):
            robot.set_joint_position_target(full_joint_pos, env_ids=torch.tensor([0], device=device))
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

    # --- Baseline ---
    print("\n=== baseline HIT pose (settled with PD held 4s) ===")
    set_pose_and_step(HIT_POSE)
    paddle_base = robot.data.body_pos_w[0, paddle_idx].clone()
    p_local = (paddle_base - env_origin).cpu().numpy()
    actual_joints = {n: robot.data.joint_pos[0, robot.find_joints(n)[0][0]].item() for n in yb_joint_names}
    print(f"paddle world-local: ({p_local[0]:+.3f}, {p_local[1]:+.3f}, {p_local[2]:+.3f})")
    print(f"actual joint pos vs commanded:")
    for n in yb_joint_names:
        cmd = HIT_POSE[n]
        act = actual_joints[n]
        print(f"  {n}: cmd={cmd:+.3f}, actual={act:+.3f}, err={act-cmd:+.3f}")
    if candidate_shoulder is not None:
        sh = robot.data.body_pos_w[0, candidate_shoulder]
        sh_local = (sh - env_origin).cpu().numpy()
        print(f"shoulder world-local: ({sh_local[0]:+.3f}, {sh_local[1]:+.3f}, {sh_local[2]:+.3f})")

    # --- Perturb each joint ±0.5 rad ---
    print("\n=== single-joint Jacobian (±0.5 rad) ===")
    print(f" {'joint':>10}  {'sign':>4}  {'dx':>7}  {'dy':>7}  {'dz':>7}  (paddle delta in m)")
    for j_name in yb_joint_names:
        for sign in [+1, -1]:
            pose = dict(HIT_POSE)
            pose[j_name] = HIT_POSE[j_name] + sign * 0.5
            set_pose_and_step(pose)
            paddle_new = robot.data.body_pos_w[0, paddle_idx]
            d = (paddle_new - paddle_base).cpu().numpy()
            print(f" {j_name:>10}  {sign:>+4d}  {d[0]:>+7.3f}  {d[1]:>+7.3f}  {d[2]:>+7.3f}")

    # --- Restore baseline ---
    set_pose_and_step(HIT_POSE)

    # --- Try a few joint combos that might bring paddle Y closer to 0 ---
    print("\n=== joint-combo trials targeting paddle world y=0 ===")
    print(f"baseline paddle local y = {p_local[1]:+.3f}, want shift = {-p_local[1]:+.3f}")
    trials = [
        ("yb_4=1.50 + yb_2=+0.30", {"joint_yb_2": 0.30, "joint_yb_4": 1.50}),
        ("yb_4=1.50 + yb_2=+0.30 + yb_3=-2.0", {"joint_yb_2": 0.30, "joint_yb_3": -2.0, "joint_yb_4": 1.50}),
        ("yb_4=1.50 + yb_2=+0.30 + yb_3=-1.5", {"joint_yb_2": 0.30, "joint_yb_3": -1.5, "joint_yb_4": 1.50}),
        ("yb_4=1.40 + yb_2=+0.30 + yb_6=+0.30", {"joint_yb_2": 0.30, "joint_yb_4": 1.40, "joint_yb_6": 0.30}),
        ("yb_4=1.50 + yb_2=+0.30 + yb_6=+0.40", {"joint_yb_2": 0.30, "joint_yb_4": 1.50, "joint_yb_6": 0.40}),
        ("yb_4=1.50 + yb_2=+0.30 + yb_6=+0.20", {"joint_yb_2": 0.30, "joint_yb_4": 1.50, "joint_yb_6": 0.20}),
        ("yb_1=1.45 + yb_2=+0.30 + yb_4=1.50", {"joint_yb_1": 1.45, "joint_yb_2": 0.30, "joint_yb_4": 1.50}),
        ("yb_1=1.50 + yb_2=+0.30 + yb_4=1.40", {"joint_yb_1": 1.50, "joint_yb_2": 0.30, "joint_yb_4": 1.40}),
    ]
    for label, override in trials:
        pose = dict(HIT_POSE)
        pose.update(override)
        set_pose_and_step(pose)
        paddle_new = robot.data.body_pos_w[0, paddle_idx]
        local = (paddle_new - env_origin).cpu().numpy()
        d = (paddle_new - paddle_base).cpu().numpy()
        print(f" {label:<45s} -> ({local[0]:+.3f},{local[1]:+.3f},{local[2]:+.3f})  delta=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
