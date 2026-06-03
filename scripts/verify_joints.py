"""Verify joint axes: set reference pose and report paddle world position.

Compare these outputs with the URDF viewer to find which joints are flipped.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate
import isaaclab.sim as sim_utils

from unitree_rl_lab.assets.robots.x1 import X1_TABLE_TENNIS_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = X1_TABLE_TENNIS_CFG.replace(
        prim_path="/World/Robot",
        spawn=X1_TABLE_TENNIS_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)
scene_cfg = SceneCfg(num_envs=1, env_spacing=5.0)
scene = InteractiveScene(scene_cfg)
sim.reset()
scene.reset()

robot = scene["robot"]
paddle_body_id = robot.find_bodies("Link_yb_paddle")[0][0]

JOINT_NAMES = [f"joint_yb_{i}" for i in range(1, 8)]
joint_ids = robot.find_joints(JOINT_NAMES)[0]

READY_POS = [1.56, 0.12, -1.70, 1.50, 2.03, 0.00, -0.39]
HIT_POS   = [1.56, -0.60, -1.70, 1.50, 2.03, 0.00, -0.39]


def set_joints_and_report(values, label=""):
    pos = robot.data.default_joint_pos[0].clone()
    for i, jid in enumerate(joint_ids):
        pos[jid] = float(values[i])
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos.unsqueeze(0), vel.unsqueeze(0))
    for _ in range(20):
        robot.write_data_to_sim()
        sim.step()
        scene.update(dt=0.005)

    paddle_pos = robot.data.body_pos_w[0, paddle_body_id].cpu().numpy()
    paddle_quat = robot.data.body_quat_w[0, paddle_body_id].cpu().numpy()
    actual_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()

    # Compute paddle normal (local +Y rotated by paddle quat)
    local_y = torch.tensor([[0.0, 1.0, 0.0]], device=robot.device)
    quat_t = torch.tensor([paddle_quat], device=robot.device)
    paddle_normal = quat_rotate(quat_t, local_y)[0].cpu().numpy()

    # Compute paddle "up" (local +Z)
    local_z = torch.tensor([[0.0, 0.0, 1.0]], device=robot.device)
    paddle_up = quat_rotate(quat_t, local_z)[0].cpu().numpy()

    print(f"\n{label}")
    print(f"  Joints:  [{', '.join(f'{v:7.3f}' for v in values)}]")
    print(f"  Actual:  [{', '.join(f'{v:7.3f}' for v in actual_pos)}]")
    print(f"  Paddle world pos:    ({paddle_pos[0]:.4f}, {paddle_pos[1]:.4f}, {paddle_pos[2]:.4f})")
    print(f"  Paddle normal (+Y):  ({paddle_normal[0]:.4f}, {paddle_normal[1]:.4f}, {paddle_normal[2]:.4f})")
    print(f"  Paddle up (+Z):      ({paddle_up[0]:.4f}, {paddle_up[1]:.4f}, {paddle_up[2]:.4f})")
    return paddle_pos


print("=" * 70)
print("TEST 1: Reference poses (middle forehand)")
print("=" * 70)
set_joints_and_report(READY_POS, "READY: yb=[1.56, 0.12, -1.70, 1.50, 2.03, 0.00, -0.39]")
set_joints_and_report(HIT_POS,   "HIT:   yb=[1.56, -0.60, -1.70, 1.50, 2.03, 0.00, -0.39]")

print("\n" + "=" * 70)
print("TEST 2: Each joint +0.5 from ready position (one at a time)")
print("=" * 70)
for j in range(7):
    test = list(READY_POS)
    test[j] += 0.5
    set_joints_and_report(test, f"yb_{j+1} += +0.5  (from {READY_POS[j]:.2f} to {test[j]:.2f})")

print("\n" + "=" * 70)
print("TEST 3: Each joint -0.5 from ready position (one at a time)")
print("=" * 70)
for j in range(7):
    test = list(READY_POS)
    test[j] -= 0.5
    set_joints_and_report(test, f"yb_{j+1} += -0.5  (from {READY_POS[j]:.2f} to {test[j]:.2f})")

print("\n" + "=" * 70)
print("TEST 4: All zeros vs ready position")
print("=" * 70)
set_joints_and_report([0]*7, "ALL ZEROS")

simulation_app.close()
