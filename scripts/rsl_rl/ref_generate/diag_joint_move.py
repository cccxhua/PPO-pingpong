"""Diagnostic: directly command A1 arm joints to verify movement."""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import numpy as np
from isaaclab.app import AppLauncher

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
import sys
sys.path.insert(0, "/root/unitree_rl_lab/source/unitree_rl_lab")
from unitree_rl_lab.assets.robots.a1 import A1_TABLE_TENNIS_CFG

sim_cfg = sim_utils.SimulationCfg(dt=0.02, device="cuda:0")
sim = SimulationContext(sim_cfg)

# Override robot position
cfg = A1_TABLE_TENNIS_CFG.copy()
cfg.prim_path = "/World/Robot"
cfg.init_state.pos = (-1.7, 0.14, 0.0)

robot = Articulation(cfg)

sim_utils.build_simulation_scene()
sim.reset()
robot.reset()

# Get right arm joint indices
joint_names = ["joint_yb_1", "joint_yb_2", "joint_yb_3",
               "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7"]
joint_ids = robot.find_joints(joint_names)[0]
print(f"Joint IDs: {joint_ids}")
print(f"Joint names: {[robot.joint_names[i] for i in joint_ids]}")

# Read init positions
robot.update(dt=0.02)
init_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
print(f"Initial joint positions: {init_pos}")

# Define target (hit pose from reference motion)
hit_pose = np.array([2.000, -0.050, -1.750, 0.500, 1.100, -0.700, -0.390])
print(f"Target (hit pose): {hit_pose}")

# Step 50 times with init pose to let PD settle
target = torch.tensor(init_pos, device="cuda:0").unsqueeze(0)
for _ in range(50):
    robot.set_joint_position_target(target, joint_ids=joint_ids)
    sim.step()
    robot.update(dt=0.02)

settled_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
print(f"After 50 steps at init target: {settled_pos}")

# Now command hit pose for 100 steps
target_hit = torch.tensor(hit_pose, dtype=torch.float32, device="cuda:0").unsqueeze(0)
for step in range(100):
    robot.set_joint_position_target(target_hit, joint_ids=joint_ids)
    sim.step()
    robot.update(dt=0.02)
    if step % 20 == 0:
        cur = robot.data.joint_pos[0, joint_ids].cpu().numpy()
        print(f"  step {step:3d}: pos={cur}")

final_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
print(f"After 100 steps at hit target: {final_pos}")
print(f"Delta from init: {final_pos - init_pos}")
print(f"Target error: {hit_pose - final_pos}")

simulation_app.close()
