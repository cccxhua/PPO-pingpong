"""Minimal diagnostic: load A1 robot, command yb_4 to move, print position each step."""
import argparse
import sys
sys.argv = ["diag_minimal.py", "--headless"]

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/root/unitree_rl_lab/source/unitree_rl_lab")
from unitree_rl_lab.assets.robots.a1 import A1_TABLE_TENNIS_CFG

sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
sim = SimulationContext(sim_cfg)

cfg = A1_TABLE_TENNIS_CFG.copy()
cfg.prim_path = "/World/Robot"

robot = Articulation(cfg)

sim.reset()
robot.reset()

joint_names = ["joint_yb_1", "joint_yb_2", "joint_yb_3",
               "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7"]
joint_ids = robot.find_joints(joint_names)[0]
print(f"Joint IDs: {joint_ids}")
print(f"Joint names: {[robot.joint_names[i] for i in joint_ids]}")

robot.update(dt=0.005)

# Manually set joint positions to init_state
init_values = torch.tensor([[1.56, -0.12, -1.70, 1.50, 2.03, 0.00, -0.39]],
                           dtype=torch.float32, device="cuda:0")
robot.write_joint_state_to_sim(init_values, torch.zeros_like(init_values), joint_ids=joint_ids)
robot.set_joint_position_target(init_values, joint_ids=joint_ids)

# Step a few times to settle
for _ in range(10):
    sim.step()
    robot.update(dt=0.005)

init_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
print(f"Initial positions (after set): {init_pos}")

# Check PhysX drive properties
art_view = robot.root_physx_view
stiff = art_view.get_dof_stiffnesses()[0].cpu()
damp = art_view.get_dof_dampings()[0].cpu()
maxf = art_view.get_dof_max_forces()[0].cpu()
maxv = art_view.get_dof_max_velocities()[0].cpu()
for i, jid in enumerate(joint_ids):
    print(f"  {joint_names[i]}: stiff={stiff[jid]:.1f} damp={damp[jid]:.2f} maxF={maxf[jid]:.1f} maxV={maxv[jid]:.1f}")

# Target: move yb_4 from 1.50 to 0.0 (big swing)
target_move = init_values.clone()
target_move[0, 3] = 0.0  # yb_4 target = 0

print(f"\nCommanding yb_4 from {init_pos[3]:.3f} to 0.0")
print(f"Step  | yb4_pos  | yb4_vel  | yb1_pos")
print("-" * 50)

for step in range(200):
    robot.set_joint_position_target(target_move, joint_ids=joint_ids)
    sim.step()
    robot.update(dt=0.005)
    if step % 10 == 0 or step < 5:
        pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
        vel = robot.data.joint_vel[0, joint_ids].cpu().numpy()
        print(f"{step:5d} | {pos[3]:+.4f} | {vel[3]:+.4f} | {pos[0]:+.4f}")

final_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
print(f"\nFinal positions: {final_pos}")
print(f"yb4 moved: {init_pos[3] - final_pos[3]:.4f} rad (from {init_pos[3]:.3f} to {final_pos[3]:.3f})")

simulation_app.close()
