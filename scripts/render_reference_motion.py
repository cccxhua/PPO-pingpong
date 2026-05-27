"""渲染 X1 机器人播放参考动作的视频。

用 Isaac Sim 离屏渲染，生成机器人挥拍动作的 mp4 视频。
运行: cd /root/unitree_rl_lab && CONDA_PREFIX="" /isaac-sim/python.sh scripts/render_reference_motion.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--npz", type=str, default=None)
parser.add_argument("--loops", type=int, default=3)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--enable_cameras"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from unitree_rl_lab.assets.robots.x1 import X1_TABLE_TENNIS_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = X1_TABLE_TENNIS_CFG.replace(
        prim_path="/World/Robot",
    )
    # Table surface for reference
    table_surface = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.74, 1.525, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.745)),
    )
    # Camera - side view watching the robot arm
    camera = CameraCfg(
        prim_path="/World/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        width=1280,
        height=720,
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, -2.0, 1.5),
            rot=(0.683, 0.183, 0.183, 0.683),
            convention="world",
        ),
        update_period=0,
    )
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

scene_cfg = SceneCfg(num_envs=1, env_spacing=5.0)
scene = InteractiveScene(scene_cfg)

sim.reset()
scene.reset()

robot = scene["robot"]
camera = scene["camera"]

# Load npz
if args.npz:
    npz_path = args.npz
else:
    npz_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/x1/forehand/forehand_upper.npz"
    )

data = np.load(npz_path, allow_pickle=True)
fps = float(data["fps"])
dof_frames = data["upper_body_dof"]
joint_names = list(data["joint_names"])
num_frames = dof_frames.shape[0]

print(f"Motion: {num_frames} frames, {fps} fps, {num_frames/fps:.2f}s")

joint_ids = robot.find_joints(joint_names)[0]

sim_dt = 0.005
motion_dt = 1.0 / fps
steps_per_frame = max(1, int(motion_dt / sim_dt))

# Collect rendered frames
all_frames = []
video_fps = 30

for loop in range(args.loops):
    for frame_idx in range(num_frames):
        pos = robot.data.default_joint_pos[0].clone()
        for i, jid in enumerate(joint_ids):
            pos[jid] = float(dof_frames[frame_idx, i])
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos.unsqueeze(0), vel.unsqueeze(0))

        for step in range(steps_per_frame):
            robot.write_data_to_sim()
            sim.step()
            scene.update(dt=sim_dt)

        # Capture camera frame
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        if rgb is not None and rgb.shape[0] > 0:
            all_frames.append(rgb[:, :, :3].copy())

        if frame_idx % 10 == 0:
            print(f"  Loop {loop+1}/{args.loops}, frame {frame_idx}/{num_frames}")

print(f"Captured {len(all_frames)} frames")

# Save as mp4 using ffmpeg via pipe
output_path = "/root/unitree_rl_lab/scripts/x1_forehand_motion.mp4"

if len(all_frames) > 0:
    import subprocess
    h, w = all_frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(video_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in all_frames:
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Video saved: {output_path}")
else:
    print("ERROR: No frames captured!")

simulation_app.close()
