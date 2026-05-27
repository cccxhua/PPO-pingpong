"""播放 X1 参考动作，Isaac Sim 仿真显示。双视角(正前方+左前侧)。

Headless + 录制视频:
  cd /root/unitree_rl_lab
  CONDA_PREFIX="" /isaac-sim/python.sh scripts/play_reference_motion.py --headless --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--npz", type=str, default=None, help="Path to forehand_upper.npz")
parser.add_argument("--loops", type=int, default=3, help="Number of loops to play")
parser.add_argument("--output", type=str, default="scripts/x1_reference_motion.mp4", help="Output video path")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import subprocess
import torch
import numpy as np
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from unitree_rl_lab.assets.robots.x1 import X1_TABLE_TENNIS_CFG


def look_at_quat(cam_pos, target_pos):
    """Compute quaternion (w,x,y,z) for Isaac Lab camera with 'world' convention.

    Isaac Lab 'world' convention: camera looks along local +X, with local +Z as up.
    """
    forward = np.array(target_pos) - np.array(cam_pos)
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, -1.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    left = -right

    # Rotation matrix (local→world): col0=+X_local(forward), col1=+Y_local(left), col2=+Z_local(up)
    R = np.column_stack([forward, left, up])

    # Matrix to quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return tuple(q.tolist())


ROBOT_CENTER = (1.7, 0.0, 0.85)

# Front view: directly in front of robot (robot faces -X)
CAM_FRONT_POS = (-0.3, 0.0, 0.85)
CAM_FRONT_ROT = look_at_quat(CAM_FRONT_POS, ROBOT_CENTER)

# Front-left view: from the left-front diagonal
CAM_LEFT_POS = (0.5, -2.2, 0.85)
CAM_LEFT_ROT = look_at_quat(CAM_LEFT_POS, ROBOT_CENTER)


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
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class SceneWithCameraCfg(SceneCfg):
    cam_front = CameraCfg(
        prim_path="/World/CamFront",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=720,
        offset=CameraCfg.OffsetCfg(
            pos=CAM_FRONT_POS,
            rot=CAM_FRONT_ROT,
            convention="world",
        ),
        update_period=0,
    )
    cam_left = CameraCfg(
        prim_path="/World/CamLeft",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=720,
        offset=CameraCfg.OffsetCfg(
            pos=CAM_LEFT_POS,
            rot=CAM_LEFT_ROT,
            convention="world",
        ),
        update_period=0,
    )


sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)

if args.enable_cameras:
    scene_cfg = SceneWithCameraCfg(num_envs=1, env_spacing=5.0)
else:
    scene_cfg = SceneCfg(num_envs=1, env_spacing=5.0)
scene = InteractiveScene(scene_cfg)

sim.reset()
scene.reset()

robot = scene["robot"]

cam_front = None
cam_left = None
has_camera = False
try:
    cam_front = scene["cam_front"]
    cam_left = scene["cam_left"]
    has_camera = True
except KeyError:
    pass

# Load npz file(s)
if args.npz:
    npz_paths = [args.npz]
else:
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/x1/forehand"
    )
    npz_paths = [
        os.path.join(base, "forehand_middle.npz"),
        os.path.join(base, "forehand_left.npz"),
        os.path.join(base, "forehand_right.npz"),
    ]

motions = []
for p in npz_paths:
    assert os.path.isfile(p), f"NPZ not found: {p}"
    data = np.load(p, allow_pickle=True)
    motions.append({
        "name": os.path.basename(p),
        "fps": float(data["fps"]),
        "dof_frames": data["upper_body_dof"],
        "joint_names": list(data["joint_names"]),
        "num_frames": data["upper_body_dof"].shape[0],
    })
    print(f"Loaded: {p} ({motions[-1]['num_frames']} frames, {motions[-1]['fps']} fps)")

joint_ids = robot.find_joints(motions[0]["joint_names"])[0]
paddle_body_id = robot.find_bodies("Link_yb_paddle")[0][0]

# joint_yb_2 is index 1 in the 7-DOF array; USD axis is +1 but values are in -1 convention
YB2_IDX = 1

sim_dt = 0.005

# Warm up
for _ in range(10):
    robot.write_data_to_sim()
    sim.step()
    scene.update(dt=sim_dt)

# Collect frames
video_frames = []
all_paddle_pos = []
video_fps = motions[0]["fps"]

for mi, motion in enumerate(motions):
    fps = motion["fps"]
    dof_frames = motion["dof_frames"]
    num_frames = motion["num_frames"]
    motion_dt = 1.0 / fps
    steps_per_frame = max(1, int(motion_dt / sim_dt))

    print(f"\nPlaying [{motion['name']}]: {num_frames} frames, {fps} fps")

    for loop in range(args.loops):
        for frame_idx in range(num_frames):
            pos = robot.data.default_joint_pos[0].clone()
            for i, jid in enumerate(joint_ids):
                val = float(dof_frames[frame_idx, i])
                if i == YB2_IDX:
                    val = -val
                pos[jid] = val

            vel = torch.zeros_like(pos)
            robot.write_joint_state_to_sim(pos.unsqueeze(0), vel.unsqueeze(0))

            for step in range(steps_per_frame):
                robot.write_data_to_sim()
                sim.step()
                scene.update(dt=sim_dt)

            paddle_pos = robot.data.body_pos_w[0, paddle_body_id].cpu().numpy()
            all_paddle_pos.append(paddle_pos.copy())

            if has_camera:
                rgb_front = cam_front.data.output.get("rgb")
                rgb_left = cam_left.data.output.get("rgb")
                if rgb_front is not None and rgb_front.numel() > 0:
                    f1 = rgb_front[0].cpu().numpy()[:, :, :3]
                    f2 = rgb_left[0].cpu().numpy()[:, :, :3]
                    combined = np.concatenate([f1, f2], axis=1)
                    video_frames.append(combined.copy())

        print(f"  Loop {loop+1}/{args.loops} done")

# Save video
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.output)
output_path = os.path.abspath(output_path)

if len(video_frames) > 0:
    h, w = video_frames[0].shape[:2]
    print(f"\nEncoding video: {len(video_frames)} frames, {w}x{h}, output: {output_path}")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(int(video_fps)),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in video_frames:
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        print(f"Video saved: {output_path}")
    else:
        err = proc.stderr.read().decode()
        print(f"ffmpeg error: {err[-500:]}")
else:
    print("WARNING: No frames captured! Camera may not be working in this mode.")

# Print trajectory stats
all_paddle_pos = np.array(all_paddle_pos)
print(f"\nPaddle X: [{all_paddle_pos[:,0].min():.3f}, {all_paddle_pos[:,0].max():.3f}]")
print(f"Paddle Y: [{all_paddle_pos[:,1].min():.3f}, {all_paddle_pos[:,1].max():.3f}]")
print(f"Paddle Z: [{all_paddle_pos[:,2].min():.3f}, {all_paddle_pos[:,2].max():.3f}]")

simulation_app.close()
