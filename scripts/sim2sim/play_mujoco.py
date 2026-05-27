"""
Sim2Sim: Play a trained IsaacSim table-tennis policy in Mujoco.

Prerequisites:
    pip install mujoco
    Clone unitree_mujoco for the G1 MJCF model:
        https://github.com/unitreerobotics/unitree_mujoco
    Export policy.pt via play.py in IsaacSim first.

Usage:
    python play_mujoco.py \
        --mjcf /path/to/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
        --policy /path/to/exported/policy.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants from IsaacSim env_cfg / deploy.yaml
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MOTION_DIR = os.path.join(
    PROJECT_ROOT,
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/g1_29dof/forehand",
)
DEFAULT_MOTION_FILES = [
    os.path.join(MOTION_DIR, "forehand_upper.npz"),
    os.path.join(MOTION_DIR, "backhand_upper.npz"),
]

UPPER_BODY_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

ACTION_SCALES = np.array(
    [0.548, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
     0.439, 0.439, 0.439, 0.0745, 0.0745, 0.0745, 0.0745],
    dtype=np.float32,
)

UPPER_BODY_KP = {
    "waist_yaw_joint": 40.18,
    "waist_roll_joint": 28.50,
    "waist_pitch_joint": 28.50,
    "left_shoulder_pitch_joint": 14.25,
    "left_shoulder_roll_joint": 14.25,
    "left_shoulder_yaw_joint": 14.25,
    "left_elbow_joint": 14.25,
    "left_wrist_roll_joint": 14.25,
    "left_wrist_pitch_joint": 16.78,
    "left_wrist_yaw_joint": 16.78,
    "right_shoulder_pitch_joint": 14.25,
    "right_shoulder_roll_joint": 14.25,
    "right_shoulder_yaw_joint": 14.25,
    "right_elbow_joint": 14.25,
    "right_wrist_roll_joint": 14.25,
    "right_wrist_pitch_joint": 16.78,
    "right_wrist_yaw_joint": 16.78,
}

UPPER_BODY_KD = {
    "waist_yaw_joint": 2.56,
    "waist_roll_joint": 1.81,
    "waist_pitch_joint": 1.81,
    "left_shoulder_pitch_joint": 0.907,
    "left_shoulder_roll_joint": 0.907,
    "left_shoulder_yaw_joint": 0.907,
    "left_elbow_joint": 0.907,
    "left_wrist_roll_joint": 0.907,
    "left_wrist_pitch_joint": 1.068,
    "left_wrist_yaw_joint": 1.068,
    "right_shoulder_pitch_joint": 0.907,
    "right_shoulder_roll_joint": 0.907,
    "right_shoulder_yaw_joint": 0.907,
    "right_elbow_joint": 0.907,
    "right_wrist_roll_joint": 0.907,
    "right_wrist_pitch_joint": 1.068,
    "right_wrist_yaw_joint": 1.068,
}

BASE_FIXED_X = 1.5
BASE_FIXED_Z = 0.76
BASE_Y_RANGE = (-1.0, 1.0)
BASE_Y_SLIDER_SCALE = 0.5

STEP_DT = 0.02
SIM_DT = 0.005
DECIMATION = 4

BALL_LAUNCH_X = (-1.0, -0.2)
BALL_LAUNCH_Y = (-0.5, 0.5)
BALL_LAUNCH_Z = (1.0, 1.5)
BALL_LAUNCH_VX = (2.0, 4.0)
BALL_LAUNCH_VY = (-0.5, 0.5)
BALL_LAUNCH_VZ = (0.0, 2.0)

RACKET_BODY_NAME = "right_wrist_yaw_link"


# ---------------------------------------------------------------------------
# MotionLoader — pure NumPy port of UpperBodyMotionLoader
# ---------------------------------------------------------------------------

class MotionLoader:
    def __init__(self, motion_files: list[str]):
        self.motions = []
        for path in motion_files:
            assert os.path.isfile(path), f"Motion file not found: {path}"
            data = np.load(path, allow_pickle=True)
            fps = float(data["fps"])
            dof = data["upper_body_dof"].astype(np.float32)
            base_y = data["base_y"].astype(np.float32)
            joint_names = list(data["joint_names"])
            n = dof.shape[0]
            dt = 1.0 / fps
            dof_vel = np.zeros_like(dof)
            dof_vel[:-1] = (dof[1:] - dof[:-1]) / dt
            dof_vel[-1] = dof_vel[-2]
            base_y_vel = np.zeros_like(base_y)
            base_y_vel[:-1] = (base_y[1:] - base_y[:-1]) / dt
            base_y_vel[-1] = base_y_vel[-2]
            self.motions.append({
                "fps": fps,
                "dof": dof,
                "dof_vel": dof_vel,
                "base_y": base_y,
                "base_y_vel": base_y_vel,
                "joint_names": joint_names,
                "num_frames": n,
                "duration": n / fps,
            })
        self.num_motions = len(self.motions)
        self.num_dof = self.motions[0]["dof"].shape[1]

    def duration(self, motion_id: int) -> float:
        return self.motions[motion_id]["duration"]

    def get_reference(self, phase: float, motion_id: int):
        motion = self.motions[motion_id]
        n = motion["num_frames"]
        frame_f = phase * (n - 1)
        lo = int(np.clip(frame_f, 0, n - 2))
        hi = lo + 1
        alpha = frame_f - lo
        ref_dof = (1 - alpha) * motion["dof"][lo] + alpha * motion["dof"][hi]
        ref_dof_vel = (1 - alpha) * motion["dof_vel"][lo] + alpha * motion["dof_vel"][hi]
        ref_base_y = (1 - alpha) * motion["base_y"][lo] + alpha * motion["base_y"][hi]
        return ref_dof, ref_dof_vel, ref_base_y


# ---------------------------------------------------------------------------
# MujocoScene — modify MJCF XML for table tennis
# ---------------------------------------------------------------------------

class MujocoScene:
    def __init__(self, mjcf_path: str):
        self.mjcf_path = mjcf_path
        self.mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))

    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        tree = ET.parse(self.mjcf_path)
        root = tree.getroot()

        self._set_options(root)
        self._wrap_robot_with_rail(root)
        self._replace_actuators(root)
        self._add_table(root)
        self._add_net(root)
        self._add_ball(root)

        xml_str = ET.tostring(root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml_str, assets=self._load_assets())
        data = mujoco.MjData(model)
        return model, data

    def _load_assets(self) -> dict:
        assets = {}
        for dirpath, _, filenames in os.walk(self.mjcf_dir):
            for fn in filenames:
                if fn.endswith((".stl", ".obj", ".png", ".STL", ".OBJ")):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, self.mjcf_dir)
                    with open(full, "rb") as f:
                        assets[rel] = f.read()
                    assets[fn] = assets[rel]
        return assets

    def _set_options(self, root: ET.Element):
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", str(SIM_DT))
        option.set("gravity", "0 0 -9.81")

    def _wrap_robot_with_rail(self, root: ET.Element):
        worldbody = root.find("worldbody")
        robot_body = None
        for child in list(worldbody):
            if child.tag == "body":
                robot_body = child
                break
        if robot_body is None:
            raise RuntimeError("Cannot find robot body in <worldbody>")

        orig_pos = robot_body.get("pos", "0 0 0")
        pos_vals = [float(v) for v in orig_pos.split()]

        rail_body = ET.Element("body")
        rail_body.set("name", "base_rail")
        rail_body.set("pos", f"{BASE_FIXED_X} 0 {BASE_FIXED_Z}")

        slide_joint = ET.SubElement(rail_body, "joint")
        slide_joint.set("name", "base_y_slide")
        slide_joint.set("type", "slide")
        slide_joint.set("axis", "0 1 0")
        slide_joint.set("limited", "true")
        slide_joint.set("range", "-1.5 1.5")
        slide_joint.set("damping", "50")

        new_z = pos_vals[2] - BASE_FIXED_Z if len(pos_vals) > 2 else 0.0
        new_x = pos_vals[0] - BASE_FIXED_X if len(pos_vals) > 0 else 0.0
        new_y = pos_vals[1] if len(pos_vals) > 1 else 0.0
        robot_body.set("pos", f"{new_x} {new_y} {new_z}")

        worldbody.remove(robot_body)
        rail_body.append(robot_body)
        worldbody.append(rail_body)

    def _replace_actuators(self, root: ET.Element):
        for act_elem in root.findall("actuator"):
            root.remove(act_elem)

        actuator = ET.SubElement(root, "actuator")

        for name in UPPER_BODY_JOINT_NAMES:
            pos_act = ET.SubElement(actuator, "position")
            pos_act.set("name", f"act_{name}")
            pos_act.set("joint", name)
            pos_act.set("kp", str(UPPER_BODY_KP[name]))
            pos_act.set("kv", str(UPPER_BODY_KD[name]))

        for name in LEG_JOINT_NAMES:
            pos_act = ET.SubElement(actuator, "position")
            pos_act.set("name", f"act_{name}")
            pos_act.set("joint", name)
            pos_act.set("kp", "10000")
            pos_act.set("kv", "1000")

    def _add_table(self, root: ET.Element):
        worldbody = root.find("worldbody")
        table = ET.SubElement(worldbody, "body")
        table.set("name", "table_surface")
        table.set("pos", "0 0 0.745")
        geom = ET.SubElement(table, "geom")
        geom.set("type", "box")
        geom.set("size", "1.37 0.7625 0.015")
        geom.set("rgba", "0 0.4 0 1")
        geom.set("friction", "0.3 0.005 0.001")
        geom.set("condim", "3")

    def _add_net(self, root: ET.Element):
        worldbody = root.find("worldbody")
        net = ET.SubElement(worldbody, "body")
        net.set("name", "table_net")
        net.set("pos", "0 0 0.83625")
        geom = ET.SubElement(net, "geom")
        geom.set("type", "box")
        geom.set("size", "0.005 0.915 0.07625")
        geom.set("rgba", "0.9 0.9 0.9 1")

    def _add_ball(self, root: ET.Element):
        worldbody = root.find("worldbody")
        ball = ET.SubElement(worldbody, "body")
        ball.set("name", "ball")
        ball.set("pos", "-1 0 1.2")
        joint = ET.SubElement(ball, "joint")
        joint.set("name", "ball_free")
        joint.set("type", "free")
        geom = ET.SubElement(ball, "geom")
        geom.set("type", "sphere")
        geom.set("size", "0.02")
        geom.set("mass", "0.0027")
        geom.set("rgba", "1 0.5 0 1")
        geom.set("friction", "0.3 0.005 0.001")
        geom.set("solref", "-0.9 0")
        geom.set("condim", "3")


# ---------------------------------------------------------------------------
# PolicyRunner — load JIT or ONNX policy
# ---------------------------------------------------------------------------

class PolicyRunner:
    def __init__(self, path: str, device: str = "cpu"):
        self.device = device
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.session = ort.InferenceSession(path)
            self.mode = "onnx"
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.model = torch.jit.load(path, map_location=device)
            self.model.eval()
            self.mode = "jit"

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "jit":
            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                out = self.model(t)
                return out.squeeze(0).cpu().numpy()
        else:
            out = self.session.run(None, {self.input_name: obs[np.newaxis].astype(np.float32)})
            return out[0].squeeze(0)


# ---------------------------------------------------------------------------
# ObservationBuilder — 105-dim observation matching IsaacSim exactly
# ---------------------------------------------------------------------------

class ObservationBuilder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        motion_loader: MotionLoader,
    ):
        self.model = model
        self.data = data
        self.motion = motion_loader

        self.ub_qpos_ids = []
        self.ub_qvel_ids = []
        self.ub_defaults = []
        for name in UPPER_BODY_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in MJCF")
            self.ub_qpos_ids.append(model.jnt_qposadr[jid])
            self.ub_qvel_ids.append(model.jnt_dofadr[jid])
            self.ub_defaults.append(DEFAULT_JOINT_POS[name])
        self.ub_defaults = np.array(self.ub_defaults, dtype=np.float32)

        slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y_slide")
        self.slide_qpos_adr = model.jnt_qposadr[slide_jid]
        self.slide_dof_adr = model.jnt_dofadr[slide_jid]

        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_rail")
        if self.root_body_id < 0:
            for candidate in ["pelvis", "base_link", "torso_link"]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if bid >= 0:
                    self.root_body_id = bid
                    break
        if self.root_body_id < 0:
            self.root_body_id = 1

        self.ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_adr = model.jnt_qposadr[ball_jid]
        self.ball_dof_adr = model.jnt_dofadr[ball_jid]

        self.racket_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_NAME)
        if self.racket_body_id < 0:
            for candidate in ["right_wrist_yaw", "right_hand_link", "right_wrist_link"]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if bid >= 0:
                    self.racket_body_id = bid
                    break
        if self.racket_body_id < 0:
            print("[WARN] Racket body not found, using root body")
            self.racket_body_id = self.root_body_id

    def compute(self, last_action: np.ndarray, phase: float, motion_id: int) -> np.ndarray:
        ref_dof, ref_dof_vel, ref_base_y = self.motion.get_reference(phase, motion_id)
        motion_cmd = np.concatenate([ref_dof, ref_dof_vel, [ref_base_y]])

        ub_pos_rel = self.data.qpos[self.ub_qpos_ids].astype(np.float32) - self.ub_defaults
        ub_vel = self.data.qvel[self.ub_qvel_ids].astype(np.float32)

        base_y_pos = np.array([self.data.qpos[self.slide_qpos_adr]], dtype=np.float32)
        base_y_vel = np.array([self.data.qvel[self.slide_dof_adr]], dtype=np.float32)

        base_ang_vel = self._base_ang_vel()
        proj_grav = self._projected_gravity()

        ball_pos_rel = self._ball_pos_relative()
        ball_vel_rel = self._ball_vel_relative()

        racket = self.data.xpos[self.racket_body_id].astype(np.float32).copy()

        phase_obs = np.array([phase], dtype=np.float32)

        obs = np.concatenate([
            motion_cmd,        # 35
            ub_pos_rel,        # 17
            ub_vel,            # 17
            base_y_pos,        # 1
            base_y_vel,        # 1
            base_ang_vel,      # 3
            proj_grav,         # 3
            ball_pos_rel,      # 3
            ball_vel_rel,      # 3
            racket,            # 3
            phase_obs,         # 1
            last_action,       # 18
        ])
        return obs.astype(np.float32)

    def _base_ang_vel(self) -> np.ndarray:
        rot = self.data.xmat[self.root_body_id].reshape(3, 3)
        ang_vel_world = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.root_body_id,
            ang_vel_world, True,
        )
        return ang_vel_world[:3].astype(np.float32)

    def _projected_gravity(self) -> np.ndarray:
        rot = self.data.xmat[self.root_body_id].reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return (rot.T @ gravity_world).astype(np.float32)

    def _ball_pos_relative(self) -> np.ndarray:
        ball_pos = self.data.xpos[self.ball_body_id]
        root_pos = self.data.xpos[self.root_body_id]
        return (ball_pos - root_pos).astype(np.float32)

    def _ball_vel_relative(self) -> np.ndarray:
        ball_vel = np.zeros(6)
        root_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id,
            ball_vel, False,
        )
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.root_body_id,
            root_vel, False,
        )
        return (ball_vel[3:6] - root_vel[3:6]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sim2SimRunner — main simulation loop
# ---------------------------------------------------------------------------

class Sim2SimRunner:
    def __init__(self, args):
        self.args = args

        print("[1/4] Loading MJCF and building scene...")
        scene = MujocoScene(args.mjcf)
        self.model, self.data = scene.build()

        print("[2/4] Loading policy...")
        self.policy = PolicyRunner(args.policy, device=args.device)

        motion_files = args.motion_files or DEFAULT_MOTION_FILES
        print(f"[3/4] Loading {len(motion_files)} motion file(s)...")
        self.motion = MotionLoader(motion_files)

        print("[4/4] Initializing observation builder...")
        self.obs = ObservationBuilder(self.model, self.data, self.motion)

        self._init_actuator_ids()

        self.last_action = np.zeros(18, dtype=np.float32)
        self.phase = 0.0
        self.motion_id = 0

    def _init_actuator_ids(self):
        self.ub_actuator_ids = []
        for name in UPPER_BODY_JOINT_NAMES:
            act_name = f"act_{name}"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{act_name}' not found")
            self.ub_actuator_ids.append(aid)

        self.leg_actuator_ids = []
        for name in LEG_JOINT_NAMES:
            act_name = f"act_{name}"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{act_name}' not found")
            self.leg_actuator_ids.append(aid)

    def _apply_action(self, action: np.ndarray):
        for i, aid in enumerate(self.ub_actuator_ids):
            target = action[i] * ACTION_SCALES[i] + DEFAULT_JOINT_POS[UPPER_BODY_JOINT_NAMES[i]]
            self.data.ctrl[aid] = target

        for i, aid in enumerate(self.leg_actuator_ids):
            self.data.ctrl[aid] = DEFAULT_JOINT_POS[LEG_JOINT_NAMES[i]]

        desired_vy = action[17] * BASE_Y_SLIDER_SCALE
        self.data.qvel[self.obs.slide_dof_adr] = desired_vy

    def _reset_ball(self):
        adr = self.obs.ball_qpos_adr
        self.data.qpos[adr:adr + 3] = [
            np.random.uniform(*BALL_LAUNCH_X),
            np.random.uniform(*BALL_LAUNCH_Y),
            np.random.uniform(*BALL_LAUNCH_Z),
        ]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        dadr = self.obs.ball_dof_adr
        self.data.qvel[dadr:dadr + 3] = [
            np.random.uniform(*BALL_LAUNCH_VX),
            np.random.uniform(*BALL_LAUNCH_VY),
            np.random.uniform(*BALL_LAUNCH_VZ),
        ]
        self.data.qvel[dadr + 3:dadr + 6] = [0, 0, 0]

    def _reset_robot(self):
        ref_dof, _, ref_base_y = self.motion.get_reference(self.phase, self.motion_id)

        for i, name in enumerate(UPPER_BODY_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = ref_dof[i]

        for name in LEG_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[name]

        self.data.qpos[self.obs.slide_qpos_adr] = ref_base_y
        self.data.qvel[self.obs.slide_dof_adr] = 0.0

        self.last_action[:] = 0.0

    def _ball_out_of_play(self) -> bool:
        ball_pos = self.data.xpos[self.obs.ball_body_id]
        return ball_pos[2] < 0.0 or abs(ball_pos[0]) > 3.0

    def _reset_episode(self):
        self.phase = np.random.uniform(0, 0.999)
        self.motion_id = np.random.randint(0, self.motion.num_motions)
        self._reset_robot()
        self._reset_ball()
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        self._reset_episode()
        step_count = 0
        max_steps = int(10.0 / STEP_DT)

        print("Starting simulation... (close viewer to exit)")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                obs = self.obs.compute(self.last_action, self.phase, self.motion_id)
                action = self.policy.infer(obs)
                action = np.clip(action, -1.0, 1.0)

                self._apply_action(action)
                self.last_action = action.copy()

                for _ in range(DECIMATION):
                    mujoco.mj_step(self.model, self.data)

                y_pos = self.data.qpos[self.obs.slide_qpos_adr]
                self.data.qpos[self.obs.slide_qpos_adr] = np.clip(y_pos, *BASE_Y_RANGE)

                dur = self.motion.duration(self.motion_id)
                self.phase += STEP_DT / dur
                step_count += 1

                if self.phase >= 1.0 or self._ball_out_of_play() or step_count >= max_steps:
                    self._reset_episode()
                    step_count = 0

                viewer.sync()

                if self.args.real_time:
                    elapsed = time.time() - t0
                    remaining = STEP_DT - elapsed
                    if remaining > 0:
                        time.sleep(remaining)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sim2Sim: play trained policy in Mujoco")
    parser.add_argument("--mjcf", required=True, help="Path to scene_29dof.xml")
    parser.add_argument("--policy", required=True, help="Path to exported policy.pt or policy.onnx")
    parser.add_argument("--motion-files", nargs="+", default=None,
                        help="Paths to motion .npz files (default: project forehand/backhand)")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--real-time", action="store_true", help="Pace simulation to real time")
    args = parser.parse_args()

    runner = Sim2SimRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
