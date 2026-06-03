"""
Sim2Sim: Play a trained X1 table tennis policy in MuJoCo.

Loads the X1 URDF (with embedded mujoco compiler hints), adds table/net/ball,
then runs the policy at 50Hz with the same observation pipeline as deployment.

Prerequisites:
    pip install mujoco numpy torch

Usage:
    python x1_play_mujoco.py \
        --policy /path/to/exported/policy.pt \
        [--real-time]
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Import from deployment script
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deploy"))
from x1_table_tennis_deploy import (
    MotionLoader,
    PolicyRunner,
    PhaseStateMachine,
    ObservationAssembler,
    process_action,
    predict_ball_hit_point,
    compute_ball_time_to_arrive,
    compute_ball_bounce_state,
    DEFAULT_MOTION_FILES,
    RIGHT_ARM_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    JOINT_LIMITS,
    ROBOT_POS,
    ROBOT_X,
    STEP_DT,
    KP, KD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

X1_URDF_PATH = "/root/x1/urdf/x1.urdf"
X1_MESH_DIR = "/root/x1/meshes"

SIM_DT = 0.002
DECIMATION = 10  # 0.002 * 10 = 0.02s = 50Hz

# Ball launch parameters (fixed point, fixed direction for initial testing)
BALL_LAUNCH = {
    "x_range": (-0.35, -0.35),
    "y_range": (0.0, 0.0),
    "z_range": (1.30, 1.30),
    "vx_range": (3.5, 3.5),
    "vy_range": (0.0, 0.0),
    "vz_range": (0.5, 0.5),
}

# Other joints to lock at default
OTHER_JOINTS = {
    "joint_lift": -0.28,
    "joint_zb_1": 0.0, "joint_zb_2": 0.0, "joint_zb_3": 0.0,
    "joint_zb_4": -1.3, "joint_zb_5": 0.0, "joint_zb_6": 0.0, "joint_zb_7": 0.0,
    "joint_head_lr": 0.0, "joint_head_ud": 0.0,
}


# ---------------------------------------------------------------------------
# MuJoCo Scene Builder
# ---------------------------------------------------------------------------

class X1MujocoScene:
    """Build MuJoCo model from X1 URDF with table tennis environment."""

    def __init__(self, urdf_path: str = X1_URDF_PATH):
        self.urdf_path = urdf_path
        self.mesh_dir = os.path.dirname(os.path.dirname(urdf_path))  # /root/x1

    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        xml_str = self._compose_scene_xml()
        assets = self._load_mesh_assets()
        model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
        model.opt.timestep = SIM_DT
        data = mujoco.MjData(model)
        return model, data

    def _compose_scene_xml(self) -> str:
        """Create a complete MJCF scene XML that includes the X1 robot."""
        scene = f"""
<mujoco model="x1_table_tennis">
  <compiler meshdir="{X1_MESH_DIR}" balanceinertia="true" fusestatic="false" angle="radian"/>

  <option timestep="{SIM_DT}" gravity="0 0 -9.81" cone="elliptic" impratio="2">
    <flag multiccd="enable"/>
  </option>

  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="4" friction="0.7 0.005 0.001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light directional="true" pos="2 2 4" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Ground -->
    <geom name="ground" type="plane" size="5 5 0.1" material="grid_mat" pos="0 0 0"/>

    <!-- Table surface -->
    <body name="table_surface" pos="0 0 0.745">
      <geom name="table_geom" type="box" size="1.37 0.7625 0.015"
            rgba="0 0.3 0.6 1" friction="0.35 0.005 0.001"
            solimp="0.95 0.99 0.001" solref="0.004 1"/>
    </body>

    <!-- Net -->
    <body name="table_net" pos="0 0 0.8363">
      <geom name="net_geom" type="box" size="0.005 0.915 0.0763"
            rgba="0.9 0.9 0.9 0.7"/>
    </body>

    <!-- Ball (free body) -->
    <body name="ball" pos="-1 0 1.2">
      <joint name="ball_free" type="free"/>
      <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027"
            rgba="1 0.5 0 1" friction="0.3 0.005 0.001"
            solref="-0.9 0" solimp="0.9 0.99 0.001" condim="4"
            priority="1"/>
    </body>

    <!-- X1 Robot (fixed base) -->
    <include file="x1_robot.xml"/>
  </worldbody>

  <!-- Actuators for right arm (position control) -->
  <actuator>
    <position name="act_joint_yb_1" joint="joint_yb_1" kp="{KP['joint_yb_1']}" kv="{KD['joint_yb_1']}" ctrlrange="-1.053 3.169"/>
    <position name="act_joint_yb_2" joint="joint_yb_2" kp="{KP['joint_yb_2']}" kv="{KD['joint_yb_2']}" ctrlrange="-3.081 0.314"/>
    <position name="act_joint_yb_3" joint="joint_yb_3" kp="{KP['joint_yb_3']}" kv="{KD['joint_yb_3']}" ctrlrange="-2.777 2.762"/>
    <position name="act_joint_yb_4" joint="joint_yb_4" kp="{KP['joint_yb_4']}" kv="{KD['joint_yb_4']}" ctrlrange="-1.911 1.948"/>
    <position name="act_joint_yb_5" joint="joint_yb_5" kp="{KP['joint_yb_5']}" kv="{KD['joint_yb_5']}" ctrlrange="-2.789 2.761"/>
    <position name="act_joint_yb_6" joint="joint_yb_6" kp="{KP['joint_yb_6']}" kv="{KD['joint_yb_6']}" ctrlrange="-1.288 1.508"/>
    <position name="act_joint_yb_7" joint="joint_yb_7" kp="{KP['joint_yb_7']}" kv="{KD['joint_yb_7']}" ctrlrange="-3.14 3.14"/>
  </actuator>
</mujoco>
"""
        # Generate the robot include XML from URDF
        robot_xml = self._urdf_to_mjcf_body()

        # We'll use a single XML string (inline the robot instead of include)
        scene = scene.replace('<include file="x1_robot.xml"/>', robot_xml)
        return scene

    def _urdf_to_mjcf_body(self) -> str:
        """Convert X1 URDF to inline MJCF body XML for the right arm chain only."""
        # Use mujoco's built-in URDF loader, then extract the XML
        # Simpler approach: let mujoco compile URDF directly, then re-export
        try:
            model = mujoco.MjModel.from_xml_path(self.urdf_path)
            xml_bytes = mujoco.mj_saveLastXML("/dev/stdout", model)
        except Exception:
            pass

        # Alternative: use mujoco to compile URDF and get the compiled XML
        # MuJoCo can load URDF directly since the file has <mujoco> hints
        return self._load_urdf_as_body()

    def _load_urdf_as_body(self) -> str:
        """Load URDF via MuJoCo, fix the base, return the robot body XML."""
        # Since X1 URDF has <mujoco> compiler hints, MuJoCo loads it directly.
        # We'll just reference it properly. The simplest approach is to load
        # the full URDF as the scene, adding table/ball programmatically.
        # Return empty — we'll use a different approach below.
        return ""

    def _load_mesh_assets(self) -> dict:
        """Load all STL mesh files for MuJoCo."""
        assets = {}
        for fn in os.listdir(X1_MESH_DIR):
            if fn.endswith((".STL", ".stl", ".obj")):
                path = os.path.join(X1_MESH_DIR, fn)
                with open(path, "rb") as f:
                    assets[fn] = f.read()
        return assets


def build_scene_from_urdf() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load X1 URDF directly (mujoco supports it), then add table/ball programmatically."""
    # MuJoCo can compile URDF with the <mujoco> hints already in the file.
    # We compile it, export to XML, modify, and recompile.
    assets = {}
    for fn in os.listdir(X1_MESH_DIR):
        if fn.endswith((".STL", ".stl")):
            path = os.path.join(X1_MESH_DIR, fn)
            with open(path, "rb") as f:
                assets[fn] = f.read()

    # Load URDF first
    with open(X1_URDF_PATH, "r") as f:
        urdf_str = f.read()

    model_tmp = mujoco.MjModel.from_xml_string(urdf_str, assets=assets)

    # Export to MJCF XML string
    xml_path = "/tmp/x1_compiled.xml"
    mujoco.mj_saveLastXML(xml_path, model_tmp)

    with open(xml_path, "r") as f:
        mjcf_str = f.read()

    # Parse and modify the XML
    root = ET.fromstring(mjcf_str)

    # Fix the robot base (make it kinematic/fixed)
    _fix_robot_base(root)

    # Set simulation options
    _set_sim_options(root)

    # Add table, net, ball
    worldbody = root.find("worldbody")
    _add_ground(worldbody)
    _add_table(worldbody)
    _add_net(worldbody)
    _add_ball(worldbody)
    _add_lights(worldbody)

    # Replace actuators with position controllers for right arm
    _replace_actuators(root)

    # Increase damping on locked joints to prevent instability
    _increase_joint_damping(root)

    # Write back and reload
    final_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(final_xml, assets=assets)
    data = mujoco.MjData(model)
    return model, data


def _fix_robot_base(root: ET.Element):
    """Fix robot at table tennis position by modifying base body."""
    worldbody = root.find("worldbody")
    # Find the robot body (usually first body in worldbody)
    robot_body = None
    for child in worldbody:
        if child.tag == "body":
            robot_body = child
            break
    if robot_body is None:
        return

    # Set position to match IsaacSim: (1.7, 0.14, 0.0)
    robot_body.set("pos", f"{ROBOT_POS[0]} {ROBOT_POS[1]} {ROBOT_POS[2]}")

    # Remove freejoint if any
    for joint in list(robot_body.findall("joint")):
        if joint.get("type", "hinge") == "free":
            robot_body.remove(joint)

    # Fix the lift joint: find it and lock it at -0.28 by making it very stiff
    # Also limit its range tightly around the default
    for joint_elem in root.iter("joint"):
        if joint_elem.get("name") == "joint_lift":
            joint_elem.set("limited", "true")
            joint_elem.set("range", "-0.281 -0.279")


def _set_sim_options(root: ET.Element):
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(SIM_DT))
    option.set("gravity", "0 0 -9.81")
    option.set("cone", "elliptic")
    option.set("impratio", "2")
    option.set("iterations", "50")
    option.set("solver", "Newton")

    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "enable")


def _add_ground(worldbody: ET.Element):
    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "ground")
    geom.set("type", "plane")
    geom.set("size", "5 5 0.1")
    geom.set("rgba", "0.2 0.2 0.2 1")


def _add_table(worldbody: ET.Element):
    body = ET.SubElement(worldbody, "body")
    body.set("name", "table_surface")
    body.set("pos", "0 0 0.745")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "table_geom")
    geom.set("type", "box")
    geom.set("size", "1.37 0.7625 0.015")
    geom.set("rgba", "0 0.3 0.6 1")
    geom.set("friction", "0.35 0.005 0.001")
    geom.set("solimp", "0.95 0.99 0.001")
    geom.set("solref", "0.004 1")


def _add_net(worldbody: ET.Element):
    body = ET.SubElement(worldbody, "body")
    body.set("name", "table_net")
    body.set("pos", "0 0 0.8363")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "net_geom")
    geom.set("type", "box")
    geom.set("size", "0.005 0.915 0.0763")
    geom.set("rgba", "0.9 0.9 0.9 0.7")


def _add_ball(worldbody: ET.Element):
    body = ET.SubElement(worldbody, "body")
    body.set("name", "ball")
    body.set("pos", "-1 0 1.2")
    joint = ET.SubElement(body, "joint")
    joint.set("name", "ball_free")
    joint.set("type", "free")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "ball_geom")
    geom.set("type", "sphere")
    geom.set("size", "0.02")
    geom.set("mass", "0.0027")
    geom.set("rgba", "1 0.5 0 1")
    geom.set("friction", "0.3 0.005 0.001")
    # Negative solref: direct spring-damper [-stiffness, -damping]
    # k=50000 N/m, d=1 Ns/m → for 2.7g ball: damping_ratio≈0.04 → restitution≈0.87
    geom.set("solref", "-50000 -1")
    geom.set("solimp", "0.9 0.95 0.001")
    geom.set("condim", "4")
    geom.set("priority", "1")
    geom.set("margin", "0.001")


def _add_lights(worldbody: ET.Element):
    light1 = ET.SubElement(worldbody, "light")
    light1.set("directional", "true")
    light1.set("pos", "0 0 5")
    light1.set("dir", "0 0 -1")
    light1.set("diffuse", "0.8 0.8 0.8")

    light2 = ET.SubElement(worldbody, "light")
    light2.set("directional", "true")
    light2.set("pos", "2 2 4")
    light2.set("dir", "-0.3 -0.3 -1")
    light2.set("diffuse", "0.4 0.4 0.4")


def _replace_actuators(root: ET.Element):
    """Replace all actuators with position controllers for right arm only."""
    for act_elem in root.findall("actuator"):
        root.remove(act_elem)

    actuator = ET.SubElement(root, "actuator")

    # MuJoCo PD gains — match IsaacSim values
    # If unstable, reduce SIM_DT or increase solver iterations
    mj_kp = {"joint_yb_1": 250.0, "joint_yb_2": 250.0, "joint_yb_3": 250.0,
             "joint_yb_4": 150.0, "joint_yb_5": 150.0, "joint_yb_6": 150.0, "joint_yb_7": 150.0}
    mj_kv = {"joint_yb_1": 1.0, "joint_yb_2": 1.0, "joint_yb_3": 1.0,
             "joint_yb_4": 0.5, "joint_yb_5": 0.5, "joint_yb_6": 0.5, "joint_yb_7": 0.5}

    for name in RIGHT_ARM_JOINT_NAMES:
        lo, hi = JOINT_LIMITS[name]
        pos_act = ET.SubElement(actuator, "position")
        pos_act.set("name", f"act_{name}")
        pos_act.set("joint", name)
        pos_act.set("kp", str(mj_kp[name]))
        pos_act.set("kv", str(mj_kv[name]))
        pos_act.set("ctrlrange", f"{lo} {hi}")

    # Lock other joints with high stiffness
    for name, default_val in OTHER_JOINTS.items():
        pos_act = ET.SubElement(actuator, "position")
        pos_act.set("name", f"lock_{name}")
        pos_act.set("joint", name)
        pos_act.set("kp", "5000")
        pos_act.set("kv", "500")

    # Lock wheels
    for wname in ["joint_right_wheel", "joint_left_wheel"]:
        pos_act = ET.SubElement(actuator, "position")
        pos_act.set("name", f"lock_{wname}")
        pos_act.set("joint", wname)
        pos_act.set("kp", "5000")
        pos_act.set("kv", "500")


def _increase_joint_damping(root: ET.Element):
    """Set appropriate damping and armature for all joints to ensure stability."""
    arm_set = set(RIGHT_ARM_JOINT_NAMES)
    for joint_elem in root.iter("joint"):
        name = joint_elem.get("name", "")
        jtype = joint_elem.get("type", "hinge")
        if not name or jtype == "free":
            continue
        if name in arm_set:
            joint_elem.set("damping", "2.0")
            joint_elem.set("armature", "0.5")
        elif name == "joint_lift":
            joint_elem.set("damping", "200")
            joint_elem.set("armature", "5.0")
        else:
            joint_elem.set("damping", "100")
            joint_elem.set("armature", "1.0")


# ---------------------------------------------------------------------------
# Sim2Sim Runner
# ---------------------------------------------------------------------------

class X1Sim2SimRunner:
    def __init__(self, args):
        self.args = args

        print("[1/3] Building MuJoCo scene from X1 URDF...")
        self.model, self.data = build_scene_from_urdf()

        print("[2/3] Loading policy...")
        self.policy = PolicyRunner(
            policy_path=args.policy,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

        print("[3/3] Loading motion files and initializing...")
        motion_files = args.motion_files or DEFAULT_MOTION_FILES
        self.motion = MotionLoader(motion_files)
        self.phase_machine = PhaseStateMachine(self.motion)
        self.obs_assembler = ObservationAssembler()

        self._init_joint_ids()
        self._init_ball_ids()
        self._init_actuator_ids()

        self.last_action = np.zeros(8, dtype=np.float32)

    def _init_joint_ids(self):
        """Map right arm joint names to MuJoCo qpos/qvel indices."""
        self.arm_qpos_ids = []
        self.arm_qvel_ids = []
        for name in RIGHT_ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in model")
            self.arm_qpos_ids.append(self.model.jnt_qposadr[jid])
            self.arm_qvel_ids.append(self.model.jnt_dofadr[jid])

        # Find paddle body
        self.paddle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Link_yb_paddle"
        )
        if self.paddle_body_id < 0:
            # Try alternate name
            self.paddle_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link_yb_7"
            )
            print(f"[WARN] Link_yb_paddle not found, using link_yb_7 (id={self.paddle_body_id})")

    def _init_ball_ids(self):
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_adr = self.model.jnt_qposadr[ball_jid]
        self.ball_dof_adr = self.model.jnt_dofadr[ball_jid]

    def _init_actuator_ids(self):
        self.arm_actuator_ids = []
        for name in RIGHT_ARM_JOINT_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
            if aid < 0:
                raise RuntimeError(f"Actuator 'act_{name}' not found")
            self.arm_actuator_ids.append(aid)

        # Set locked joints to default
        for name, val in OTHER_JOINTS.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"lock_{name}")
            if aid >= 0:
                self.data.ctrl[aid] = val

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self.arm_qpos_ids], dtype=np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[i] for i in self.arm_qvel_ids], dtype=np.float32)

    def _get_ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].astype(np.float32).copy()

    def _get_ball_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id,
            vel, False,
        )
        return vel[3:6].astype(np.float32)  # linear velocity

    def _get_ball_ang_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id,
            vel, False,
        )
        return vel[0:3].astype(np.float32)  # angular velocity

    def _get_racket_pos(self) -> np.ndarray:
        return self.data.xpos[self.paddle_body_id].astype(np.float32).copy()

    def _apply_targets(self, targets: np.ndarray):
        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = targets[i]

    def _reset_ball(self):
        adr = self.ball_qpos_adr
        self.data.qpos[adr:adr + 3] = [
            np.random.uniform(*BALL_LAUNCH["x_range"]),
            np.random.uniform(*BALL_LAUNCH["y_range"]),
            np.random.uniform(*BALL_LAUNCH["z_range"]),
        ]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        dadr = self.ball_dof_adr
        self.data.qvel[dadr:dadr + 3] = [
            np.random.uniform(*BALL_LAUNCH["vx_range"]),
            np.random.uniform(*BALL_LAUNCH["vy_range"]),
            np.random.uniform(*BALL_LAUNCH["vz_range"]),
        ]
        self.data.qvel[dadr + 3:dadr + 6] = [0, 0, 0]

    def _reset_robot_to_ref(self):
        ref_dof, _, _ = self.phase_machine.get_reference()
        for i, qpos_id in enumerate(self.arm_qpos_ids):
            self.data.qpos[qpos_id] = ref_dof[i]
            self.data.qvel[self.arm_qvel_ids[i]] = 0.0
        self._apply_targets(ref_dof)

    def _ball_out_of_play(self) -> bool:
        ball_pos = self._get_ball_pos()
        return ball_pos[2] < 0.3 or abs(ball_pos[0]) > 3.0 or abs(ball_pos[1]) > 2.0

    def _reset_episode(self):
        ball_pos_init = np.array([
            np.random.uniform(*BALL_LAUNCH["x_range"]),
            np.random.uniform(*BALL_LAUNCH["y_range"]),
            np.random.uniform(*BALL_LAUNCH["z_range"]),
        ], dtype=np.float32)
        ball_vel_init = np.array([
            np.random.uniform(*BALL_LAUNCH["vx_range"]),
            np.random.uniform(*BALL_LAUNCH["vy_range"]),
            np.random.uniform(*BALL_LAUNCH["vz_range"]),
        ], dtype=np.float32)

        self.phase_machine.reset(ball_pos_init, ball_vel_init)
        self._reset_robot_to_ref()
        self._reset_ball()
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        self._reset_episode()
        step_count = 0
        max_steps = int(10.0 / STEP_DT)
        hit_count = 0
        episode_count = 0

        print("Starting sim2sim... (close viewer to exit)")
        print(f"  SIM_DT={SIM_DT}, STEP_DT={STEP_DT}, DECIMATION={DECIMATION}")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 4.0
            viewer.cam.lookat[:] = [1.0, 0.0, 1.0]

            while viewer.is_running():
                t0 = time.time()

                # Get sensor data
                joint_pos = self._get_joint_pos()
                joint_vel = self._get_joint_vel()
                ball_pos = self._get_ball_pos()
                ball_vel = self._get_ball_vel()
                ball_ang_vel = self._get_ball_ang_vel()
                racket_pos = self._get_racket_pos()

                # Assemble observation
                obs = self.obs_assembler.compute(
                    joint_pos, joint_vel,
                    ball_pos, ball_vel, ball_ang_vel,
                    racket_pos, self.phase_machine, self.last_action,
                )

                # Inference
                action = self.policy.infer(obs)
                action = np.clip(action, -1.0, 1.0)

                # Process action
                ref_dof, _, _ = self.phase_machine.get_reference()
                targets, phase_speed = process_action(action, ref_dof)

                # Apply to MuJoCo
                self._apply_targets(targets)

                # Step physics
                for _ in range(DECIMATION):
                    mujoco.mj_step(self.model, self.data)

                # Advance phase
                self.phase_machine.step(phase_speed)
                self.last_action = action.copy()
                step_count += 1

                # Check for ball-paddle contact (simple distance check)
                dist = np.linalg.norm(ball_pos - racket_pos)
                if dist < 0.05 and not self.phase_machine.ball_was_hit:
                    self.phase_machine.mark_hit()
                    hit_count += 1

                # Episode end conditions
                if self._ball_out_of_play() or step_count >= max_steps or \
                   self.phase_machine.is_done:
                    episode_count += 1
                    was_hit = self.phase_machine.ball_was_hit
                    print(f"  Episode {episode_count}: steps={step_count}, "
                          f"hit={'YES' if was_hit else 'no'}, total_hits={hit_count}")
                    self._reset_episode()
                    step_count = 0

                viewer.sync()

                if self.args.real_time:
                    elapsed = time.time() - t0
                    remaining = STEP_DT - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

    @property
    def is_done(self) -> bool:
        return self.phase_machine.swing_done and self.phase_machine.phase_speed == 0.0


# Add is_done property to PhaseStateMachine
PhaseStateMachine.is_done = property(
    lambda self: self.swing_done and self.phase_speed == 0.0
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="X1 Table Tennis Sim2Sim in MuJoCo")
    parser.add_argument("--policy", default=None,
                        help="Path to exported policy.pt or policy.onnx")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model_XXXX.pt (raw checkpoint)")
    parser.add_argument("--motion-files", nargs="+", default=None,
                        help="Paths to motion .npz files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-time", action="store_true",
                        help="Pace simulation to real time")
    args = parser.parse_args()

    if args.policy is None and args.checkpoint is None:
        # Try default path
        default = os.path.join(
            os.path.dirname(SCRIPT_DIR), "..",
            "logs/rsl_rl/x1_tabletennis/2026-05-20_07-26-56/exported/policy.pt"
        )
        default = os.path.abspath(default)
        if os.path.exists(default):
            args.policy = default
            print(f"Using default policy: {default}")
        else:
            print("ERROR: No policy specified. Use --policy or --checkpoint")
            return

    runner = X1Sim2SimRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
