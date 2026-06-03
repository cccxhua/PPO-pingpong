"""
Sim2Sim: Play a trained A1 table tennis policy in MuJoCo.

Loads the A1 URDF (with embedded mujoco compiler hints), adds table/net/ball,
then runs the policy at 50Hz with the same observation pipeline as deployment.

Coordinate convention: +X toward opponent, robot at -X.

Prerequisites:
    pip install mujoco numpy torch

Usage:
    python a1_play_mujoco.py \
        --policy /path/to/exported/policy.pt \
        [--real-time]
"""

from __future__ import annotations

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET

# Must set GL backend BEFORE importing mujoco when running headless
if "--headless" in os.sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Import from A1 deployment script
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "deploy"))
from a1_table_tennis_deploy import (
    MotionLoader,
    PolicyRunner,
    PhaseStateMachine,
    ObservationAssembler,
    A1RallyController,
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
    HIT_PHASE,
    KP, KD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
A1_BASE_DIR = os.path.join(REPO_ROOT, "A1")
A1_URDF_PATH = os.path.join(A1_BASE_DIR, "urdf/a1.urdf")
A1_MESH_DIR = os.path.join(A1_BASE_DIR, "meshes")
TABLE_USD_PATH = os.path.join(A1_BASE_DIR, "table_tennis_table.usd")
BALL_USD_PATH = os.path.join(A1_BASE_DIR, "ping_pong_ball.usd")

SIM_DT = 0.005
DECIMATION = 4  # 0.005 * 4 = 0.02s = 50Hz control (same as IsaacSim training)

# Ball launch parameters — matching A1 training (ball from +X toward -X)
BALL_LAUNCH = {
    "x_range": (0.30, 0.40),
    "y_range": (-0.08, 0.08),
    "z_range": (1.05, 1.15),
    "vx_range": (-3.2, -2.8),
    "vy_range": (-0.2, 0.2),
    "vz_range": (0.1, 0.3),
}

# Other joints to lock at default
OTHER_JOINTS = {
    "joint_lift": -0.28,
    "joint_zb_1": 0.0, "joint_zb_2": 0.0, "joint_zb_3": 0.0,
    "joint_zb_4": 0.0, "joint_zb_5": 0.0, "joint_zb_6": 0.0, "joint_zb_7": 0.0,
    "joint_head_lr": 0.0, "joint_head_ud": 0.0,
}


# ---------------------------------------------------------------------------
# MuJoCo Scene Builder
# ---------------------------------------------------------------------------

def build_scene_from_urdf() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load A1 URDF directly, then add table/ball programmatically."""
    # Parse USD files for table and ball parameters
    table_cfg = _parse_usd_table(TABLE_USD_PATH)
    ball_cfg = _parse_usd_ball(BALL_USD_PATH)

    # Build assets dict: MuJoCo URDF compiler resolves mesh filenames by BASENAME only.
    # Load all STL files recursively, keyed by basename (right arm takes priority for
    # duplicate names since left arm is locked and only needs visual).
    assets = {}
    for root_dir, _dirs, files in os.walk(A1_MESH_DIR):
        for fn in files:
            if fn.endswith((".STL", ".stl", ".obj")):
                full_path = os.path.join(root_dir, fn)
                # Right arm (A1_r) takes priority over left arm (A1_l) for duplicate basenames
                if fn not in assets or "A1_r" in root_dir:
                    with open(full_path, "rb") as f:
                        assets[fn] = f.read()

    # Read URDF and set meshdir to empty (we provide all meshes via assets dict)
    with open(A1_URDF_PATH, "r") as f:
        urdf_str = f.read()

    # Strip meshdir and directory prefixes from filenames so MuJoCo uses bare basenames
    urdf_str = urdf_str.replace('meshdir="../meshes" ', '')
    urdf_str = re.sub(r'filename="[^"]*?([^/"]+\.STL)"', r'filename="\1"', urdf_str, flags=re.IGNORECASE)

    model_tmp = mujoco.MjModel.from_xml_string(urdf_str, assets=assets)

    # Export to MJCF XML string
    xml_path = "/tmp/a1_compiled.xml"
    mujoco.mj_saveLastXML(xml_path, model_tmp)

    with open(xml_path, "r") as f:
        mjcf_str = f.read()

    # Parse and modify the XML
    root = ET.fromstring(mjcf_str)

    # Fix the robot base (make it kinematic/fixed)
    _fix_robot_base(root)

    # Set simulation options
    _set_sim_options(root)

    # Add table, net, ball from USD specs
    worldbody = root.find("worldbody")
    _add_ground(worldbody, table_cfg)
    _add_table(worldbody, table_cfg)
    _add_net(worldbody, table_cfg)
    _add_ball(worldbody, ball_cfg)
    _add_lights(worldbody)

    # Replace actuators with position controllers for right arm
    _replace_actuators(root)

    # Increase damping on locked joints to prevent instability
    _increase_joint_damping(root)

    # Weld non-controlled joints to lock them rigidly
    _weld_locked_joints(root)

    # Set offscreen framebuffer size for headless rendering
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "720")

    # Write back and reload with mesh assets
    final_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(final_xml, assets=assets)
    data = mujoco.MjData(model)
    return model, data


def _fix_robot_base(root: ET.Element):
    """Fix robot at table tennis position by modifying base body."""
    worldbody = root.find("worldbody")
    robot_body = None
    for child in worldbody:
        if child.tag == "body":
            robot_body = child
            break
    if robot_body is None:
        return

    # A1: robot at (-1.7, 0.0, 0.0), facing +X (identity rotation)
    robot_body.set("pos", f"{ROBOT_POS[0]} {ROBOT_POS[1]} {ROBOT_POS[2]}")
    robot_body.set("quat", "1 0 0 0")

    # Remove freejoint if any
    for joint in list(robot_body.findall("joint")):
        if joint.get("type", "hinge") == "free":
            robot_body.remove(joint)

    # Fix the lift joint: use moderate range to allow compliance flex
    for joint_elem in root.iter("joint"):
        if joint_elem.get("name") == "joint_lift":
            joint_elem.set("limited", "true")
            joint_elem.set("range", "-0.35 0.0")


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


def _parse_usd_table(usd_path: str) -> dict:
    """Parse table_tennis_table.usd to extract geometry and physics parameters."""
    import re
    with open(usd_path, "r") as f:
        content = f.read()

    def _extract_block(name: str) -> str:
        pattern = rf'def \w+ "{name}".*?\{{(.*?)\n    \}}'
        m = re.search(pattern, content, re.DOTALL)
        return m.group(1) if m else ""

    def _get_translate(block: str):
        m = re.search(r'xformOp:translate = \(([\d.e\-]+),\s*([\d.e\-]+),\s*([\d.e\-]+)\)', block)
        return [float(m.group(i)) for i in (1, 2, 3)] if m else [0, 0, 0]

    def _get_scale(block: str):
        m = re.search(r'xformOp:scale = \(([\d.e\-]+),\s*([\d.e\-]+),\s*([\d.e\-]+)\)', block)
        return [float(m.group(i)) for i in (1, 2, 3)] if m else [1, 1, 1]

    def _get_color(block: str):
        m = re.search(r'displayColor = \[\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\]', block)
        return [float(m.group(i)) for i in (1, 2, 3)] if m else [0.5, 0.5, 0.5]

    def _get_float(block: str, key: str):
        m = re.search(rf'{key} = ([\d.]+)', block)
        return float(m.group(1)) if m else None

    # TableTop
    top_block = _extract_block("TableTop")
    top_pos = _get_translate(top_block)
    top_scale = _get_scale(top_block)
    top_color = _get_color(top_block)

    # Net
    net_block = _extract_block("Net")
    net_pos = _get_translate(net_block)
    net_scale = _get_scale(net_block)
    net_color = _get_color(net_block)

    # Ground
    ground_block = _extract_block("GroundPlane")
    ground_color = _get_color(ground_block)

    # Table material
    mat_block = ""
    m = re.search(r'def Material "TableMaterial".*?\{(.*?)\n    \}', content, re.DOTALL)
    if m:
        mat_block = m.group(1)
    restitution = _get_float(mat_block, "physics:restitution") or 0.92
    static_friction = _get_float(mat_block, "physics:staticFriction") or 0.4
    dynamic_friction = _get_float(mat_block, "physics:dynamicFriction") or 0.35

    # Legs
    legs = []
    for leg_name in ["LegFrontLeft", "LegFrontRight", "LegBackLeft", "LegBackRight"]:
        lb = _extract_block(leg_name)
        if lb:
            legs.append({"pos": _get_translate(lb), "scale": _get_scale(lb),
                         "color": _get_color(lb)})

    return {
        "table_pos": top_pos, "table_scale": top_scale, "table_color": top_color,
        "table_restitution": restitution,
        "table_static_friction": static_friction,
        "table_dynamic_friction": dynamic_friction,
        "net_pos": net_pos, "net_scale": net_scale, "net_color": net_color,
        "ground_color": ground_color,
        "legs": legs,
    }


def _parse_usd_ball(usd_path: str) -> dict:
    """Parse ping_pong_ball.usd to extract ball parameters."""
    import re
    with open(usd_path, "r") as f:
        content = f.read()

    def _get_float(key: str):
        m = re.search(rf'{key} = ([\d.]+)', content)
        return float(m.group(1)) if m else None

    def _get_color():
        m = re.search(r'displayColor = \[\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\]', content)
        return [float(m.group(i)) for i in (1, 2, 3)] if m else [1.0, 0.5, 0.0]

    radius = _get_float("double radius") or 0.02
    mass = _get_float("physics:mass") or 0.0027
    restitution = _get_float("physics:restitution") or 0.905
    static_friction = _get_float("physics:staticFriction") or 0.35
    dynamic_friction = _get_float("physics:dynamicFriction") or 0.25
    color = _get_color()

    return {
        "radius": radius, "mass": mass, "restitution": restitution,
        "static_friction": static_friction, "dynamic_friction": dynamic_friction,
        "color": color,
    }


def _add_ground(worldbody: ET.Element, table_cfg: dict):
    color = table_cfg["ground_color"]
    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "ground")
    geom.set("type", "plane")
    geom.set("size", "10 6 0.1")
    geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")


def _add_table(worldbody: ET.Element, table_cfg: dict):
    pos = table_cfg["table_pos"]
    scale = table_cfg["table_scale"]
    color = table_cfg["table_color"]
    half_size = [scale[0] / 2, scale[1] / 2, scale[2] / 2]

    body = ET.SubElement(worldbody, "body")
    body.set("name", "table_surface")
    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "table_geom")
    geom.set("type", "box")
    geom.set("size", f"{half_size[0]} {half_size[1]} {half_size[2]}")
    geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")
    geom.set("friction",
             f"{table_cfg['table_dynamic_friction']} 0.005 0.001")
    geom.set("solimp", "0.95 0.99 0.001")
    geom.set("solref", "0.004 1")

    # Table legs
    for i, leg in enumerate(table_cfg["legs"]):
        leg_body = ET.SubElement(worldbody, "body")
        leg_body.set("name", f"table_leg_{i}")
        leg_body.set("pos", f"{leg['pos'][0]} {leg['pos'][1]} {leg['pos'][2]}")
        lg = ET.SubElement(leg_body, "geom")
        lg.set("type", "box")
        lg.set("size", f"{leg['scale'][0]/2} {leg['scale'][1]/2} {leg['scale'][2]/2}")
        lg.set("rgba", f"{leg['color'][0]} {leg['color'][1]} {leg['color'][2]} 1")


def _add_net(worldbody: ET.Element, table_cfg: dict):
    pos = table_cfg["net_pos"]
    scale = table_cfg["net_scale"]
    color = table_cfg["net_color"]
    half_size = [scale[0] / 2, scale[1] / 2, scale[2] / 2]

    body = ET.SubElement(worldbody, "body")
    body.set("name", "table_net")
    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "net_geom")
    geom.set("type", "box")
    geom.set("size", f"{half_size[0]} {half_size[1]} {half_size[2]}")
    geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")


def _add_ball(worldbody: ET.Element, ball_cfg: dict):
    body = ET.SubElement(worldbody, "body")
    body.set("name", "ball")
    body.set("pos", "0.35 0 1.1")
    joint = ET.SubElement(body, "joint")
    joint.set("name", "ball_free")
    joint.set("type", "free")
    geom = ET.SubElement(body, "geom")
    geom.set("name", "ball_geom")
    geom.set("type", "sphere")
    geom.set("size", str(ball_cfg["radius"]))
    geom.set("mass", str(ball_cfg["mass"]))
    color = ball_cfg["color"]
    geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")
    geom.set("friction",
             f"{ball_cfg['dynamic_friction']} 0.005 0.001")
    geom.set("solref", "-80000 -0.5")
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
    light2.set("pos", "-2 2 4")
    light2.set("dir", "0.3 -0.3 -1")
    light2.set("diffuse", "0.4 0.4 0.4")


def _replace_actuators(root: ET.Element):
    """Replace all actuators with position controllers for right arm only."""
    for act_elem in root.findall("actuator"):
        root.remove(act_elem)

    actuator = ET.SubElement(root, "actuator")

    mj_kp = {"joint_yb_1": 250.0, "joint_yb_2": 250.0, "joint_yb_3": 250.0,
             "joint_yb_4": 120.0, "joint_yb_5": 120.0, "joint_yb_6": 120.0, "joint_yb_7": 120.0}
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

    # Lift joint: high-stiffness position hold
    lift_act = ET.SubElement(actuator, "position")
    lift_act.set("name", "act_joint_lift")
    lift_act.set("joint", "joint_lift")
    lift_act.set("kp", "50000")
    lift_act.set("kv", "5000")
    lift_act.set("ctrlrange", "-0.35 0.0")


def _increase_joint_damping(root: ET.Element):
    """Lock non-controlled joints by constraining their range."""
    arm_set = set(RIGHT_ARM_JOINT_NAMES)
    lock_joints = (set(OTHER_JOINTS.keys()) | {"joint_right_wheel", "joint_left_wheel"}) - {"joint_lift"}

    for joint_elem in root.iter("joint"):
        name = joint_elem.get("name", "")
        jtype = joint_elem.get("type", "hinge")
        if not name or jtype == "free":
            continue
        if name in arm_set:
            joint_elem.set("damping", "1.0")
            joint_elem.set("armature", "0.1")
        elif name == "joint_lift":
            joint_elem.set("damping", "100")
            joint_elem.set("armature", "0.5")
        elif name in lock_joints:
            default_val = OTHER_JOINTS.get(name, 0.0)
            eps = 0.001
            joint_elem.set("limited", "true")
            joint_elem.set("range", f"{default_val - eps} {default_val + eps}")
            joint_elem.set("damping", "500")
            joint_elem.set("armature", "5.0")
        else:
            joint_elem.set("damping", "100")
            joint_elem.set("armature", "1.0")


def _weld_locked_joints(root: ET.Element):
    """Remove non-controlled joints to freeze bodies."""
    arm_set = set(RIGHT_ARM_JOINT_NAMES) | {"ball_free", "joint_lift"}

    for body_elem in root.iter("body"):
        for joint_elem in list(body_elem.findall("joint")):
            name = joint_elem.get("name", "")
            if not name or name in arm_set:
                continue
            body_elem.remove(joint_elem)


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

class URDFForwardKinematics:
    """Lightweight FK for A1 right arm chain: base_link → lift → yb_1..7 → paddle."""

    def __init__(self, robot_pos: np.ndarray):
        self.robot_pos = robot_pos.copy()
        self.chain = [
            # (xyz, rpy, axis, joint_type)
            # joint_lift: base_link → link_lift
            (np.array([-0.0175, 0.0, 1.273331]),
             np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 1.0]), "prismatic"),
            # joint_yb_1: link_lift → link_yb_1
            (np.array([0.0575, -0.1025, -0.03823]),
             np.array([-np.pi/2, 0.0, 0.0]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_2: link_yb_1 → link_yb_2
            (np.array([0.02825, 0.0, -0.0932]),
             np.array([-np.pi/2, 0.0, -np.pi/2]),
             np.array([0.0, 0.0, 1.0]), "revolute"),
            # joint_yb_3: link_yb_2 → link_yb_3
            (np.array([-0.1175, 0.0, -0.02825]),
             np.array([np.pi, np.pi/2, 0.0]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_4: link_yb_3 → link_yb_4
            (np.array([0.0, -0.02825, 0.116]),
             np.array([-np.pi/2, 0.0, -np.pi]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_5: link_yb_4 → link_yb_5
            (np.array([0.0, -0.107, -0.02825]),
             np.array([np.pi/2, 0.0, 0.0]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_6: link_yb_5 → link_yb_6
            (np.array([0.0, -0.02375, 0.114]),
             np.array([np.pi/2, 0.0, 0.0]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_7: link_yb_6 → link_yb_7
            (np.array([0.0, 0.1185, -0.02825]),
             np.array([np.pi/2, 0.0, 0.0]),
             np.array([0.0, 0.0, -1.0]), "revolute"),
            # joint_yb_paddle: link_yb_7 → Link_yb_paddle (fixed)
            (np.array([0.0, 0.0, -0.172]),
             np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0]), "fixed"),
        ]

    @staticmethod
    def _rot_x(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    @staticmethod
    def _rot_y(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    @staticmethod
    def _rot_z(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def _rpy_to_rot(self, rpy: np.ndarray) -> np.ndarray:
        return self._rot_z(rpy[2]) @ self._rot_y(rpy[1]) @ self._rot_x(rpy[0])

    def compute(self, joint_lift: float, joint_angles: np.ndarray) -> np.ndarray:
        """Compute paddle world position from joint states.

        Args:
            joint_lift: lift joint displacement (prismatic, typically -0.28)
            joint_angles: 7 right arm joint angles (rad)

        Returns:
            3D world position of Link_yb_paddle origin
        """
        # A1 at -X facing +X: no rotation needed (identity)
        pos = self.robot_pos.copy()
        rot = np.eye(3)

        # All joint values: lift + 7 arm + 1 fixed
        q_all = [joint_lift] + list(joint_angles) + [0.0]

        for i, (xyz, rpy, axis, jtype) in enumerate(self.chain):
            # Apply fixed transform (joint origin)
            R_joint = self._rpy_to_rot(rpy)
            pos = pos + rot @ xyz
            rot = rot @ R_joint

            # Apply joint motion
            q = q_all[i]
            if jtype == "prismatic":
                pos = pos + rot @ (axis * q)
            elif jtype == "revolute":
                angle = q
                ax = axis / (np.linalg.norm(axis) + 1e-12)
                c, s = np.cos(angle), np.sin(angle)
                K = np.array([[0, -ax[2], ax[1]],
                              [ax[2], 0, -ax[0]],
                              [-ax[1], ax[0], 0]])
                R_q = np.eye(3) + s * K + (1 - c) * (K @ K)
                rot = rot @ R_q

        return pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Sim2Sim Runner
# ---------------------------------------------------------------------------

class A1Sim2SimRunner:
    def __init__(self, args):
        self.args = args

        print("[1/3] Building MuJoCo scene from A1 URDF...")
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

        self.fk = URDFForwardKinematics(ROBOT_POS)
        self._lift_qpos_id = None
        lift_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")
        if lift_jid >= 0:
            self._lift_qpos_id = self.model.jnt_qposadr[lift_jid]

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
            self.paddle_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link_yb_7"
            )
            print(f"[WARN] Link_yb_paddle not found, using link_yb_7 (id={self.paddle_body_id})")

    def _init_ball_ids(self):
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_adr = self.model.jnt_qposadr[ball_jid]
        self.ball_dof_adr = self.model.jnt_dofadr[ball_jid]
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        # Collect all geom IDs belonging to the paddle body and its parent/child
        self.paddle_geom_ids = set()
        paddle_body_names = ["Link_yb_paddle", "link_yb_paddle",
                             "link_yb_7", "Link_yb_7",
                             "link_yb_6", "Link_yb_6"]
        for body_name in paddle_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == bid:
                        self.paddle_geom_ids.add(gid)

        if not self.paddle_geom_ids and self.paddle_body_id >= 0:
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == self.paddle_body_id:
                    self.paddle_geom_ids.add(gid)

        if self.paddle_body_id >= 0:
            for bid in range(self.model.nbody):
                if self.model.body_parentid[bid] == self.paddle_body_id:
                    for gid in range(self.model.ngeom):
                        if self.model.geom_bodyid[gid] == bid:
                            self.paddle_geom_ids.add(gid)

        print(f"  ball_geom_id={self.ball_geom_id}, paddle_geom_ids={self.paddle_geom_ids}")
        paddle_geom_names = []
        for gid in self.paddle_geom_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            paddle_geom_names.append(name or f"geom_{gid}")
        print(f"  paddle geom names: {paddle_geom_names}")

    def _check_ball_paddle_contact(self) -> bool:
        """Check MuJoCo contacts for ball-paddle collision."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.ball_geom_id and g2 in self.paddle_geom_ids:
                return True
            if g2 == self.ball_geom_id and g1 in self.paddle_geom_ids:
                return True
        return False

    def _init_actuator_ids(self):
        self.arm_actuator_ids = []
        for name in RIGHT_ARM_JOINT_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
            if aid < 0:
                raise RuntimeError(f"Actuator 'act_{name}' not found")
            self.arm_actuator_ids.append(aid)

        self.lift_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_joint_lift"
        )
        if self.lift_actuator_id >= 0:
            self.data.ctrl[self.lift_actuator_id] = -0.28
        else:
            print("[WARN] Lift actuator not found")

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self.arm_qpos_ids], dtype=np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        vel = np.array([self.data.qvel[i] for i in self.arm_qvel_ids], dtype=np.float32)
        vel_limits = np.array([8.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)
        return np.clip(vel, -vel_limits, vel_limits)

    def _get_ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].astype(np.float32).copy()

    def _get_ball_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id,
            vel, False,
        )
        return vel[3:6].astype(np.float32)

    def _get_ball_ang_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.ball_body_id,
            vel, False,
        )
        return vel[0:3].astype(np.float32)

    def _get_racket_pos(self) -> np.ndarray:
        """Compute racket position via FK from joint angles (same as real deployment)."""
        joint_angles = self._get_joint_pos()
        lift_q = self.data.qpos[self._lift_qpos_id] if self._lift_qpos_id is not None else -0.28
        fk_pos = self.fk.compute(lift_q, joint_angles)

        mj_pos = self.data.xpos[self.paddle_body_id].astype(np.float32)
        diff = np.abs(fk_pos - mj_pos)
        if diff.max() > 0.01:
            print(f"    [FK WARN] diff={diff}, fk={fk_pos}, mj={mj_pos}")

        return fk_pos

    def _apply_targets(self, targets: np.ndarray):
        """Apply joint targets with velocity-dependent limiting."""
        vel_limits = np.array([8.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0])
        for i, aid in enumerate(self.arm_actuator_ids):
            qvel = abs(self.data.qvel[self.arm_qvel_ids[i]])
            vlim = vel_limits[i]
            if qvel > vlim * 0.7:
                scale = max(0.0, 1.0 - (qvel - vlim * 0.7) / (vlim * 0.3))
                current = self.data.qpos[self.arm_qpos_ids[i]]
                self.data.ctrl[aid] = current + scale * (targets[i] - current)
            else:
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
        """Reset ALL robot joints to their initial positions."""
        default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        for i, qpos_id in enumerate(self.arm_qpos_ids):
            self.data.qpos[qpos_id] = default_pos[i]
            self.data.qvel[self.arm_qvel_ids[i]] = 0.0
        self._apply_targets(default_pos)

        # Reset lift joint
        lift_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")
        if lift_jid >= 0:
            lift_qadr = self.model.jnt_qposadr[lift_jid]
            lift_vadr = self.model.jnt_dofadr[lift_jid]
            self.data.qpos[lift_qadr] = -0.28
            self.data.qvel[lift_vadr] = 0.0
            if self.lift_actuator_id >= 0:
                self.data.ctrl[self.lift_actuator_id] = -0.28

    def _ball_out_of_play(self) -> bool:
        ball_pos = self._get_ball_pos()
        return ball_pos[2] < 0.3 or abs(ball_pos[0]) > 3.0 or abs(ball_pos[1]) > 2.0

    def _correct_phase(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """Closed-loop phase correction based on real-time ball state."""
        if ball_vel[0] > -0.5:
            return
        pm = self.phase_machine
        duration = pm.motion.duration(pm.motion_id)
        t_remain = max(0.05, (ROBOT_X - ball_pos[0]) / ball_vel[0])
        ideal_phase = (HIT_PHASE - t_remain / duration) % 1.0
        error = ideal_phase - pm.phase
        if error > 0.5:
            error -= 1.0
        elif error < -0.5:
            error += 1.0
        correction = np.clip(error * 0.08, -0.003, 0.003)
        pm.phase += correction

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

    def _run_step(self, step_count: int) -> dict:
        """Run one control step. Returns info dict."""
        joint_pos = self._get_joint_pos()
        joint_vel = self._get_joint_vel()
        ball_pos = self._get_ball_pos()
        ball_vel = self._get_ball_vel()
        ball_ang_vel = self._get_ball_ang_vel()
        racket_pos = self._get_racket_pos()

        if self.args.no_policy:
            action = np.zeros(8, dtype=np.float32)
        else:
            obs = self.obs_assembler.compute(
                joint_pos, joint_vel,
                ball_pos, ball_vel, ball_ang_vel,
                racket_pos, self.phase_machine, self.last_action,
            )
            action = self.policy.infer(obs)
            action = np.clip(action, -1.0, 1.0)

            if self.args.diag and step_count % 10 == 0:
                print(f"  [DIAG step={step_count}] phase={self.phase_machine.phase:.3f}")
                print(f"    joint_pos_rel: {obs[15:22]}")
                print(f"    joint_vel:     {obs[22:29]}")
                print(f"    ball_pos_rel:  {obs[29:32]}")
                print(f"    ball_vel_rel:  {obs[32:35]}")
                print(f"    racket_pos:    {obs[38:41]}")
                print(f"    action_out:    {action}")

        if self.args.diag and step_count % 5 == 0:
            dist = np.linalg.norm(ball_pos - racket_pos)
            print(f"    [PROX step={step_count}] ball={ball_pos}, racket={racket_pos}, dist={dist:.3f}m")

        ref_dof, _, _ = self.phase_machine.get_reference()
        targets, phase_speed = process_action(action, ref_dof)

        if self.args.no_policy:
            phase_speed = 1.0

        if self.args.diag and step_count % 10 == 0 and self.args.no_policy:
            print(f"  [REF step={step_count}] phase={self.phase_machine.phase:.3f}")
            print(f"    ref_dof:    {ref_dof}")
            print(f"    actual_pos: {joint_pos}")
            print(f"    error:      {joint_pos - ref_dof}")
            print(f"    joint_vel:  {joint_vel}")

        self._apply_targets(targets)

        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self.phase_machine.step(phase_speed)
        self.last_action = action.copy()

        hit_this_step = False
        if not self.phase_machine.ball_was_hit and self._check_ball_paddle_contact():
            self.phase_machine.mark_hit()
            hit_this_step = True
            print(f"    [HIT] step={step_count}, ball_pos={self._get_ball_pos()}")

        if self.args.debug:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2
                if g1 == self.ball_geom_id or g2 == self.ball_geom_id:
                    other = g2 if g1 == self.ball_geom_id else g1
                    other_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom_{other}"
                    in_paddle = "PADDLE" if other in self.paddle_geom_ids else ""
                    print(f"    ball contact: {other_name} (id={other}) {in_paddle}")

        return {"hit": hit_this_step}

    def run(self):
        """Run with MuJoCo GUI viewer."""
        self._reset_episode()
        step_count = 0
        max_steps = int(10.0 / STEP_DT)
        hit_count = 0
        episode_count = 0

        print("Starting A1 sim2sim... (close viewer to exit)")
        print(f"  SIM_DT={SIM_DT}, STEP_DT={STEP_DT}, DECIMATION={DECIMATION}")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 4.0
            viewer.cam.lookat[:] = [-0.5, 0.0, 1.0]

            while viewer.is_running():
                t0 = time.time()
                info = self._run_step(step_count)
                step_count += 1
                if info["hit"]:
                    hit_count += 1

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

    def run_headless(self, num_episodes: int = 5, video_path: str = None):
        """Run without GUI. Optionally record video."""
        self._reset_episode()
        step_count = 0
        max_steps = int(10.0 / STEP_DT)
        hit_count = 0
        episode_count = 0

        renderer = None
        frames = []
        cam = None
        if video_path:
            try:
                renderer = mujoco.Renderer(self.model, height=720, width=1280)
                cam = mujoco.MjvCamera()
                cam.azimuth = 45
                cam.elevation = -20
                cam.distance = 4.0
                cam.lookat[:] = [-0.5, 0.0, 1.0]
            except Exception as e:
                print(f"  [WARN] Cannot create renderer ({e}), skipping video")
                renderer = None

        print(f"Running A1 headless for {num_episodes} episodes...")
        print(f"  SIM_DT={SIM_DT}, STEP_DT={STEP_DT}, DECIMATION={DECIMATION}")

        while episode_count < num_episodes:
            info = self._run_step(step_count)
            step_count += 1
            if info["hit"]:
                hit_count += 1

            if renderer and step_count % 2 == 0:
                renderer.update_scene(self.data, camera=cam)
                frames.append(renderer.render().copy())

            if self._ball_out_of_play() or step_count >= max_steps or \
               self.phase_machine.is_done:
                episode_count += 1
                was_hit = self.phase_machine.ball_was_hit
                print(f"  Episode {episode_count}: steps={step_count}, "
                      f"hit={'YES' if was_hit else 'no'}, total_hits={hit_count}")
                self._reset_episode()
                step_count = 0

        if renderer and frames and video_path:
            self._save_video(frames, video_path)
            print(f"  Video saved to: {video_path}")

        print(f"\nDone. Total hits: {hit_count}/{num_episodes} episodes")

    def _save_video(self, frames: list, path: str):
        """Save frames as mp4 video."""
        try:
            import imageio
            imageio.mimwrite(path, frames, fps=25, quality=8)
        except ImportError:
            print("  [WARN] imageio not installed, saving as .npy instead")
            np.save(path.replace(".mp4", ".npy"), np.array(frames))

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
    parser = argparse.ArgumentParser(description="A1 Table Tennis Sim2Sim in MuJoCo")
    parser.add_argument("--policy", default=None,
                        help="Path to exported policy.pt or policy.onnx")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model_XXXX.pt (raw checkpoint)")
    parser.add_argument("--motion-files", nargs="+", default=None,
                        help="Paths to motion .npz files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-time", action="store_true",
                        help="Pace simulation to real time")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI viewer (for servers without display)")
    parser.add_argument("--video", default="a1_sim2sim.mp4",
                        help="Save video to this path (requires --headless and imageio)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes in headless mode")
    parser.add_argument("--no-policy", action="store_true",
                        help="Pure reference motion playback (no policy inference)")
    parser.add_argument("--debug", action="store_true",
                        help="Print ball contact debug info every step")
    parser.add_argument("--diag", action="store_true",
                        help="Print observation diagnostics every 10 steps")
    args = parser.parse_args()

    if args.policy is None and args.checkpoint is None:
        default = os.path.join(
            PROJECT_ROOT,
            "logs/rsl_rl/a1_tabletennis/exported/policy.pt"
        )
        if os.path.exists(default):
            args.policy = default
            print(f"Using default policy: {default}")
        else:
            print("ERROR: No policy specified. Use --policy or --checkpoint")
            return

    runner = A1Sim2SimRunner(args)
    if args.headless:
        runner.run_headless(num_episodes=args.episodes, video_path=args.video)
    else:
        runner.run()


if __name__ == "__main__":
    main()
