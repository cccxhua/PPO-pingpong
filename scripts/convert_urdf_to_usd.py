"""Convert X1 URDF to USD using Isaac Sim URDF importer (headless)."""
import argparse
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=str, default="/root/x1/urdf/x1.urdf")
parser.add_argument("--output", type=str, default="/tmp/x1_from_urdf.usd")
args = parser.parse_args()

kit = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import omni.kit.commands
from pxr import Usd, Sdf

status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=args.urdf,
    import_config=import_config,
    get_articulation_root=True,
)

print(f"Import status: {status}, prim_path: {prim_path}")

stage = omni.usd.get_context().get_stage()
stage.Export(args.output)
print(f"USD saved to: {args.output}")

# Verify joints
for prim in stage.Traverse():
    name = prim.GetName()
    if name.startswith("joint_yb_") and prim.GetTypeName() == "PhysicsRevoluteJoint":
        rot0 = prim.GetAttribute("physics:localRot0").Get()
        lo = prim.GetAttribute("physics:lowerLimit").Get()
        hi = prim.GetAttribute("physics:upperLimit").Get()
        w = rot0.GetReal()
        im = rot0.GetImaginary()
        print(f"  {name}: rot0=(w={w:.4f}, xyz=({im[0]:.4f}, {im[1]:.4f}, {im[2]:.4f})), limits=[{lo:.2f}, {hi:.2f}]")

kit.close()
