"""Play pure reference motion (zero residual) — diagnostic script.

Bypasses the policy entirely: feeds zero actions to the env so the robot
tracks `ref_dof` exactly with phase_speed=1.0. Use this to inspect what
the reference motion *itself* produces — independent of any trained policy.

== RIGHT MOTION 开发规范 ==
- RIGHT motion (正手) 基准姿态来自 forehand_right_v50.npz 初始帧:
    [+0.440, +1.300, -0.800, -1.350, +0.000, +0.720, -0.140]
    (yb1=0.44, yb2=1.3, yb3=-0.8, yb4=-1.35, yb5=0, yb6=0.72, yb7=-0.14)
- 挥拍方向: 球拍向 +Y, -X 方向挥动 (yb2 增大 = +Y, yb3 减小 = -X)
- 不要使用 MIDDLE (反手) 的 production posture!
- ball_preset "right": 球从对面中间发出 (y=0), vy 使球到达 x≈1.2 时 y≈-0.5
    → 需要 vy≈-1.1 (因为飞行时间约 0.45s, 0.45*1.1≈0.5)
- 每次生成新版本视频: --output_dir logs/pure_ref/right/<version_name>

Usage:
    python scripts/rsl_rl/play_pure_ref.py --task X1-TableTennis --video --video_length 400

    # play a custom npz (overrides cfg's motion_files, disables ball terminations,
    # starts phase at 0 so the whole clip plays once):
    python scripts/rsl_rl/play_pure_ref.py --task X1-TableTennis \\
        --npz logs/pure_ref/realpingpong.npz \\
        --video --video_length 2800
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play pure reference motion (no policy).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="logs/pure_ref")
parser.add_argument(
    "--npz",
    type=str,
    default=None,
    help="Path to a single motion npz to play instead of the cfg's default motion_files. "
         "Disables ball-related terminations and starts phase at 0 so the full clip plays once.",
)
parser.add_argument(
    "--ball_preset",
    type=str,
    default=None,
    choices=["middle", "left", "right", "high"],
    help="Override ball launch params for testing specific motions. "
         "Keeps hit_phase from cfg (unlike --npz which sets hit_phase=0).",
)
parser.add_argument(
    "--arrive_time",
    type=float,
    default=None,
    help="Override ball_arrive_time_est for phase alignment tuning.",
)
parser.add_argument(
    "--hit_phase",
    type=float,
    default=None,
    help="Override commands.motion.hit_phase (pure-ref only) to pick which npz "
         "frame is the designated contact pose. Lets contact land on a higher / "
         "more-forward paddle frame than the cfg default (0.475).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if "--headless" not in sys.argv:
        args_cli.headless = True
        sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.dict import print_dict

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Disable randomization for pure reference playback
    if hasattr(env_cfg, 'events'):
        if hasattr(env_cfg.events, 'randomize_gains'):
            env_cfg.events.randomize_gains = None
        if hasattr(env_cfg.events, 'randomize_effort'):
            env_cfg.events.randomize_effort = None

    # Override lift column height
    env_cfg.scene.robot.init_state.joint_pos["joint_lift"] = -0.28

    # Match from_csv init: all joints from real policy episode 0
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_1"] = 1.769
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_2"] = -0.762
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_3"] = -1.863
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_4"] = 1.445
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_5"] = 0.206
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_6"] = -0.827
    env_cfg.scene.robot.init_state.joint_pos["joint_yb_7"] = 1.043

    # Phase alignment for v56c motion (105 frames, hit_frame=51)
    env_cfg.commands.motion.hit_phase = 0.486
    env_cfg.commands.motion.hit_phase_noise = 0.0
    env_cfg.commands.motion.ball_arrive_time_est = 0.65
    env_cfg.commands.motion.ball_arrive_time_noise = 0.0

    if args_cli.npz is not None:
        npz_abs = os.path.abspath(args_cli.npz)
        assert os.path.isfile(npz_abs), f"--npz file not found: {npz_abs}"
        if args_cli.ball_preset is None:
            env_cfg.commands.motion.motion_files = [npz_abs]
            env_cfg.commands.motion.match_ball_direction = False
            env_cfg.commands.motion.hit_phase = 0.0
            env_cfg.commands.motion.hit_phase_noise = 0.0
            env_cfg.terminations.ball_on_own_table = None
            env_cfg.terminations.ball_missed_paddle = None
            env_cfg.events.relaunch_ball = None
            if hasattr(env_cfg.events, 'reset_ball'):
                env_cfg.events.reset_ball = None
        else:
            env_cfg.commands.motion.motion_files = [npz_abs, npz_abs, npz_abs]
            env_cfg.commands.motion.hit_phase_noise = 0.0
        print(f"[INFO] --npz override: motion_files = [{npz_abs}]")

    if args_cli.ball_preset is not None:
        # Infer robot_side for ball presets
        _rs = robot_side if 'robot_side' in dir() else 1
        _rp = env_cfg.scene.robot.init_state.pos[0]
        _rs = 1 if _rp > 0 else -1
        BALL_PRESETS = {
            "middle": dict(x_range=(-0.35 * _rs, -0.35 * _rs), y_range=(0.0, 0.0),
                           z_range=(1.10, 1.10), vx_range=(3.0 * _rs, 3.0 * _rs),
                           vy_range=(0.0, 0.0), vz_range=(0.2, 0.2)),
            "left":   dict(x_range=(-0.35 * _rs, -0.35 * _rs), y_range=(+0.3, +0.3),
                           z_range=(1.28, 1.28), vx_range=(2.5 * _rs, 2.5 * _rs),
                           vy_range=(+0.4, +0.4), vz_range=(0.4, 0.4)),
            "right":  dict(x_range=(-0.35 * _rs, -0.35 * _rs), y_range=(+0.10, +0.10),
                           z_range=(1.28, 1.28), vx_range=(2.5 * _rs, 2.5 * _rs),
                           vy_range=(+0.30, +0.30), vz_range=(0.4, 0.4)),
            # "high": moderate-high/slow ball. Paired with --hit_phase ~0.54 (later
            # contact pose: paddle higher, more +X, lower faceY) + --arrive_time ~0.70
            # so the ball sits at x~-1.42 z~0.95 when the paddle reaches that frame.
            # "high": moderate-high/slow ball that arrives near apex at the paddle's
            # far-reach zone. Best result with --hit_phase 0.54 --arrive_time 0.70:
            # clean high contact (z~1.02, gap 3.7cm), ball returns +X and reaches the
            # net ~3cm short of clearing (speed-limited, see logs/pure_ref/whip_high3).
            "high":   dict(x_range=(-0.35 * _rs, -0.35 * _rs), y_range=(0.0, 0.0),
                           z_range=(1.16, 1.16), vx_range=(3.45 * _rs, 3.45 * _rs),
                           vy_range=(0.0, 0.0), vz_range=(0.55, 0.55)),
        }
        bp = BALL_PRESETS[args_cli.ball_preset]
        env_cfg.events.reset_ball.params.update({"ball_cfg": env_cfg.events.reset_ball.params["ball_cfg"], **bp})
        env_cfg.events.relaunch_ball.params.update({"ball_cfg": env_cfg.events.relaunch_ball.params["ball_cfg"], **bp})
        env_cfg.terminations.ball_missed_paddle = None
        env_cfg.terminations.ball_on_own_table = None
        env_cfg.events.relaunch_ball.interval_range_s = (3.5, 3.5)
        print(f"[INFO] --ball_preset={args_cli.ball_preset}: {bp}")

    if args_cli.arrive_time is not None:
        env_cfg.commands.motion.ball_arrive_time_est = args_cli.arrive_time
        env_cfg.commands.motion.ball_arrive_time_noise = 0.0
        print(f"[INFO] ball_arrive_time_est overridden to {args_cli.arrive_time}")

    if args_cli.hit_phase is not None:
        env_cfg.commands.motion.hit_phase = args_cli.hit_phase
        env_cfg.commands.motion.hit_phase_noise = 0.0
        print(f"[INFO] hit_phase overridden to {args_cli.hit_phase}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        out_dir = os.path.abspath(args_cli.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": out_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording pure-reference video to:", out_dir)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    zero_action = torch.zeros((args_cli.num_envs, action_dim), device=device)

    print(f"[INFO] action_dim = {action_dim}, feeding zeros (residual=0, phase_speed=1.0).")

    scene = env.unwrapped.scene
    robot = scene["robot"]
    act = robot.actuators["right_arm"]
    print(f"  [ACT] right_arm stiffness: {act.stiffness[0].cpu().tolist()}")
    print(f"  [ACT] right_arm damping:   {act.damping[0].cpu().tolist()}")
    print(f"  [ACT] right_arm effort:    {act.effort_limit[0].cpu().tolist()}")
    print(f"  [ACT] right_arm vel_limit: {act.velocity_limit[0].cpu().tolist()}")
    jids_yb = robot.find_joints(["joint_yb_1","joint_yb_2","joint_yb_3","joint_yb_4","joint_yb_5","joint_yb_6","joint_yb_7"])[0]
    print(f"  [JNT] right arm joint_ids: {jids_yb}")
    print(f"  [JNT] all joint_names: {robot.joint_names}")
    # Check PhysX drive properties directly
    import omni.physics.tensors as pt
    art_view = robot._root_physx_view
    print(f"  [PHX] dof_stiffnesses: {art_view.get_dof_stiffnesses()[0].cpu().tolist()}")
    print(f"  [PHX] dof_dampings: {art_view.get_dof_dampings()[0].cpu().tolist()}")
    print(f"  [PHX] dof_max_forces: {art_view.get_dof_max_forces()[0].cpu().tolist()}")
    print(f"  [PHX] dof_max_velocities: {art_view.get_dof_max_velocities()[0].cpu().tolist()}")
    ball = scene["ball"]
    NET_X = 0.0
    NET_Z = 0.94
    TELEPORT_THRESH = 0.30  # ball position jump > 0.3m means it was relaunched

    # Detect robot side from env config
    robot_side = getattr(env_cfg, '_robot_side', None)
    if robot_side is None:
        # Infer from robot position: if robot x < 0, robot_side = -1
        robot_pos_x = env_cfg.scene.robot.init_state.pos[0]
        robot_side = 1 if robot_pos_x > 0 else -1

    num_trials = 0
    num_clears = 0
    num_direct_clears = 0  # ball clears net WITHOUT bouncing on own table after being hit
    cleared_this_trial = False
    bounced_own_table = False
    ball_was_hit_back = False  # ball started moving away from robot (was hit by paddle)
    prev_bx = None
    prev_bz = None
    prev_pos = None
    TABLE_Z = 0.78  # ball z < this means it touched table surface
    OWN_TABLE_X_MIN = 0.0 if robot_side > 0 else -1.37
    OWN_TABLE_X_MAX = 1.37 if robot_side > 0 else 0.0

    timestep = 0
    # Debug: track racket-ball gap at hit phase
    robot = scene["robot"]
    racket_body_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    min_gap = 999.0
    min_gap_step = 0
    min_gap_racket = None
    min_gap_ball = None
    min_gap_face_normal = None
    min_gap_joints = None  # (actual[7], target[7]) at the closest-approach (contact) step
    steps_in_trial = 0
    _debug_printed = False

    # Torque logging
    torque_log = []  # list of (step, [torque_yb1..yb7])
    # Paddle trajectory logging: world pos (local to env) + face-normal axes per step
    paddle_log = []  # list of (step, px,py,pz, nx,ny,nz) where n = paddle local +X in world
    # Actual joint-angle logging (same step key as paddle_log) -> correlate joints<->paddle, no phase guess
    joints_log = []  # list of (step, [act_yb1..7], [tgt_yb1..7])
    # Ball trajectory logging (same step key) -> pin the TRUE contact frame
    ball_log = []  # list of (step, bx, by, bz)
    import isaaclab.utils.math as _mu

    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_action)

        # Record torque for right arm joints
        jids = robot.find_joints(["joint_yb_1","joint_yb_2","joint_yb_3","joint_yb_4","joint_yb_5","joint_yb_6","joint_yb_7"])[0]
        pos = robot.data.joint_pos[0, jids]
        tgt = robot.data.joint_pos_target[0, jids]
        vel = robot.data.joint_vel[0, jids]
        stiff = act.stiffness[0, :7]
        damp = act.damping[0, :7]
        effort_lim = act.effort_limit[0, :7]
        torque_raw = stiff * (tgt - pos) - damp * vel
        torque_clamped = torch.clamp(torque_raw, -effort_lim, effort_lim)
        torque_log.append((timestep, torque_clamped.cpu().numpy().copy()))
        joints_log.append((timestep, pos.cpu().numpy().copy(), tgt.cpu().numpy().copy()))

        # DEBUG: print joint targets vs actual positions for first 30 steps of first trial
        if not _debug_printed and steps_in_trial < 40 and num_trials == 0:
            jids = robot.find_joints(["joint_yb_1","joint_yb_2","joint_yb_3","joint_yb_4","joint_yb_5","joint_yb_6","joint_yb_7"])[0]
            actual = robot.data.joint_pos[0, jids].cpu().numpy()
            target = robot.data.joint_pos_target[0, jids].cpu().numpy()
            # Also read PhysX drive target directly
            art_view = robot._root_physx_view
            phx_targets = art_view.get_dof_position_targets()[0].cpu().numpy()
            phx_yb4_target = phx_targets[11]  # index 11 = joint_yb_4
            phx_yb1_target = phx_targets[5]   # index 5 = joint_yb_1
            if steps_in_trial in (0, 10, 20, 25, 28, 29, 35):
                print(f"  [DBG] step={steps_in_trial:3d} tgt_yb4={target[3]:+.3f} act_yb4={actual[3]:+.3f} PHX_tgt_yb4={phx_yb4_target:+.3f} | tgt_yb1={target[0]:+.3f} PHX_tgt_yb1={phx_yb1_target:+.3f}")
            if steps_in_trial == 39:
                _debug_printed = True

        ball_pos_local = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
        ball_log.append((timestep, float(ball_pos_local[0]), float(ball_pos_local[1]), float(ball_pos_local[2])))
        racket_pos_w = robot.data.body_pos_w[0, racket_body_idx]
        racket_pos_local = (racket_pos_w - scene.env_origins[0]).cpu().numpy()
        # Log paddle world trajectory + ALL THREE local axes in world (to identify true face normal)
        _rq = robot.data.body_quat_w[0, racket_body_idx]
        _ax = _mu.quat_apply(_rq.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=_rq.device))[0].cpu().numpy()
        _ay = _mu.quat_apply(_rq.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=_rq.device))[0].cpu().numpy()
        _az = _mu.quat_apply(_rq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=_rq.device))[0].cpu().numpy()
        paddle_log.append((timestep, float(racket_pos_local[0]), float(racket_pos_local[1]), float(racket_pos_local[2]),
                           float(_ax[0]), float(_ax[1]), float(_ax[2]),
                           float(_ay[0]), float(_ay[1]), float(_ay[2]),
                           float(_az[0]), float(_az[1]), float(_az[2])))
        gap = float(((ball_pos_local - racket_pos_local) ** 2).sum() ** 0.5)
        steps_in_trial += 1
        if gap < min_gap:
            min_gap = gap
            min_gap_step = steps_in_trial
            min_gap_racket = racket_pos_local.copy()
            min_gap_ball = ball_pos_local.copy()
            # compute face normal: try all local axes
            import isaaclab.utils.math as math_utils
            racket_quat_w = robot.data.body_quat_w[0, racket_body_idx]  # (4,)
            local_x = torch.tensor([[1.0, 0.0, 0.0]], device=racket_quat_w.device)
            local_y = torch.tensor([[0.0, 1.0, 0.0]], device=racket_quat_w.device)
            local_z = torch.tensor([[0.0, 0.0, 1.0]], device=racket_quat_w.device)
            fn_x = math_utils.quat_apply(racket_quat_w.unsqueeze(0), local_x)[0].cpu().numpy()
            fn_y = math_utils.quat_apply(racket_quat_w.unsqueeze(0), local_y)[0].cpu().numpy()
            fn_z = math_utils.quat_apply(racket_quat_w.unsqueeze(0), local_z)[0].cpu().numpy()
            min_gap_face_normal = f"X={fn_x} Y={fn_y} Z={fn_z}"
            min_gap_joints = (pos.cpu().numpy().copy(), tgt.cpu().numpy().copy())

        bx, bz = float(ball_pos_local[0]), float(ball_pos_local[2])

        teleported = (
            prev_pos is not None
            and float(((ball_pos_local - prev_pos) ** 2).sum() ** 0.5) > TELEPORT_THRESH
        )
        if teleported:
            num_trials += 1
            if cleared_this_trial:
                num_clears += 1
                if not bounced_own_table:
                    num_direct_clears += 1
            rate = num_clears / num_trials * 100.0
            direct_rate = num_direct_clears / num_trials * 100.0
            print(f"[trial {num_trials:4d}] cleared={cleared_this_trial} direct={cleared_this_trial and not bounced_own_table}  "
                  f"rate = {num_clears}/{num_trials} = {rate:.1f}%  "
                  f"direct = {num_direct_clears}/{num_trials} = {direct_rate:.1f}%")
            print(f"  min_gap={min_gap:.3f}m at step {min_gap_step}  "
                  f"racket={min_gap_racket}  ball={min_gap_ball}"
                  f"  face_normal={min_gap_face_normal}")
            cleared_this_trial = False
            bounced_own_table = False
            ball_was_hit_back = False
            prev_bx = None
            prev_bz = None
            min_gap = 999.0
            steps_in_trial = 0

        # Detect ball hit: ball starts moving away from robot (was reflected by paddle)
        if not ball_was_hit_back and prev_bx is not None and (bx - prev_bx) * robot_side < -0.01 and bx * robot_side > 0.5:
            ball_was_hit_back = True

        # Detect own-table bounce AFTER hit: ball z drops to table height on robot side
        if ball_was_hit_back and not bounced_own_table and not cleared_this_trial:
            if bx > OWN_TABLE_X_MIN and bx < OWN_TABLE_X_MAX and bz < TABLE_Z and prev_bz is not None and prev_bz > TABLE_Z:
                bounced_own_table = True

        if prev_bx is not None and (prev_bx - NET_X) * robot_side > 0 and (bx - NET_X) * robot_side <= 0 and not cleared_this_trial:
            f = abs(prev_bx - NET_X) / abs(prev_bx - bx) if prev_bx != bx else 0.0
            z_at_net = prev_bz + f * (bz - prev_bz)
            if z_at_net > NET_Z:
                cleared_this_trial = True

        prev_bx, prev_bz = bx, bz
        prev_pos = ball_pos_local.copy()

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    # Save torque log
    torque_out = os.path.join(os.path.abspath(args_cli.output_dir), "torques.txt")
    os.makedirs(os.path.dirname(torque_out), exist_ok=True)
    with open(torque_out, "w") as f:
        f.write(f"{'step':>5} {'yb1':>8} {'yb2':>8} {'yb3':>8} {'yb4':>8} {'yb5':>8} {'yb6':>8} {'yb7':>8}\n")
        for step, tau in torque_log:
            f.write(f"{step:5d} {tau[0]:+8.3f} {tau[1]:+8.3f} {tau[2]:+8.3f} {tau[3]:+8.3f} {tau[4]:+8.3f} {tau[5]:+8.3f} {tau[6]:+8.3f}\n")
    print(f"\n[INFO] Torque log saved to: {torque_out} ({len(torque_log)} steps)")

    # Save paddle trajectory log
    paddle_out = os.path.join(os.path.abspath(args_cli.output_dir), "paddle_traj.txt")
    with open(paddle_out, "w") as f:
        f.write(f"{'step':>5} {'px':>8} {'py':>8} {'pz':>8} "
                f"{'Xx':>7} {'Xy':>7} {'Xz':>7} "
                f"{'Yx':>7} {'Yy':>7} {'Yz':>7} "
                f"{'Zx':>7} {'Zy':>7} {'Zz':>7}\n")
        for step, px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz in paddle_log:
            f.write(f"{step:5d} {px:+8.3f} {py:+8.3f} {pz:+8.3f} "
                    f"{ax:+7.3f} {ay:+7.3f} {az:+7.3f} "
                    f"{bx:+7.3f} {by:+7.3f} {bz:+7.3f} "
                    f"{cx:+7.3f} {cy:+7.3f} {cz:+7.3f}\n")
    print(f"[INFO] Paddle trajectory saved to: {paddle_out} ({len(paddle_log)} steps)")

    # Save actual joint angles (act) + commanded targets (tgt), keyed by step
    joints_out = os.path.join(os.path.abspath(args_cli.output_dir), "joints.txt")
    with open(joints_out, "w") as f:
        f.write(f"{'step':>5} "
                f"{'a1':>7} {'a2':>7} {'a3':>7} {'a4':>7} {'a5':>7} {'a6':>7} {'a7':>7} "
                f"{'t1':>7} {'t2':>7} {'t3':>7} {'t4':>7} {'t5':>7} {'t6':>7} {'t7':>7}\n")
        for step, a, t in joints_log:
            f.write(f"{step:5d} "
                    f"{a[0]:+7.3f} {a[1]:+7.3f} {a[2]:+7.3f} {a[3]:+7.3f} {a[4]:+7.3f} {a[5]:+7.3f} {a[6]:+7.3f} "
                    f"{t[0]:+7.3f} {t[1]:+7.3f} {t[2]:+7.3f} {t[3]:+7.3f} {t[4]:+7.3f} {t[5]:+7.3f} {t[6]:+7.3f}\n")
    print(f"[INFO] Joint-angle log saved to: {joints_out} ({len(joints_log)} steps)")

    # Plot actual-vs-target joint position at the contact (closest-approach) step.
    # x-axis = 7 joints, y-axis = position (rad); reveals the actuator tracking gap
    # (large where torque saturates during the swing).
    if min_gap_joints is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            act_c, tgt_c = min_gap_joints
            xs = np.arange(7)
            w = 0.38
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(xs - w / 2, tgt_c, w, label="target (reference)", color="#4C78A8")
            ax.bar(xs + w / 2, act_c, w, label="actual (sim)", color="#F58518")
            for i in range(7):
                ax.annotate(f"{act_c[i]-tgt_c[i]:+.2f}", (xs[i], max(act_c[i], tgt_c[i])),
                            ha="center", va="bottom", fontsize=8, color="#888")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"yb{i+1}" for i in range(7)])
            ax.set_xlabel("joint")
            ax.set_ylabel("position (rad)")
            ax.set_title(f"Contact-step joint tracking (step {min_gap_step}, gap {min_gap:.3f} m)\n"
                         "gap label = actual - target")
            ax.axhline(0, color="k", lw=0.5)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            plot_out = os.path.join(os.path.abspath(args_cli.output_dir), "joint_tracking.png")
            fig.savefig(plot_out, dpi=120)
            plt.close(fig)
            print(f"[INFO] Joint tracking plot saved to: {plot_out}")
        except Exception as e:
            print(f"[WARN] Could not save joint tracking plot: {e}")

    # Save ball trajectory (same step key as paddle/joints) -> read TRUE contact frame
    ball_out = os.path.join(os.path.abspath(args_cli.output_dir), "ball_traj.txt")
    with open(ball_out, "w") as f:
        f.write(f"{'step':>5} {'bx':>8} {'by':>8} {'bz':>8}\n")
        for step, bx, by, bz in ball_log:
            f.write(f"{step:5d} {bx:+8.3f} {by:+8.3f} {bz:+8.3f}\n")
    print(f"[INFO] Ball trajectory saved to: {ball_out} ({len(ball_log)} steps)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
