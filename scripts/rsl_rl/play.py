# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--residual_scale",
    type=float,
    default=None,
    help="Temporarily override actions.right_arm.residual_scale at play time (e.g. 0.05 to match an old checkpoint trained at scale=0.05).",
)
parser.add_argument(
    "--safe_serve",
    action="store_true",
    default=False,
    help="At play time, narrow the serve ranges to a friendlier center subset (drops edge/fast/dropping balls). Useful for demo videos where you want to showcase the policy's confident hits, not the OOD edges.",
)
parser.add_argument(
    "--fixed_serve",
    action="store_true",
    default=False,
    help="Lock every serve dimension to a single deterministic point (midpoint of the current preset). Combined with --num_episodes, produces a fully reproducible input sequence — meant for sim2sim (IsaacLab vs MuJoCo) replay verification.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=20,
    help="Number of episodes to run before exiting when --fixed_serve is set (defaults to 20).",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override RobotPlayEnvCfg.episode_length_s (default 1e9 = no timeout). When --fixed_serve is set this auto-defaults to 4.0 so every fixed-ball trial finishes promptly.",
)
parser.add_argument(
    "--record_obs",
    action="store_true",
    default=False,
    help="Record per-step policy observation vector (raw, before any normalization) to play_obs.npz. Used for sim2sim verification: load on the MuJoCo side and diff term-by-term to find where observations diverge.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Optional: override residual_scale at play time (e.g. play an old checkpoint
    # trained with scale=0.05 after the env file has moved on to 0.15).
    if args_cli.residual_scale is not None:
        right_arm_cfg = getattr(env_cfg.actions, "right_arm", None)
        if right_arm_cfg is None or not hasattr(right_arm_cfg, "residual_scale"):
            print("[WARN] --residual_scale ignored: env_cfg.actions.right_arm.residual_scale not found.")
        else:
            old = right_arm_cfg.residual_scale
            new = [args_cli.residual_scale] * 7
            right_arm_cfg.residual_scale = new
            print(f"[OVERRIDE] residual_scale: {old} -> {new}")

    # --safe_serve: keep the central / slower subset of the current serve preset.
    # Drops edge balls (large |y|, very low/high z), fast balls (large |vx|), and
    # downward serves (vz < 0). Same friendly subset is applied to both initial
    # reset_ball and the relaunch_ball event.
    if args_cli.safe_serve:
        SAFE_FRAC_Y  = 0.5   # keep central 50% of y range
        SAFE_FRAC_Z  = 0.4   # keep central 40% of z range (drops too-low / too-high)
        SAFE_FRAC_VY = 0.5
        SAFE_VX_KEEP = "slow"   # keep the slow half of vx (smaller |vx|)
        SAFE_VZ_MIN  = 0.0      # require non-negative vz (drop dropping balls)

        def _shrink_sym(rng, frac):
            lo, hi = float(rng[0]), float(rng[1])
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * frac
            return (mid - half, mid + half)

        def _vx_slow_half(rng):
            lo, hi = float(rng[0]), float(rng[1])
            # vx is negative when ROBOT_SIDE=-1 (ball flies toward robot at -X);
            # "slow" = closer to 0, i.e. larger value when both negative.
            mid = 0.5 * (lo + hi)
            if abs(hi) < abs(lo):     # hi closer to 0 → slow side is [mid, hi]
                return (mid, hi)
            else:
                return (lo, mid)

        def _vz_drop_negative(rng):
            lo, hi = float(rng[0]), float(rng[1])
            return (max(lo, SAFE_VZ_MIN), max(hi, SAFE_VZ_MIN))

        def _apply(params):
            new_params = dict(params)
            if "y_range" in params:  new_params["y_range"]  = _shrink_sym(params["y_range"], SAFE_FRAC_Y)
            if "z_range" in params:  new_params["z_range"]  = _shrink_sym(params["z_range"], SAFE_FRAC_Z)
            if "vy_range" in params: new_params["vy_range"] = _shrink_sym(params["vy_range"], SAFE_FRAC_VY)
            if "vx_range" in params: new_params["vx_range"] = _vx_slow_half(params["vx_range"])
            if "vz_range" in params: new_params["vz_range"] = _vz_drop_negative(params["vz_range"])
            return new_params

        for ev_name in ("reset_ball", "relaunch_ball"):
            ev = getattr(env_cfg.events, ev_name, None)
            if ev is None or ev.params is None:
                continue
            old_params = {k: ev.params.get(k) for k in ("x_range","y_range","z_range","vx_range","vy_range","vz_range")}
            new_params = _apply(ev.params)
            ev.params = new_params
            print(f"[SAFE_SERVE] {ev_name}:")
            for k in ("y_range","z_range","vx_range","vy_range","vz_range"):
                if k in old_params and old_params[k] is not None:
                    print(f"    {k:9s} {old_params[k]} -> {new_params[k]}")

    # --fixed_serve: collapse every serve range to its midpoint so every reset
    # produces the EXACT same ball. Combined with --num_episodes this gives a
    # deterministic, replayable input sequence for sim2sim (IsaacLab vs MuJoCo).
    if args_cli.fixed_serve:
        def _to_point(rng):
            lo, hi = float(rng[0]), float(rng[1])
            mid = 0.5 * (lo + hi)
            return (mid, mid)

        for ev_name in ("reset_ball", "relaunch_ball"):
            ev = getattr(env_cfg.events, ev_name, None)
            if ev is None or ev.params is None:
                continue
            old_params = {k: ev.params.get(k) for k in
                          ("x_range", "y_range", "z_range", "vx_range", "vy_range", "vz_range")}
            new_params = dict(ev.params)
            for k in old_params:
                if old_params[k] is not None:
                    new_params[k] = _to_point(old_params[k])
            ev.params = new_params
            print(f"[FIXED_SERVE] {ev_name}:")
            for k in ("x_range", "y_range", "z_range", "vx_range", "vy_range", "vz_range"):
                if k in old_params and old_params[k] is not None:
                    print(f"    {k:9s} {old_params[k]} -> {new_params[k]}")

        # Also kill any motion-side noise so command timing is deterministic too
        try:
            env_cfg.commands.motion.hit_phase_noise = 0.0
            env_cfg.commands.motion.ball_arrive_time_noise = 0.0
        except AttributeError:
            pass

        # Default to a finite episode length so each trial actually finishes.
        if args_cli.episode_length_s is None:
            args_cli.episode_length_s = 4.0

    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
        print(f"[OVERRIDE] episode_length_s -> {args_cli.episode_length_s}s")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        print("[INFO] Pretrained checkpoint not supported in this version.")
        return
    elif args_cli.checkpoint:
        # If --checkpoint is a bare filename (no path separators) and --load_run is set,
        # resolve relative to log_root/load_run. Otherwise treat as a direct path.
        ckpt_arg = args_cli.checkpoint
        is_bare_name = ("/" not in ckpt_arg) and (os.sep not in ckpt_arg)
        if is_bare_name and agent_cfg.load_run:
            candidate = os.path.join(log_root_path, agent_cfg.load_run, ckpt_arg)
            if os.path.exists(candidate):
                resume_path = candidate
            else:
                resume_path = retrieve_file_path(ckpt_arg)
        else:
            resume_path = retrieve_file_path(ckpt_arg)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # joint position logging
    joint_log = []
    target_log = []         # PD target joint position (ref + residual * scale)
    torque_log = []         # applied_torque per step (post effort_limit clip)
    computed_torque_log = []  # PD-controller-requested torque (pre clip)
    reset_steps = []        # frame indices where env.step triggered a reset (joint pos jumps)
    self_dist_log = []      # min distance between distal arm links and body links each frame
    arm_inter_dist_log = [] # min distance between non-adjacent arm links (yb_i vs yb_j, |i-j|>=2)
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    print(f"[INFO] Robot joint names ({len(joint_names)}): {joint_names}")

    # Pick distal-arm vs body link indices for self-distance check.
    # Distal arm = yb_4..yb_7 + paddle (avoid false positives at yb_1..3 hinge-adjacent to lift column).
    body_names = robot.body_names
    distal_keys = ["yb_4", "yb_5", "yb_6", "yb_7", "paddle"]
    arm_distal_idx = [i for i, n in enumerate(body_names) if any(k in n for k in distal_keys)]
    body_idx = [i for i, n in enumerate(body_names) if "yb_" not in n and i not in arm_distal_idx]
    SELF_DIST_THRESHOLD = 0.10   # m; distal arm vs body
    if arm_distal_idx and body_idx:
        print(f"[INFO] Self-collision check (arm vs body): distal arm {len(arm_distal_idx)} × body {len(body_idx)} "
              f"(threshold {SELF_DIST_THRESHOLD*100:.0f} cm)")
    else:
        print("[WARN] Self-collision check (arm vs body) disabled: could not partition links")

    # Inter-arm self-collision: pairs of arm links with chain distance >= 2 (skip
    # kinematic neighbors which are always close at the joint hinge).
    # Build chain order: yb_1 ... yb_7, paddle (treated as 8).
    arm_chain = []   # list of (chain_idx, body_idx, name)
    for ci in range(1, 8):
        for k, n in enumerate(body_names):
            if f"yb_{ci}" in n:
                arm_chain.append((ci, k, n))
                break
    for k, n in enumerate(body_names):
        if "paddle" in n.lower():
            arm_chain.append((8, k, n))
            break
    arm_inter_pairs = [
        (a[1], b[1], a[2], b[2])
        for a in arm_chain for b in arm_chain
        if b[0] - a[0] >= 2
    ]
    ARM_INTER_THRESHOLD = 0.05   # m; non-adjacent arm links should keep > 5 cm apart
    if arm_inter_pairs:
        print(f"[INFO] Self-collision check (inter-arm): {len(arm_inter_pairs)} non-adjacent pairs "
              f"(threshold {ARM_INTER_THRESHOLD*100:.0f} cm)")
    else:
        print("[WARN] Self-collision check (inter-arm) disabled: arm chain not found")

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    episodes_done = 0

    # --record_obs: optional per-step observation logging for sim2sim verification.
    obs_log = []
    obs_term_names = None
    obs_term_dims = None
    obs_term_offsets = None
    if args_cli.record_obs:
        try:
            om = env.unwrapped.observation_manager
            obs_term_names = list(om.active_terms.get("policy", []))
            raw_dims = list(om.group_obs_dim.get("policy", []))
            # group_obs_dim entries can be tuples; flatten to scalar dim per term
            obs_term_dims = [int(np.prod(d)) if hasattr(d, "__len__") else int(d) for d in raw_dims]
            obs_term_offsets = [0]
            for d in obs_term_dims:
                obs_term_offsets.append(obs_term_offsets[-1] + d)
            print(f"[INFO] Recording obs: {len(obs_term_names)} terms, total dim {obs_term_offsets[-1]}")
            for n, d, o in zip(obs_term_names, obs_term_dims, obs_term_offsets[:-1]):
                print(f"    [{o:3d}:{o+d:3d}]  {n}  (dim={d})")
        except Exception as e:
            print(f"[WARN] --record_obs cannot read observation_manager schema: {e}")

    if args_cli.fixed_serve:
        print(f"[INFO] Fixed-serve mode: will exit after {args_cli.num_episodes} episodes "
              f"(episode_length_s = {env_cfg.episode_length_s}s)")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # log raw policy obs *before* feeding it to policy() — this captures
            # exactly the input the policy saw, which is what MuJoCo must replicate.
            # RslRlVecEnvWrapper returns a TensorDict with "policy" / "critic" groups;
            # extract the policy group tensor explicitly.
            if args_cli.record_obs:
                obs_t = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
                obs_log.append(obs_t[0].detach().cpu().numpy().copy())
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # log joint positions (right arm)
            joint_pos = robot.data.joint_pos[0].cpu().numpy().copy()
            joint_log.append(joint_pos)

            # log PD target joint position (what the PD controller is trying to reach)
            target_log.append(robot.data.joint_pos_target[0].cpu().numpy().copy())

            # log applied torques (post effort_limit clip; same shape as joint_pos)
            torque_log.append(robot.data.applied_torque[0].cpu().numpy().copy())
            # log computed torques (what PD wanted, pre-clip — diverges from applied at saturation)
            computed_torque_log.append(robot.data.computed_torque[0].cpu().numpy().copy())

            # mark reset frame: when dones is True, the returned joint_pos is already
            # the new-episode initial pose, so pos[k]-pos[k-1] is a teleport — flag it.
            done_val = False
            try:
                done_val = bool(dones[0].item()) if hasattr(dones, "__getitem__") else bool(dones)
            except Exception:
                done_val = False
            if done_val:
                reset_steps.append(len(joint_log) - 1)
                episodes_done += 1
                if args_cli.fixed_serve and episodes_done >= args_cli.num_episodes:
                    print(f"[INFO] Fixed-serve: completed {episodes_done} episodes — exiting play loop.")
                    break

            # self-collision distance: min over (distal arm × body) link pairs
            if arm_distal_idx and body_idx:
                body_pos_w = robot.data.body_pos_w[0].cpu().numpy()    # (N_links, 3)
                arm_pos = body_pos_w[arm_distal_idx]                    # (n_arm, 3)
                trunk_pos = body_pos_w[body_idx]                        # (n_body, 3)
                diffs = arm_pos[:, None, :] - trunk_pos[None, :, :]    # (n_arm, n_body, 3)
                dists = np.linalg.norm(diffs, axis=2)
                self_dist_log.append(float(dists.min()))

            # inter-arm self-collision: min over non-adjacent (yb_i, yb_j, |i-j|>=2) pairs
            if arm_inter_pairs:
                if not (arm_distal_idx and body_idx):
                    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
                ds = [float(np.linalg.norm(body_pos_w[a] - body_pos_w[b]))
                      for a, b, _, _ in arm_inter_pairs]
                arm_inter_dist_log.append(min(ds))

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save joint log
    if joint_log:
        joint_log_arr = np.array(joint_log)
        save_path = os.path.join(log_dir, "play_joint_pos.npz")
        save_dict = {"joint_pos": joint_log_arr, "joint_names": joint_names}
        if target_log:
            save_dict["joint_pos_target"] = np.array(target_log)
        if torque_log:
            save_dict["applied_torque"] = np.array(torque_log)
        if computed_torque_log:
            save_dict["computed_torque"] = np.array(computed_torque_log)
        np.savez(save_path, **save_dict)
        print(f"[INFO] Saved joint positions: {save_path}  shape={joint_log_arr.shape}")
        print(f"[INFO] Joint names: {joint_names}")

        # also save as human-readable txt
        txt_path = os.path.join(log_dir, "play_joint_pos.txt")
        with open(txt_path, "w") as f:
            f.write("# Joint position log\n")
            f.write(f"# Columns: {joint_names}\n")
            f.write(f"# Shape: {joint_log_arr.shape} (steps x joints)\n")
            f.write(f"# dt = {dt:.4f}s\n\n")
            header = "step\t" + "\t".join(joint_names)
            f.write(header + "\n")
            for i, pos in enumerate(joint_log_arr):
                line = f"{i}\t" + "\t".join(f"{v:.4f}" for v in pos)
                f.write(line + "\n")
        print(f"[INFO] Saved joint positions (txt): {txt_path}")

        # save policy obs log if requested
        if args_cli.record_obs and obs_log:
            obs_arr = np.asarray(obs_log, dtype=np.float32)
            obs_npz = os.path.join(log_dir, "play_obs.npz")
            obs_save = {
                "obs": obs_arr,                      # (N, obs_dim) — pre-policy raw obs
                "dt": np.float32(dt),
            }
            if obs_term_names is not None:
                obs_save["term_names"] = np.array(obs_term_names, dtype=object)
                obs_save["term_dims"] = np.array(obs_term_dims, dtype=np.int32)
                obs_save["term_offsets"] = np.array(obs_term_offsets, dtype=np.int32)
            if reset_steps:
                obs_save["reset_steps"] = np.array(reset_steps, dtype=np.int32)
            np.savez(obs_npz, **obs_save)
            print(f"[INFO] Saved policy obs: {obs_npz}  shape={obs_arr.shape}")

        # plot per-joint subplot figures for the right-arm (yb_*) joints if present:
        # position, velocity, acceleration. Only per-joint subplots, no overlay variant.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator, AutoMinorLocator

            yb_idx = [i for i, n in enumerate(joint_names) if n.startswith("joint_yb_")]
            if yb_idx:
                yb_names = [joint_names[i] for i in yb_idx]
                yb_pos = joint_log_arr[:, yb_idx]
                # backward finite differences (matches typical real-robot estimation:
                # current velocity uses pos[i] - pos[i-1]; first frame is padded)
                yb_vel = np.zeros_like(yb_pos)
                yb_vel[1:] = (yb_pos[1:] - yb_pos[:-1]) / dt
                yb_vel[0] = yb_vel[1]
                yb_acc = np.zeros_like(yb_vel)
                yb_acc[1:] = (yb_vel[1:] - yb_vel[:-1]) / dt
                yb_acc[0] = yb_acc[1]

                # break the differences across episode resets: pos teleports at the reset
                # frame would otherwise produce huge fake spikes that distort the y-axis.
                # vel[reset] is invalid (cross-episode diff); acc[reset] and acc[reset+1]
                # both depend on it, so NaN them out and let matplotlib auto-break the line.
                for r in reset_steps:
                    if 0 <= r < yb_vel.shape[0]:
                        yb_vel[r] = np.nan
                    if 0 <= r < yb_acc.shape[0]:
                        yb_acc[r] = np.nan
                    if 0 <= r + 1 < yb_acc.shape[0]:
                        yb_acc[r + 1] = np.nan

                t_axis = np.arange(joint_log_arr.shape[0]) * dt
                reset_t = [r * dt for r in reset_steps if 0 <= r < joint_log_arr.shape[0]]
                n = len(yb_idx)

                # velocity limits from articulation (per joint), fall back to None on failure
                try:
                    vel_lim = robot.data.joint_vel_limits[0].cpu().numpy()
                    yb_vel_lim = [float(vel_lim[joint_names.index(name)]) for name in yb_names]
                except Exception:
                    yb_vel_lim = [None] * n

                def _plot_panel(arr, ylabel, title, fname, hlines=None, mark_resets=True):
                    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n + 1), sharex=True)
                    if n == 1:
                        axes = [axes]
                    for i, ax in enumerate(axes):
                        ax.plot(t_axis, arr[:, i], lw=1.2, color=f"C{i % 10}")
                        ax.set_ylabel(f"{yb_names[i].replace('joint_', '')}\n({ylabel})", fontsize=9)
                        # denser ticks + minor grid so small data ranges are still readable
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
                        ax.yaxis.set_minor_locator(AutoMinorLocator())
                        ax.grid(True, which="major", alpha=0.30)
                        ax.grid(True, which="minor", alpha=0.12)
                        ax.tick_params(axis="y", labelsize=8)

                        # tight asymmetric y-range strictly to data (NaN-safe).
                        # Don't force 0 to center — data like yb_1 sitting at +1.0~+1.5
                        # otherwise wastes half the y-axis on empty negative space.
                        finite = arr[:, i][np.isfinite(arr[:, i])]
                        if finite.size:
                            vmin = float(finite.min())
                            vmax = float(finite.max())
                            span = max(vmax - vmin, 1e-6)
                            pad = max(0.05, 0.15 * span)
                            ymin, ymax = vmin - pad, vmax + pad
                        else:
                            ymin, ymax = -1.0, 1.0
                        ax.set_ylim(ymin, ymax)
                        # only draw the y=0 reference if it's inside the visible range
                        if ymin <= 0.0 <= ymax:
                            ax.axhline(0, color="gray", lw=0.5, ls="--")

                        if hlines is not None and hlines[i] is not None:
                            lim = float(hlines[i])
                            drew = False
                            if lim <= ymax:
                                ax.axhline(lim, color="r", lw=0.6, ls=":")
                                drew = True
                            if -lim >= ymin:
                                ax.axhline(-lim, color="r", lw=0.6, ls=":")
                                drew = True
                            if not drew:
                                ax.text(0.99, 0.05, f"v_lim=±{lim:.0f} (off-axis)",
                                        transform=ax.transAxes, ha="right", va="bottom",
                                        fontsize=7, color="red", alpha=0.7)

                        # mark resets at the bottom of each axes (axes-fraction y) so
                        # markers stay visible regardless of where the data lives.
                        if mark_resets and reset_t:
                            trans = ax.get_xaxis_transform()  # x: data, y: axes 0..1
                            ax.scatter(reset_t, [0.04] * len(reset_t),
                                       marker="x", color="red", s=28, zorder=5,
                                       transform=trans, clip_on=False,
                                       label="reset" if i == 0 else None)
                    if reset_t and mark_resets:
                        axes[0].legend(loc="upper right", fontsize=8)
                    axes[-1].set_xlabel("time (s)")
                    fig.suptitle(title, fontsize=12)
                    plt.tight_layout()
                    out = os.path.join(log_dir, fname)
                    plt.savefig(out, dpi=110)
                    plt.close(fig)
                    print(f"[INFO] Saved plot: {out}")

                run = os.path.basename(log_dir)
                _plot_panel(yb_pos, "rad", f"right-arm joint positions — {run}",
                            "play_joint_pos.png")
                _plot_panel(yb_vel, "rad/s", f"right-arm joint velocities — {run}  (red dotted = velocity_limit)",
                            "play_joint_vel.png", hlines=yb_vel_lim)
                _plot_panel(yb_acc, "rad/s²", f"right-arm joint accelerations — {run}",
                            "play_joint_acc.png")

                # torque plot (applied_torque, after effort_limit clip)
                if torque_log:
                    yb_torque = np.array(torque_log)[:, yb_idx]
                    # effort limits per joint, fall back to None if accessor missing
                    try:
                        eff_lim = robot.data.joint_effort_limits[0].cpu().numpy()
                        yb_eff_lim = [float(eff_lim[joint_names.index(name)]) for name in yb_names]
                    except Exception:
                        yb_eff_lim = [None] * n
                    _plot_panel(yb_torque, "N·m",
                                f"right-arm joint torques — {run}  (red dotted = effort_limit)",
                                "play_joint_torque.png", hlines=yb_eff_lim)

                # joint-pos tracking: PD target vs actual (7 subplots, two lines each)
                if target_log:
                    yb_target = np.array(target_log)[:, yb_idx]
                    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n + 1), sharex=True)
                    if n == 1:
                        axes = [axes]
                    for i, ax in enumerate(axes):
                        ax.plot(t_axis, yb_target[:, i], lw=1.0, color="black",
                                ls="--", alpha=0.7, label="target" if i == 0 else None)
                        ax.plot(t_axis, yb_pos[:, i], lw=1.2, color=f"C{i % 10}",
                                label="actual" if i == 0 else None)
                        ax.set_ylabel(f"{yb_names[i].replace('joint_', '')}\n(rad)", fontsize=9)
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
                        ax.yaxis.set_minor_locator(AutoMinorLocator())
                        ax.grid(True, which="major", alpha=0.30)
                        ax.grid(True, which="minor", alpha=0.12)
                        ax.tick_params(axis="y", labelsize=8)
                        # tight ylim covering both lines
                        col = np.concatenate([yb_target[:, i], yb_pos[:, i]])
                        col = col[np.isfinite(col)]
                        if col.size:
                            vmin, vmax = float(col.min()), float(col.max())
                            span = max(vmax - vmin, 1e-6)
                            pad = max(0.05, 0.15 * span)
                            ax.set_ylim(vmin - pad, vmax + pad)
                        if reset_t:
                            trans = ax.get_xaxis_transform()
                            ax.scatter(reset_t, [0.04] * len(reset_t),
                                       marker="x", color="red", s=28, zorder=5,
                                       transform=trans, clip_on=False)
                    axes[0].legend(loc="upper right", fontsize=8)
                    axes[-1].set_xlabel("time (s)")
                    fig.suptitle(f"right-arm joint pos tracking (target vs actual) — {run}", fontsize=12)
                    plt.tight_layout()
                    out = os.path.join(log_dir, "play_joint_pos_track.png")
                    plt.savefig(out, dpi=110); plt.close(fig)
                    err = yb_target - yb_pos
                    rms = np.sqrt(np.nanmean(err ** 2, axis=0))
                    print(f"[INFO] Saved plot: {out}  per-joint RMS error (rad): "
                          f"{[round(float(x), 3) for x in rms]}")

                # torque tracking: computed (PD wanted) vs applied (after clip)
                if computed_torque_log and torque_log:
                    yb_comp = np.array(computed_torque_log)[:, yb_idx]
                    yb_appl = np.array(torque_log)[:, yb_idx]
                    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n + 1), sharex=True)
                    if n == 1:
                        axes = [axes]
                    for i, ax in enumerate(axes):
                        ax.plot(t_axis, yb_comp[:, i], lw=1.0, color="black",
                                ls="--", alpha=0.7, label="target (computed)" if i == 0 else None)
                        ax.plot(t_axis, yb_appl[:, i], lw=1.2, color=f"C{i % 10}",
                                label="actual (applied)" if i == 0 else None)
                        ax.set_ylabel(f"{yb_names[i].replace('joint_', '')}\n(N·m)", fontsize=9)
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
                        ax.yaxis.set_minor_locator(AutoMinorLocator())
                        ax.grid(True, which="major", alpha=0.30)
                        ax.grid(True, which="minor", alpha=0.12)
                        ax.tick_params(axis="y", labelsize=8)
                        ax.axhline(0, color="gray", lw=0.5, ls="--")
                        # ylim fits BOTH curves; effort_limit drawn only if it falls inside
                        cols = np.concatenate([yb_comp[:, i], yb_appl[:, i]])
                        cols = cols[np.isfinite(cols)]
                        if cols.size:
                            peak = float(np.max(np.abs(cols)))
                            ymax_i = peak + max(0.5, 0.15 * peak)
                            ax.set_ylim(-ymax_i, ymax_i)
                            if yb_eff_lim[i] is not None and yb_eff_lim[i] <= ymax_i:
                                ax.axhline(yb_eff_lim[i], color="r", lw=0.6, ls=":")
                                ax.axhline(-yb_eff_lim[i], color="r", lw=0.6, ls=":")
                        if reset_t:
                            trans = ax.get_xaxis_transform()
                            ax.scatter(reset_t, [0.04] * len(reset_t),
                                       marker="x", color="red", s=28, zorder=5,
                                       transform=trans, clip_on=False)
                    axes[0].legend(loc="upper right", fontsize=8)
                    axes[-1].set_xlabel("time (s)")
                    fig.suptitle(f"right-arm torque tracking (computed vs applied) — {run}", fontsize=12)
                    plt.tight_layout()
                    out = os.path.join(log_dir, "play_joint_torque_track.png")
                    plt.savefig(out, dpi=110); plt.close(fig)
                    # report saturation gap = how much PD wanted but got clipped
                    gap = np.abs(yb_comp) - np.abs(yb_appl)   # >0 when clip kicks in
                    sat_mask = gap > 0.5
                    pct = sat_mask.mean(axis=0) * 100.0
                    print(f"[INFO] Saved plot: {out}  per-joint clip% (computed > applied >0.5 N·m): "
                          f"{[round(float(x), 1) for x in pct]}")

            # self-collision distance plot (arm vs body)
            if self_dist_log:
                d = np.array(self_dist_log)
                t_d = np.arange(len(d)) * dt
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(t_d, d, lw=1.2, color="C0", label="min dist (distal arm vs body)")
                ax.axhline(SELF_DIST_THRESHOLD, color="red", lw=0.8, ls="--",
                           label=f"threshold {SELF_DIST_THRESHOLD*100:.0f} cm")
                violation = d < SELF_DIST_THRESHOLD
                if violation.any():
                    ax.fill_between(t_d, 0, d.max() * 1.05, where=violation,
                                    color="red", alpha=0.15, label="self-collision risk")
                if reset_t:
                    for rt in reset_t:
                        ax.axvline(rt, color="gray", lw=0.4, ls=":")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("min link distance (m)")
                ax.set_title(f"self-collision distance (arm vs body) — {os.path.basename(log_dir)}")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.3)
                ax.legend(loc="lower right", fontsize=9)
                plt.tight_layout()
                out = os.path.join(log_dir, "play_self_distance.png")
                plt.savefig(out, dpi=110)
                plt.close(fig)
                pct = 100.0 * float(violation.mean())
                print(f"[INFO] Saved plot: {out}  (frames below {SELF_DIST_THRESHOLD*100:.0f} cm: {pct:.1f}%, "
                      f"global min: {float(d.min())*100:.1f} cm)")

            # inter-arm self-collision plot (non-adjacent yb link pairs)
            if arm_inter_dist_log:
                d = np.array(arm_inter_dist_log)
                t_d = np.arange(len(d)) * dt
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(t_d, d, lw=1.2, color="C2", label="min dist (non-adjacent arm pairs)")
                ax.axhline(ARM_INTER_THRESHOLD, color="red", lw=0.8, ls="--",
                           label=f"threshold {ARM_INTER_THRESHOLD*100:.0f} cm")
                violation = d < ARM_INTER_THRESHOLD
                if violation.any():
                    ax.fill_between(t_d, 0, d.max() * 1.05, where=violation,
                                    color="red", alpha=0.15, label="inter-arm collision risk")
                if reset_t:
                    for rt in reset_t:
                        ax.axvline(rt, color="gray", lw=0.4, ls=":")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("min link distance (m)")
                ax.set_title(f"inter-arm self-collision distance — {os.path.basename(log_dir)}")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.3)
                ax.legend(loc="lower right", fontsize=9)
                plt.tight_layout()
                out = os.path.join(log_dir, "play_arm_inter_distance.png")
                plt.savefig(out, dpi=110)
                plt.close(fig)
                pct = 100.0 * float(violation.mean())
                # also report which pair held the global minimum
                kmin = int(np.argmin(d))
                # recompute that frame's per-pair distance to find culprit
                try:
                    body_pos_kmin = robot.data.body_pos_w[0].cpu().numpy()
                    pair_d = [(np.linalg.norm(body_pos_kmin[a] - body_pos_kmin[b]), an, bn)
                              for a, b, an, bn in arm_inter_pairs]
                    pair_d.sort()
                    closest_now = pair_d[0]
                    print(f"[INFO] Saved plot: {out}  (frames below {ARM_INTER_THRESHOLD*100:.0f} cm: {pct:.1f}%, "
                          f"global min: {float(d.min())*100:.1f} cm; closest pair NOW: "
                          f"{closest_now[1]} ↔ {closest_now[2]} = {closest_now[0]*100:.1f} cm)")
                except Exception:
                    print(f"[INFO] Saved plot: {out}  (frames below {ARM_INTER_THRESHOLD*100:.0f} cm: {pct:.1f}%, "
                          f"global min: {float(d.min())*100:.1f} cm)")
        except Exception as e:
            print(f"[WARN] Failed to render play_joint_*.png: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
