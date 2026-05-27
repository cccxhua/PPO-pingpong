import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-TableTennis-SAC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RobotSacEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:RobotSacPlayEnvCfg",
        "rl_games_cfg_entry_point": "unitree_rl_lab.tasks.table_tennis.agents:rl_games_sac_cfg.yaml",
    },
)
