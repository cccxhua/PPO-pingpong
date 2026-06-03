import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-TableTennis",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.table_tennis.agents.rsl_rl_ppo_cfg:TableTennisPPORunnerCfg",
    },
)
