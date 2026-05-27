from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TableTennisPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        noise_std_type="log",  # 防止 std 学到负值 (默认 "scalar" 无正值约束, 训中可推过零 → normal expects std>=0)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Fix L: 0.0 → 0.005, 解 action_std 塌至 0.02 → 探索完全死的死结. 之前所有 reward 调整 (J/K) 因 std 太小学不进去, 先恢复探索再谈 shaping.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
