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
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class A1TableTennisPPORunnerCfg(TableTennisPPORunnerCfg):
    """A1 乒乓球 (稳定加强版 v2 — 修复 std 自膨胀):
    - clip_actions=2.0:    防 actor inf/NaN 传到 env (但与 entropy_coef>0 互相作用会让 std 涨上天)
    - entropy_coef=0.0:    ★ 关键修复 — 关掉熵奖励, policy 不再有动机膨胀 std
    - init_noise_std=0.2:  起点 std 略低, 避免 resume 时立刻撞 clip_actions=2 边界
    - max_grad_norm=0.5:   梯度裁剪更严
    - num_learning_epochs=3: 每个 batch 少 update
    - clip_param=0.15:     PPO ratio clip 更紧
    - value_loss_coef=0.5: vf loss 不主导
    - learning_rate=1e-4:  更慢学习
    - desired_kl=0.008:    KL 更紧
    - save_interval=100:   更频繁 ckpt
    """

    experiment_name = "a1_tabletennis"
    save_interval = 100
    clip_actions = 2.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )
