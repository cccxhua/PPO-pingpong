import numpy as np
import matplotlib.pyplot as plt

# ---------------- 自适应卡尔曼滤波器 ----------------
class AdaptiveKalmanFilter:
    def __init__(self, process_variance=1e-4, base_measurement_variance=0.02**2,
                 beta=0.005, delta_ref=0.03, delta_t=0.0333,
                 trust_threshold=0.01, min_R_factor=1e-3, eps=1e-12):
        """
        自适应卡尔曼滤波（归一化差值），并在 |delta| < trust_threshold 时高度信任测量。
        """
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.Q = process_variance * np.eye(2)
        self.base = base_measurement_variance
        self.beta = beta
        self.delta_ref = delta_ref
        self.A = np.array([[1, delta_t],
                           [0, 1]])
        self.H = np.array([[1, 0]])
        self.prev_measurement = 0.0
        self.R = self.base
        self.trust_threshold = trust_threshold
        self.min_R_factor = min_R_factor
        self.eps = eps

    def initialize(self, z0):
        self.x[0] = z0
        self.P = np.eye(2) * 0.1
        self.prev_measurement = float(z0)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = float(z)
        delta = z - self.prev_measurement
        abs_delta = abs(delta)

        # 自适应 R
        if abs_delta < self.trust_threshold:
            adaptive_variance = max(self.base * self.min_R_factor, self.eps)
        else:
            adaptive_variance = self.base + self.beta * (delta / self.delta_ref) ** 2 + self.eps

        self.R = adaptive_variance

        S = (self.H @ self.P @ self.H.T)[0,0] + self.R
        K = (self.P @ self.H.T) / S
        y = z - (self.H @ self.x)[0]
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(2) - K @ self.H) @ self.P

        self.prev_measurement = z
        return self.x.copy(), self.R

# ---------------- 模拟数据 ----------------
np.random.seed(0)
steps = 200
dt = 0.0333
t = np.arange(steps) * dt
true_angle = 0.5 * np.sin(2 * np.pi * 0.5 * t)
true_angle[100:] += 0.2  # 突变
measurements = true_angle + np.random.normal(0, 0.02, size=steps)

# ---------------- 试验多个 beta ----------------
betas = [0.001, 0.005, 0.01]
delta_ref = 0.03
trust_threshold = 0.01
min_R_factor = 1e-3
base_var = 0.02**2

results = {}
for beta in betas:
    akf = AdaptiveKalmanFilter(
        base_measurement_variance=base_var,
        beta=beta,
        delta_ref=delta_ref,
        trust_threshold=trust_threshold,
        min_R_factor=min_R_factor
    )
    akf.initialize(measurements[0])
    est = []
    Rhist = []
    for z in measurements:
        akf.predict()
        x, R_t = akf.update(z)
        est.append(x[0])
        Rhist.append(R_t)
    results[beta] = {"est": np.array(est), "R": np.array(Rhist)}
    mse = np.mean((results[beta]["est"] - true_angle) ** 2)
    print(f"beta={beta}: MSE = {mse:.6e}")

# ---------------- 绘图（彩色热力图式标记 trust_threshold 触发） ----------------
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                        gridspec_kw={'height_ratios':[3,1]})

# 上图：角度 + 背景热力图
axs[0].plot(t, true_angle, 'k-', label="True angle")
axs[0].plot(t, measurements, 'r.', alpha=0.4, markersize=4, label="Measurements")

for beta in betas:
    est = results[beta]["est"]
    axs[0].plot(t, est, label=f"Adaptive KF (β={beta})")

    # 热力图区域：R 接近最小值时，表示只相信观测
    R = results[beta]["R"]
    normalized = (base_var * min_R_factor * 1.1 - R) / (base_var * min_R_factor * 1.1)
    normalized[normalized < 0] = 0  # 非触发区域置为0
    for i in range(len(t)-1):
        if normalized[i] > 0:
            axs[0].axvspan(t[i], t[i+1], color=f"C{betas.index(beta)}", alpha=0.2*normalized[i])

axs[0].set_ylabel("Angle [rad]")
axs[0].legend()
axs[0].grid(True)

# 下图：R(t)
for beta in betas:
    axs[1].plot(t, results[beta]["R"], label=f"R(t), β={beta}")
axs[1].axhline(base_var, color="C3", linestyle="--", label="base variance")
axs[1].set_ylabel("R (variance)")
axs[1].set_xlabel("Time [s]")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
