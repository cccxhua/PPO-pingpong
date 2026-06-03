"""可视化 A1 部署日志：7个关节的实际位置 vs 目标位置。

功能：
  - 球出现/消失时间点标注（红色竖线）
  - 鼠标交互：悬停显示各关节当前值（10ms对齐），虚线标注数值
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    "worobots/a1_robot_ppo/logs/a1_deploy_20260603_143800.csv"

data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
t = data["time_s"].astype(float)

jp_cols = [f"jp_{i+1}" for i in range(7)]
tgt_cols = [f"tgt_{i+1}" for i in range(7)]
jp = np.column_stack([data[c].astype(float) for c in jp_cols])
tgt = np.column_stack([data[c].astype(float) for c in tgt_cols])

state = data["state"].astype(int)

# --- Detect ball appear (state 0->1) and swing done (state 2->3) ---
appear_times = []
done_times = []
for i in range(1, len(state)):
    if state[i] == 1 and state[i - 1] == 0:
        appear_times.append(t[i])
    if state[i] == 3 and state[i - 1] == 2:
        done_times.append(t[i])

# --- Plot ---
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
fig.suptitle("A1 Deploy: Joint Position vs Target", fontsize=14)

lines_actual = []
lines_target = []
for i in range(7):
    ax = axes[i]
    la, = ax.plot(t, jp[:, i], label="actual", linewidth=1)
    lt, = ax.plot(t, tgt[:, i], label="target", linewidth=1, linestyle="--")
    lines_actual.append(la)
    lines_target.append(lt)
    ax.set_ylabel(f"J{i+1} (rad)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    for ta in appear_times:
        ax.axvline(ta, color="red", linewidth=1.2, alpha=0.7, linestyle="-")
    for td in done_times:
        ax.axvline(td, color="blue", linewidth=1.2, alpha=0.7, linestyle="-")

axes[-1].set_xlabel("Time (s)")

# Legend for ball events
if appear_times or done_times:
    axes[0].axvline(np.nan, color="red", linewidth=1.2, label="ball appear")
    axes[0].axvline(np.nan, color="blue", linewidth=1.2, label="swing done")
    axes[0].legend(loc="upper right", fontsize=8)

# --- Interactive cursor (10ms snap) ---
SNAP_MS = 10
dt_snap = SNAP_MS / 1000.0

vlines = []
hlines = []
annotations = []
time_annotations = []
for ax in axes:
    vl = ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", visible=False)
    hl = ax.axhline(0, color="green", linewidth=0.8, linestyle=":", visible=False)
    ann = ax.annotate("", xy=(0, 0), fontsize=8, color="green",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                      visible=False)
    vlines.append(vl)
    hlines.append(hl)
    annotations.append(ann)

# Time label on the top subplot
time_label = axes[0].annotate("", xy=(0, 1), xycoords=("data", "axes fraction"),
                              fontsize=9, color="gray", ha="center", va="bottom",
                              bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9),
                              visible=False)


def on_mouse_move(event):
    if event.inaxes is None:
        for vl, hl, ann in zip(vlines, hlines, annotations):
            vl.set_visible(False)
            hl.set_visible(False)
            ann.set_visible(False)
        time_label.set_visible(False)
        fig.canvas.draw_idle()
        return

    x_raw = event.xdata
    x_snap = round(x_raw / dt_snap) * dt_snap
    idx = np.searchsorted(t, x_snap)
    idx = np.clip(idx, 0, len(t) - 1)

    # Update time label
    time_label.xy = (x_snap, 1)
    time_label.set_text(f"t = {x_snap:.3f} s")
    time_label.set_visible(True)

    for i, ax in enumerate(axes):
        vlines[i].set_xdata([x_snap])
        vlines[i].set_visible(True)

        val = jp[idx, i]
        hlines[i].set_ydata([val])
        hlines[i].set_visible(True)

        annotations[i].set_position((x_snap, val))
        annotations[i].xy = (x_snap, val)
        annotations[i].set_text(f"{val:.3f}")
        annotations[i].set_visible(True)

    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

plt.tight_layout()
plt.savefig(csv_path.replace(".csv", "_joints.png"), dpi=150)
print(f"Saved: {csv_path.replace('.csv', '_joints.png')}")
print(f"Ball appear (state 0->1): {len(appear_times)} events")
for i, ta in enumerate(appear_times):
    print(f"  [{i+1}] t = {ta:.3f}s")
print(f"Swing done (state 2->3): {len(done_times)} events")
for i, td in enumerate(done_times):
    print(f"  [{i+1}] t = {td:.3f}s")
plt.show()
