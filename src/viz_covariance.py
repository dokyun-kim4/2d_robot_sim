import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
with open("./output/kalman_filter.pkl", "rb") as f:
    data = pickle.load(f)

states = np.array([d[0] for d in data])  # (T, 3)
P_list = np.array([d[1] for d in data])  # (T, 3, 3)
T = len(data)
timesteps = np.arange(T)

# Extract diagonal variances and off-diagonal covariances
var_x = P_list[:, 0, 0]
var_y = P_list[:, 1, 1]
var_theta = P_list[:, 2, 2]
cov_xy = P_list[:, 0, 1]
cov_xth = P_list[:, 0, 2]
cov_yth = P_list[:, 1, 2]

# Frobenius norm of P over time (overall "uncertainty magnitude")
frob_norm = np.linalg.norm(P_list, ord="fro", axis=(1, 2))

fig = plt.figure(figsize=(16, 12))
fig.suptitle("EKF Covariance Matrix P — Over Time", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1: Diagonal variances ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(timesteps, var_x, label="P[x,x]", color="royalblue")
ax1.plot(timesteps, var_y, label="P[y,y]", color="darkorange")
ax1.plot(timesteps, var_theta, label="P[θ,θ]", color="green")
ax1.set_title("Diagonal Variances (should stay bounded)")
ax1.set_xlabel("Timestep")
ax1.set_ylabel("Variance")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight "unreasonably large" variance regions
THRESHOLD_POS = 5.0  # meters^2 — adjust as needed
THRESHOLD_THETA = 0.5  # rad^2    — adjust as needed
for ax, var, thr, label in [
    (ax1, var_x, THRESHOLD_POS, "x threshold"),
    (ax1, var_y, THRESHOLD_POS, "y threshold"),
    (ax1, var_theta, THRESHOLD_THETA, "θ threshold"),
]:
    ax.axhline(thr, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label=label)
ax1.legend(fontsize=8)

# --- Plot 2: Off-diagonal covariances ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(timesteps, cov_xy, label="P[x,y]", color="purple")
ax2.plot(timesteps, cov_xth, label="P[x,θ]", color="brown")
ax2.plot(timesteps, cov_yth, label="P[y,θ]", color="teal")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_title("Off-Diagonal Covariances")
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Covariance")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Frobenius norm ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(timesteps, frob_norm, color="crimson")
ax3.set_title("Frobenius Norm of P (overall uncertainty)")
ax3.set_xlabel("Timestep")
ax3.set_ylabel("||P||_F")
ax3.grid(True, alpha=0.3)

plt.savefig("./output/kalman_filter_covariance.png", dpi=150, bbox_inches="tight")
plt.show()