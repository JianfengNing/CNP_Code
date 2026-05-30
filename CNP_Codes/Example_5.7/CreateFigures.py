import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

n_plot = 256
filename = f"data/hybrid_inverse_problem.pth"
loaded_data = torch.load(filename)
C_exact = loaded_data['C_exa'].numpy().reshape(n_plot, n_plot)
C_pred = loaded_data['C_pred'].numpy().reshape(n_plot, n_plot)
F_exact = loaded_data['F_exa'].numpy().reshape(n_plot, n_plot)
F_pred = loaded_data['F_pred'].numpy().reshape(n_plot, n_plot)
X_1 = loaded_data['x_1'].cpu().numpy()
X_2 = loaded_data['x_2'].cpu().numpy()

error_C = (np.mean((C_pred-C_exact)**2)/np.mean(C_exact**2))**0.5
error_F = (np.mean((F_pred-F_exact)**2)/np.mean(F_exact**2))**0.5



print(f'Error_sigma: {error_C:.6f}, Error_F: {error_F:.6f}, ')
fig = plt.figure(figsize=(22, 5))

ax = fig.add_subplot(1,4, 1, projection='3d')
surf1 = ax.plot_surface(X_1, X_2, C_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax.set_title('(a) Exact $\sigma$', pad=0)
ax.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax.grid(True, alpha=0.3)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # 第二个视角：强调细节特征
ax = fig.add_subplot(1,4, 2, projection='3d')
surf2 = ax.plot_surface(X_1, X_2, C_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax.set_title('(b) Numerical  $\sigma$\nRelative L2 Error: $1.01x10^{{-2}}$',pad=0)
ax.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax.grid(True, alpha=0.3)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])


ax = fig.add_subplot(1,4, 3, projection='3d')
surf1 = ax.plot_surface(X_1, X_2, F_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax.set_title('(c) Exact $f$', pad=0)
ax.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax.grid(True, alpha=0.3)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

ax = fig.add_subplot(1,4, 4, projection='3d')
surf2 = ax.plot_surface(X_1, X_2, F_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax.set_title('(d) Numerical  $f$\nRelative L2 Error: $2.10x10^{{-2}}$',pad=0)
ax.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])




plt.tight_layout()
plt.savefig("data/hybrid_inv_prob.png", dpi=150, bbox_inches='tight')
plt.close()