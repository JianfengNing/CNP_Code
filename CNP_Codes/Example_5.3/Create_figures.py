import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


penalty_para = 10 ** 4
#filename = f"data/penalty_state_bound_{penalty_para}.pth"
filename = f"data/CNP_state_bound.pth"
loaded_data = torch.load(filename)
U_pred = loaded_data['U_pred'].numpy()
Y_pred = loaded_data['Y_pred'].numpy()


n_plot = 256
x_plot = torch.linspace(0, 1, n_plot)
y_plot = torch.linspace(0, 1, n_plot)
X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
X = X.numpy()
Y = Y.numpy()


Y_c, U_c = np.load('data/classical_solu.npy', allow_pickle=True)


error_Y = (np.mean((Y_pred-Y_c)**2)/np.mean(Y_c**2))**0.5
error_U = (np.mean((U_pred-U_c)**2)/np.mean(U_c**2))**0.5

print('error_Y',error_Y,'error_U',error_U)



fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Y_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax1.set_title('(a) Optimal State')
ax1.view_init(elev=30, azim=250)
ax1.grid(True, alpha=0.3)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(b) Optimal Control')
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)


plt.tight_layout()
plt.savefig("data/CNP_state_bound.png", dpi=150, bbox_inches='tight')
plt.close()



fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Y_c, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax1.set_title('(a) Optimal State')
ax1.view_init(elev=30, azim=250)
ax1.grid(True, alpha=0.3)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_c, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(b) Optimal Control')
ax2.view_init(elev=30, azim=250)
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)


plt.tight_layout()
plt.savefig("data/classical_state_bound.png", dpi=150, bbox_inches='tight')
plt.close()