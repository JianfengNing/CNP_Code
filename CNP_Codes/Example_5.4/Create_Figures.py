import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")



filename = f"data/Direct_MPEC.pth"
n_plot = 256
loaded_data = torch.load(filename)
X_1 = loaded_data['x_1'].cpu().numpy().reshape(n_plot,n_plot)
X_2 = loaded_data['x_2'].cpu().numpy().reshape(n_plot,n_plot)
U_exact = loaded_data['U_exa'].numpy().reshape(n_plot,n_plot)
U_pred = loaded_data['U_pred'].numpy().reshape(n_plot,n_plot)
Y_exact = loaded_data['Y_exa'].numpy().reshape(n_plot,n_plot)
Y_pred = loaded_data['Y_pred'].numpy().reshape(n_plot,n_plot)
Xi_exact = loaded_data['Xi_exa'].numpy().reshape(n_plot,n_plot)
Xi_pred = loaded_data['Xi_pred'].numpy().reshape(n_plot,n_plot)

UU = np.concatenate([U_pred, U_exact])
u_max = np.max(UU)
u_min = np.min(UU)

YY = np.concatenate([Y_pred, Y_exact])
y_max = np.max(YY)
y_min = np.min(YY)

Xx = np.concatenate([Xi_exact, Xi_pred])

xi_max = np.max(Xx)
xi_min = np.min(Xx)

error_u = (np.mean((U_pred-U_exact)**2)/np.mean(U_exact**2))**0.5
error_y = (np.mean((Y_pred-Y_exact)**2)/np.mean(Y_exact**2))**0.5
error_xi = (np.mean((Xi_pred-Xi_exact)**2)/np.mean(Xi_exact**2))**0.5

z_min= np.min(np.concatenate((U_pred,U_exact)))
z_max= np.max(np.concatenate((U_pred,U_exact)))


print(f'Error_u: {error_u:.6f}, '
    f'Error_y: {error_y:.6f}, 'f'Error_xi: {error_xi:.6f}, ')
fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(2,3, 1, projection='3d')
surf1 = ax1.plot_surface(X_1, X_2, Y_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax1.set_title('(a) Exact State',pad=0)
ax1.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax1.grid(True, alpha=0.3)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

    # 第二个视角：强调细节特征
ax2 = fig.add_subplot(2,3, 2, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, Xi_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(b) Exact multiplier',pad=0)
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2 = fig.add_subplot(2, 3, 3, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, U_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(c) Exact Control', pad=0)
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(z_min, z_max)

ax2 = fig.add_subplot(2, 3, 4, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, Y_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(d) Numerical State\nRelative L2 Error: $1.00x10^{{-3}}$', pad=0)
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2 = fig.add_subplot(2, 3, 5, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, Xi_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(e) Numerical multiplier\nRelative L2 Error: $2.07x10^{{-2}}$', pad=0)
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2 = fig.add_subplot(2, 3, 6, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, U_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title('(f) Numerical Control\nRelative L2 Error: $1.05x10^{{-2}}$', pad=0)
ax2.view_init(elev=30, azim=250)  # 较低视角突出高度变化
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(z_min, z_max)





plt.tight_layout()
plt.savefig("data/reduced_MPEC.png", dpi=150, bbox_inches='tight')
plt.close()