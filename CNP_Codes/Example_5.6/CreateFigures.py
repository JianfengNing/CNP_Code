import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
plt.rcParams['axes.titlesize'] = 18

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


filename = f"data/ERNM_singularity.pth"

loaded_data = torch.load(filename)
u_pred = loaded_data['U_pred'].numpy()
y_pred = loaded_data['Y_pred'].numpy()
u_true = loaded_data['U_exact'].numpy()
y_true = loaded_data['Y_exact'].numpy()
x_np = loaded_data['X'].numpy()
y_np = loaded_data['Y'].numpy()

filename2 = f"data/ERNM_singularity_ori.pth"

loaded_data2 = torch.load(filename2)
u_pred2 = loaded_data2['U_pred'].numpy()
y_pred2 = loaded_data2['Y_pred'].numpy()


error_y = (np.mean((y_pred - y_true) ** 2)/np.mean(y_true ** 2)) ** 0.5
error_u = (np.mean((u_pred - u_true) ** 2) / np.mean(u_true ** 2))**0.5
print(f'Singularity enriched: error_y: {error_y:.6f}, error_u: {error_u:.6f}')

error_y2 = (np.mean((y_pred2 - y_true) ** 2)/np.mean(y_true ** 2)) ** 0.5
error_u2 = (np.mean((u_pred2 - u_true) ** 2) / np.mean(u_true ** 2))**0.5
print(f'Original method: error_y: {error_y2:.6f}, error_u: {error_u2:.6f}')


err_y1 = np.abs(y_true - y_pred)
err_y2 = np.abs(y_true - y_pred2)
err_u1 = np.abs(u_true - u_pred)
err_u2 = np.abs(u_true - u_pred2)

y_err_vmin = 0.0
y_err_vmax = max(err_y1.max(), err_y2.max())

u_err_vmin = 0.0
u_err_vmax = max(err_u1.max(), err_u2.max())


fig = plt.figure(figsize=(28, 10))


ax = fig.add_subplot(2, 5, 1)
sc = ax.scatter(x_np, y_np,c=y_true, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('Exact state')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 2)
sc = ax.scatter(x_np, y_np,c=y_pred, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('State: SEERNM')

ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 3)
sc = ax.scatter(x_np, y_np,c=y_pred2, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('State: ERNM ')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 4)
sc = ax.scatter(x_np, y_np,c=np.abs(y_true-y_pred), cmap='jet',s=0.5,vmin=y_err_vmin, vmax=y_err_vmax)
#sc = ax.scatter(x_np, y_np,c=np.abs(y_true-y_pred), cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('Absolute state error: SEERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 5)
sc = ax.scatter(x_np, y_np,c=np.abs(y_true-y_pred2), cmap='jet',s=0.5,vmin=y_err_vmin, vmax=y_err_vmax)
ax.set_aspect('equal')
ax.set_title('Absolute state error: ERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()





ax = fig.add_subplot(2, 5, 6)
sc = ax.scatter(x_np, y_np,c=u_true, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('Exact control')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 7)
sc = ax.scatter(x_np, y_np,c=u_pred, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('Control: SEERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 8)
sc = ax.scatter(x_np, y_np,c=u_pred2, cmap='jet',s=0.5)
ax.set_aspect('equal')
ax.set_title('Control: ERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()


ax = fig.add_subplot(2, 5, 9)
sc = ax.scatter(x_np, y_np,c=np.abs(u_true-u_pred), cmap='jet',s=0.5,vmin=u_err_vmin, vmax=u_err_vmax)
ax.set_aspect('equal')
ax.set_title('Absolute control error: SEERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

ax = fig.add_subplot(2, 5, 10)
sc = ax.scatter(x_np, y_np,c=np.abs(u_true-u_pred2), cmap='jet',s=0.5,vmin=u_err_vmin, vmax=u_err_vmax)
ax.set_aspect('equal')
ax.set_title('Absolute control error: ERNM')
ax.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("data/singular.png", dpi=100, bbox_inches='tight')
plt.close()

