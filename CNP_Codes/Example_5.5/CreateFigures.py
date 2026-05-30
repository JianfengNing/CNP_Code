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
filename = f"data/Direct_Navier_Stoke.pth"
loaded_data = torch.load(filename)
U_exact = loaded_data['U_exa'].detach().cpu().numpy()[:,0].reshape(n_plot, n_plot)
U_pred = loaded_data['U_pred'].detach().cpu().numpy()[:,0].reshape(n_plot, n_plot)
X_1 = loaded_data['x_1'].numpy()
X_2 = loaded_data['x_2'].numpy()
error_history_reduced = loaded_data['error_history']

error_u = (np.mean((U_pred-U_exact)**2)/np.mean(U_exact**2))**0.5

filename_2 = f"data/Adjoint_Navier_Stoke.pth"
loaded_data2 = torch.load(filename_2)

U_pred_adjoint = loaded_data2['U_pred'].detach().cpu().numpy()[:,0].reshape(n_plot, n_plot)

error_history_adjoint = loaded_data2['error_history']


error_u_adjoint = (np.mean((U_pred_adjoint-U_exact)**2)/np.mean(U_exact**2))**0.5



print(f'Error_u_reduced: {error_u:.6f}, ')
print(f'Error_u_adjoint: {error_u_adjoint:.6f}, ')

fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(1,3, 1, projection='3d')
surf1 = ax1.plot_surface(X_1, X_2, U_exact, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
#ax1.set_zlabel('Control Value')
ax1.set_title('Exact Control $u_1$',pad=0)
ax1.view_init(elev=30, azim=250)
ax1.grid(True, alpha=0.3)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

    # 第二个视角：强调细节特征
ax2 = fig.add_subplot(1,3, 2, projection='3d')
surf2 = ax2.plot_surface(X_1, X_2, U_pred, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title(f'Control $u_1$: Exact Reduced Neural Method\nRelative L2 Error: $6.18x10^{{-3}}$',pad=0)
ax2.view_init(elev=30, azim=250)
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2 = fig.add_subplot(1,3, 3, projection='3d')
ax2.plot_surface(X_1, X_2, U_pred_adjoint, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
#ax2.set_zlabel('State Value')
ax2.set_title(f'Control $u_1$: KKT-based Method\nRelative L2 Error: $1.47x10^{{-2}}$',pad=0)
ax2.view_init(elev=30, azim=250)
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)






plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
plt.savefig("data/NS_high_fre.png", dpi=150, bbox_inches='tight')
plt.close()


plt.figure(figsize=(6, 5))

plt.plot(error_history_reduced, label=f'Exact Reduced Neural Method', linewidth=1.2)
plt.plot(error_history_adjoint, label=f'KKT-based Method', linewidth=1.2)
plt.xlabel('Epoch')
plt.ylabel('Relative L2 Error')
plt.legend()
plt.yscale('log')
#plt.title('Training Error History')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/error_history_NS.png',dpi=150)
plt.show()
plt.close()