import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


filename = f"data/KKT_CNP_10D.pth"
loaded_data = torch.load(filename)
U_exact = loaded_data['U_exa'].numpy()
Y_exact = loaded_data['Y_exa'].numpy()
X_1 = loaded_data['x_1'].numpy()
X_2 = loaded_data['x_2'].numpy()


U_preds = []
Y_preds = []
errors = []
errors_y = []
rel_errors = []
rel_errors_y = []


U_pred = loaded_data['U_pred'].numpy()
Y_pred = loaded_data['Y_pred'].numpy()
U_preds.append(U_pred)
Y_preds.append(Y_pred)

rel_error = (np.mean((U_pred - U_exact) ** 2) / np.mean(U_exact ** 2)) ** 0.5
rel_errors.append(rel_error)
print(f"KKT: Relative error U = {rel_error:.6f}")

rel_error_y = (np.mean((Y_pred - Y_exact) ** 2) / np.mean(Y_exact ** 2)) ** 0.5
rel_errors_y.append(rel_error_y)
print(f"KKT: Relative error Y = {rel_error_y:.6f}")

error = np.abs(U_pred - U_exact)
errors.append(error)
error_y = np.abs(Y_pred - Y_exact)
errors_y.append(error_y)



filename = f"data/RNM_high_cos.pth"
#filename = f"data/KKT_CNP_4D.pth"

loaded_data = torch.load(filename)
X_1 = loaded_data['x_1'].numpy()
X_2 = loaded_data['x_2'].numpy()
U_pred_hard = loaded_data['U_pred'].numpy()
Y_pred_hard = loaded_data['Y_pred'].numpy()
U_exact = loaded_data['U_exa'].numpy()
Y_exact = loaded_data['Y_exa'].numpy()
error_hard = np.abs(U_pred_hard - U_exact)
rel_error_hard = (np.mean((U_pred_hard - U_exact) ** 2) / np.mean(U_exact ** 2)) ** 0.5
error_hard_y = np.abs(Y_pred_hard - Y_exact)
rel_error_hard_y = (np.mean((Y_pred_hard - Y_exact) ** 2) / np.mean(Y_exact ** 2)) ** 0.5
print(f"Hard-constraint: Relative error U= {rel_error_hard:.6f}")
print(f"Hard-constraint: Relative error Y= {rel_error_hard_y:.6f}")


all_solutions = [U_exact, U_pred_hard] + U_preds
all_solutions_flat = np.concatenate([sol.flatten() for sol in all_solutions])
z_min = np.min(all_solutions_flat)
z_max = np.max(all_solutions_flat)
all_solutions_y = [Y_exact, Y_pred_hard] + Y_preds
all_solutions_flat_y = np.concatenate([sol.flatten() for sol in all_solutions_y])
z_min_y = np.min(all_solutions_flat_y)
z_max_y = np.max(all_solutions_flat_y)

all_errors = [error_hard] + errors
all_errors_flat = np.concatenate([err.flatten() for err in all_errors])
error_min = 0
error_max = np.max(all_errors_flat)

all_errors_y = [error_hard_y] + errors_y
all_errors_flat_y = np.concatenate([err.flatten() for err in all_errors_y])
error_min_y = 0
error_max_y = np.max(all_errors_flat_y)

fig = plt.figure(figsize=(30, 12))
gs = gridspec.GridSpec(2, 5, height_ratios=[1.2, 1], hspace=0.05, wspace=0.01)

titles_top = ['Exact Solution', f'CNP Method\nRelative L2 Error: $2.19x10^{{-2}}$',
              f'KKT-based Method\nRelative L2 Error: $2.36x10^{{-1}}$']

solutions = [U_exact, U_pred_hard, U_preds[0]]



fig = plt.figure(figsize=(30, 12))
gs = gridspec.GridSpec(2, 5, height_ratios=[1.2, 1], hspace=0.05, wspace=0.01)

titles_top_y = ['Exact State', f'CNP Method\nRelative L2 Error: $2.19x10^{{-2}}$',
              f'Penalty Method ($\\beta=10$)\nRelative L2 Error: $2.36x10^{{-1}}$',
              f'Penalty Method ($\\beta=10^2$)\nRelative L2 Error: $7.49x10^{{-2}}$',
              f'Penalty Method ($\\beta=10^3$)\nRelative L2 Error: $1.66x10^{{-1}}$']

solutions_y = [Y_exact, Y_pred_hard, Y_preds[0]]





titles_top_y = ['Exact State', f'State: Exact Reduced Neural Method\nRelative L2 Error: $3.70x10^{{-3}}$ ',
              f'State: KKT-based Method\nRelative L2 Error: $1.22x10^{{-2}}$',]

titles_top_y_error = [f'Exact Reduced Neural Method\n Absolute Error:$|y_{{NN}}-y|$',
              f'KKT-based Method\n Absolute Error:$|y_{{NN}}-y|$']


fig = plt.figure(figsize=(30, 6))
gs_main = gridspec.GridSpec(1, 2, width_ratios=[1.0, 0.6], wspace=0.05)

gs_left = gridspec.GridSpecFromSubplotSpec(1, 3,
                                          subplot_spec=gs_main[0],
                                          wspace=0.00)  # 3D图之间的间距


gs_right = gridspec.GridSpecFromSubplotSpec(1, 2,
                                           subplot_spec=gs_main[1],
                                           wspace=0.2)  # 只控制这两个图的间距
top_axes = []
solutions_y = [Y_exact, Y_pred_hard, Y_preds[0]]
for i in range(3):
    ax = fig.add_subplot(gs_left[0, i], projection='3d')
    surf = ax.plot_surface(X_1, X_2, solutions_y[i], cmap='inferno',
                           rstride=2, cstride=2, alpha=0.9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlim([z_min_y, z_max_y])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(titles_top_y[i], fontsize=14, pad=1)
    ax.view_init(elev=30, azim=250)
    ax.grid(True, alpha=0.3)
    top_axes.append(ax)


solutions_y = [Y_exact, Y_pred_hard, Y_preds[0]]
errors_list_y = [error_hard_y, errors_y[0]]
error_cmap = 'hot_r'

bottom_axes = []
contours = []
for i in range(2):
    spacing_adjust = 0.1
    ax = fig.add_subplot(gs_right[0, i])
    contour = ax.contourf(X_1, X_2, errors_list_y[i],
                              levels=50, cmap=error_cmap,
                              vmin=error_min_y, vmax=error_max_y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(titles_top_y_error[i], fontsize=14, pad=25)
    ax.set_aspect('equal',adjustable='box')
    contours.append(contour)

    bottom_axes.append(ax)


cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.68])
cbar = plt.colorbar(contours[-1], cax=cbar_ax)



#plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)

filename = f"data/PDE_state_figure_highd_CNPKKT.png"
plt.savefig(filename, dpi=100, bbox_inches='tight')
#plt.show()
plt.close()


titles_top_u = ['Exact Control', f'Control: Exact Reduced Neural Method\nRelative L2 Error: $5.28x10^{{-3}}$ ',
              f'Control: KKT-based Method \nRelative L2 Error: $8.18x10^{{-3}}$',]

titles_top_u_error = [f' Exact Reduced Neural Method\n Absolute Error:$|u_{{NN}}-u|$',
              f'KKT-based Method\n Absolute Error:$|u_{{NN}}-u|$']




fig = plt.figure(figsize=(30, 6))
gs_main = gridspec.GridSpec(1, 2, width_ratios=[1.0, 0.6], wspace=0.05)

gs_left = gridspec.GridSpecFromSubplotSpec(1, 3,
                                          subplot_spec=gs_main[0],
                                          wspace=0.00)  # 3D图之间的间距


gs_right = gridspec.GridSpecFromSubplotSpec(1, 2,
                                           subplot_spec=gs_main[1],
                                           wspace=0.2)  # 只控制这两个图的间距



errors_list = [error_hard, errors[0]]
error_max = np.max([np.max(error_hard.flatten()),np.max(errors[0].flatten())])
error_cmap = 'hot_r'

top_axes = []
solutions_y = [Y_exact, Y_pred_hard, Y_preds[0]]
for i in range(3):
    ax = fig.add_subplot(gs_left[0, i], projection='3d')
    surf = ax.plot_surface(X_1, X_2, solutions[i], cmap='inferno',
                           rstride=2, cstride=2, alpha=0.9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlim([z_min, z_max])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(titles_top_u[i], fontsize=14, pad=1)
    ax.view_init(elev=30, azim=250)
    ax.grid(True, alpha=0.3)
    top_axes.append(ax)

bottom_axes = []
contours = []
for i in range(2):
    spacing_adjust = 0.1
    ax = fig.add_subplot(gs_right[0, i])
    contour = ax.contourf(X_1, X_2, errors_list[i],
                              levels=50, cmap=error_cmap,
                              vmin=error_min, vmax=error_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(titles_top_u_error[i], fontsize=14, pad=25)
    ax.set_aspect('equal',adjustable='box')
    contours.append(contour)

    bottom_axes.append(ax)


cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.68])
cbar = plt.colorbar(contours[-1], cax=cbar_ax)



#plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)

filename = f"data/PDE_control_figure_highd_CNPKKT.png"
plt.savefig(filename, dpi=100, bbox_inches='tight')
#plt.show()
plt.close()
