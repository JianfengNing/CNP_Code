import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")



penalty_values = [10, 100, 1000,10000]
U_preds = []
U_preds_deri = []
rel_errors = []
rel_errors_derivative = []
Loss_history = []
Error_history = []


for penalty_para in penalty_values:
    filename = f"data/penalty_{penalty_para}_example2.pth"
    loaded_data = torch.load(filename)
    U_pred = loaded_data['u_pred'].numpy()
    U_exact = loaded_data['u_exact'].numpy()
    U_pred_deri = loaded_data['u_deriv'].numpy()
    U_exact_deri = loaded_data['u_exa_deri'].numpy()
    loss_history = loaded_data['loss_history']
    error_history = loaded_data['error_history']
    U_preds.append(U_pred)
    U_preds_deri.append(U_pred_deri)
    Loss_history.append(loss_history)
    Error_history.append(error_history)

    rel_error = (np.mean((U_pred - U_exact) ** 2) / np.mean(U_exact ** 2)) ** 0.5
    rel_errors.append(rel_error)

    rel_error_deriva = (np.mean((U_pred_deri - U_exact_deri) ** 2) / np.mean(U_exact_deri ** 2)) ** 0.5
    rel_errors_derivative.append(rel_error_deriva)
    print(f"Penalty λ={penalty_para}: Relative error = {rel_error:.6f}")
    print(f"Penalty λ={penalty_para}: Relative derivative error = {rel_error_deriva:.6f}")



filename = f"data/my_example2.pth"
loaded_data = torch.load(filename)
x_test = loaded_data['x_test'].detach().cpu().numpy()
U_pred_hard = loaded_data['u_pred'].numpy()
U_exact = loaded_data['u_exact'].numpy()
U_pred_deri_hard = loaded_data['u_deriv'].numpy()
U_exact_deri = loaded_data['u_exa_deri'].detach().cpu().numpy()
loss_history_hard = loaded_data['loss_history']
error_history_hard = loaded_data['error_history']
rel_error_hard = (np.mean((U_pred_hard - U_exact) ** 2) / np.mean(U_exact ** 2)) ** 0.5
rel_error_deriva_hard = (np.mean((U_pred_deri_hard - U_exact_deri) ** 2) / np.mean(U_exact_deri ** 2)) ** 0.5
print(f"Hard-constraint: Relative error = {rel_error_hard:.6f}")
print(f"Hard-constraint: Relative derivative error = {rel_error_deriva_hard:.6f}")

plt.figure(figsize=(6, 5))

plt.plot(Loss_history[0], label=f'Penalty Method $\\beta=10$', linewidth=1.2)
plt.plot(Loss_history[1], label=f'Penalty Method $\\beta=10^2$', linewidth=1.2)
plt.plot(Loss_history[2], label=f'Penalty Method $\\beta=10^3$', linewidth=1.2)
plt.plot(loss_history_hard , label='CNP Method', linewidth=1.2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
#plt.title('Training Loss History')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/loss_history.png',dpi=150)
plt.show()
plt.close()


plt.figure(figsize=(6, 5))

plt.plot(Error_history[0], label=f'Penalty Method $\\beta=10$', linewidth=1.2)
plt.plot(Error_history[1], label=f'Penalty Method $\\beta=10^2$', linewidth=1.2)
plt.plot(Error_history[2], label=f'Penalty Method $\\beta=10^3$', linewidth=1.2)
plt.plot(error_history_hard , label='CNP Method', linewidth=1.2)
plt.xlabel('Epoch')
plt.ylabel('Relative L2 Error')
plt.legend()
plt.yscale('log')
#plt.title('Training Error History')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/error_history.png',dpi=150)
plt.show()
plt.close()


fig = plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_test, U_exact, 'r-', linewidth=2, label='Exact Solution')
plt.plot(x_test, U_pred_hard, 'b--', linewidth=2, label='CNP Solution')
plt.xlabel('x',fontsize=12)
#plt.ylabel(r"u(x)")
plt.legend(fontsize=10)
#plt.title('Solutions')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_test, U_exact_deri, 'r-', linewidth=2, label='Exact Derivative')
plt.plot(x_test, U_pred_deri_hard, 'b--', linewidth=2, label='CNP Derivative')
plt.xlabel('x',fontsize=12)
#plt.ylabel(r"u'(x)")
plt.ylim([-25, 25.5])
plt.legend(fontsize=10)
#plt.title('Solutions')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/hard_method.png',dpi=150)
plt.show()

labels = [f'Penalty Method Solution ($\\beta=10$)', f'Penalty Method Solution ($\\beta=10^2$)',
          f'Penalty Method Solution ($\\beta=10^3$))']

labels_deri = [f'Penalty Method Derivative ($\\beta=10$)', f'Penalty Method Derivative ($\\beta=10^2$)',
          f'Penalty Method Derivative ($\\beta=10^3$)']

for i in range(3):
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test, U_exact, 'r-', linewidth=2, label=f'Exact Solution')
    plt.plot(x_test, U_preds[i], 'b--', linewidth=2, label=labels[i])
    plt.xlabel('x', fontsize=12)
    # plt.ylabel(r"u(x)")
    plt.legend(fontsize=10)
    # plt.title('Solutions')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_test, U_exact_deri, 'r-', linewidth=2, label=r"Exact Derivative")
    plt.plot(x_test, U_preds_deri[i], 'b--', linewidth=2, label=labels_deri[i])
    plt.xlabel('x', fontsize=12)
    # plt.ylabel(r"u'(x)")
    plt.ylim([-25, 25.5])
    plt.legend(fontsize=10)
    # plt.title('Solutions')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'data/penalty_{i}.png', dpi=150)
    #plt.show()
    plt.close()

kkkk
