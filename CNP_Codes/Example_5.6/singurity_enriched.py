import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import argparse


pi = math.pi
import random

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


print(f"Using device: {device}")


# Set random seeds
torch.manual_seed(42)
if device.type == 'mps':
    torch.mps.manual_seed(42)
np.random.seed(45)


R=0.5
alpha=0.5
rate = 50
class FCResBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out + identity)


class ResNetFunction(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=64,
            num_layers=4,
            activation_fn=nn.Tanh
    ):
        super().__init__()
        self.activation = activation_fn()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[FCResBlock(hidden_dim, self.activation) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


def singular_part(X):
    x = X[:, 0:1].numpy()
    y = X[:, 1:2].numpy()
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    sd = r ** (2 / 3) * np.sin(2 * theta / 3)
    etad = np.where(r < R, 15 / 16 * (
            8 / 15 - (4 * r / R - 3) + 2 / 3 * (4 * r / R - 3) ** 3 - 1 / 5 * (4 * r / R - 3) ** 5), 0)
    etad = np.where(r < R / 2, 1, etad)
    return torch.tensor(sd*etad)


def Deltap(X):
    x_1_f = X[:,0:1]
    x_2_f = X[:,1:2]
    r = torch.sqrt(x_1_f ** 2 + x_2_f ** 2)
    theta = np.arctan2(x_2_f, x_1_f)
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
    deltap = torch.where(r < R,
                         -4 * (-7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (8 * r - 3) ** 3 + 3.3125) * np.sin(
                             2 * theta / 3) / (9 * r ** (4 / 3)) + (2 / 3 * (
                                 -7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (
                                 8 * r - 3) ** 3 + 3.3125) * np.sin(2 * theta / 3) / r ** (1 / 3) + r ** (
                                                                            2 / 3) * (
                                                                            -7.5 * (8 * r - 3) ** 4 + 15.0 * (
                                                                            8 * r - 3) ** 2 - 7.5) * np.sin(
                             2 * theta / 3) + r * (-2 / 9 * (-7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (
                                 8 * r - 3) ** 3 + 3.3125) * np.sin(2 * theta / 3) / r ** (4 / 3) + 4 / 3 * (
                                                           -7.5 * (8 * r - 3) ** 4 + 15.0 * (
                                                           8 * r - 3) ** 2 - 7.5) * np.sin(
                             2 * theta / 3) / r ** (1 / 3) + r ** (2 / 3) * (
                                                           1920.0 * r - 240.0 * (8 * r - 3) ** 3 - 720.0) * np.sin(
                             2 * theta / 3))) / r, 0)
    deltap = torch.where(r < R / 2, 0, deltap)
    return deltap




def y_exact(X):
    x = X[:,0:1].numpy()
    y = X[:,1:2].numpy()
    v_true = np.where(y <= 0, np.sin(2 * np.pi * x) * (1 / 2 * y ** 2 + y) * (y ** 2 - 1),
                      np.sin(2 * np.pi * x) * (-1 / 2 * y ** 2 + y) * (y ** 2 - 1))
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    sd = r ** (2 / 3) * np.sin(2 * theta / 3)
    etad = np.where(r < R, 15 / 16 * (
            8 / 15 - (4 * r / R - 3) + 2 / 3 * (4 * r / R - 3) ** 3 - 1 / 5 * (4 * r / R - 3) ** 5), 0)
    etad = np.where(r < R / 2, 1, etad)
    y_true = v_true + sd * etad
    return torch.tensor(y_true)

def F_r(X):
    x_1_f = X[:,0:1]
    x_2_f = X[:,1:2]
    r = torch.sqrt(x_1_f ** 2 + x_2_f ** 2)
    theta = np.arctan2(x_2_f, x_1_f)
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
    deltap = torch.where(r < R,
                         -4 * (-7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (8 * r - 3) ** 3 + 3.3125) * np.sin(
                             2 * theta / 3) / (9 * r ** (4 / 3)) + (2 / 3 * (
                                 -7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (
                                 8 * r - 3) ** 3 + 3.3125) * np.sin(2 * theta / 3) / r ** (1 / 3) + r ** (
                                                                            2 / 3) * (
                                                                            -7.5 * (8 * r - 3) ** 4 + 15.0 * (
                                                                            8 * r - 3) ** 2 - 7.5) * np.sin(
                             2 * theta / 3) + r * (-2 / 9 * (-7.5 * r - 0.1875 * (8 * r - 3) ** 5 + 0.625 * (
                                 8 * r - 3) ** 3 + 3.3125) * np.sin(2 * theta / 3) / r ** (4 / 3) + 4 / 3 * (
                                                           -7.5 * (8 * r - 3) ** 4 + 15.0 * (
                                                           8 * r - 3) ** 2 - 7.5) * np.sin(
                             2 * theta / 3) / r ** (1 / 3) + r ** (2 / 3) * (
                                                           1920.0 * r - 240.0 * (8 * r - 3) ** 3 - 720.0) * np.sin(
                             2 * theta / 3))) / r, 0)
    deltap = torch.where(r < R / 2, 0, deltap)
    F = torch.where(x_2_f <= 0,
                    np.sin(2 * np.pi * x_1_f) * (2 * np.pi ** 2 * (x_2_f ** 2 + 2 * x_2_f) * (x_2_f ** 2 - 1) - (
                            6 * x_2_f ** 2 + 6 * x_2_f - 1)) - deltap,
                    np.sin(2 * np.pi * x_1_f) * (2 * np.pi ** 2 * (-x_2_f ** 2 + 2 * x_2_f) * (x_2_f ** 2 - 1) - (
                            -6 * x_2_f ** 2 + 6 * x_2_f + 1)) - deltap)
    return F

def f_r(X):
    return F_r(X) - rate * y_exact(X)


def y_d(X):
    yd = (alpha*rate) * F_r(X) + y_exact(X)
    return yd




def bound_whole(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return  (1 - x ** 2) * (1 - y ** 2)


def boundary_function(X,model_b):
    return bound_whole(X)*model_b(X)



def y_theta(X,model,model_b):
    return model(X) * boundary_function(X,model_b)


def lshape_boundary_distance(X: torch.Tensor) -> torch.Tensor:
    x = X[:,0:1]
    y = X[:,1:2]

    # S1: x=-1, y in [-1,1]
    d1 = torch.abs(x + 1)

    # S2: y=1, x in [-1,1]
    d2 = torch.abs(y - 1)

    # S3: x=1, y in [0,1]
    d3 = torch.sqrt((x - 1) ** 2 + torch.clamp(-y, min=0.0) ** 2)

    # S4: y=0, x in [0,1]
    d4 = torch.sqrt(torch.clamp(-x, min=0.0) ** 2 + y ** 2)

    # S5: x=0, y in [-1,0]
    d5 = torch.sqrt(x ** 2 + torch.clamp(y, min=0.0) ** 2)

    # S6: y=-1, x in [-1,0]
    d6 = torch.sqrt(torch.clamp(x, min=0.0) ** 2 + (y + 1) ** 2)

    d = torch.stack([d1, d2, d3, d4, d5, d6], dim=-1)
    return torch.min(d, dim=-1).values

def sample_boundary(n_each: int, device=device) -> torch.Tensor:
    t1 = torch.rand(n_each, 1, device=device)
    s1 = torch.cat([t1, torch.zeros_like(t1)], dim=1)  # y=0, x in [0,1]
    t2 = -torch.rand(n_each, 1, device=device)
    s2 = torch.cat([torch.zeros_like(t2), t2], dim=1)  # x=0, y in [-1,0]
    return torch.cat([s1, s2], dim=0)


def lifting_diagnostic_points(interior_n=256, boundary_n=4096):
    """Fixed evaluation grids, independent of the randomly varying training grid.

    The interior grid has spacing 1/interior_n and excludes the whole boundary
    of the open L-shaped domain. Gamma quadrature includes segment endpoints;
    their inclusion does not change the continuous boundary integral.
    """
    if interior_n < 2 or boundary_n < 1:
        raise ValueError("Require interior_n >= 2 and boundary_n >= 1.")
    axis = np.arange(-interior_n + 1, interior_n, dtype=np.float64) / interior_n
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    inside = (xx < 0.0) | (yy > 0.0)
    interior = np.column_stack((xx[inside], yy[inside]))

    t = np.linspace(0.0, 1.0, boundary_n + 1)
    horizontal = np.column_stack((t, np.zeros_like(t)))
    vertical = np.column_stack((np.zeros_like(t), -t))
    return interior, horizontal, vertical


def summarize_lifting_diagnostics(interior, w_interior, w_gamma,
                                  interior_n, boundary_n):
    """Compute sampled extrema and an arc-length L2 quadrature on Gamma.

    w_gamma has shape (2, boundary_n + 1), one row per unit-length segment.
    No finite grid certifies positivity or a supremum over the continuum.
    """
    w_interior = np.asarray(w_interior, dtype=np.float64).reshape(-1)
    w_gamma = np.asarray(w_gamma, dtype=np.float64)
    if w_interior.size != len(interior) or w_gamma.shape != (2, boundary_n + 1):
        raise ValueError("Diagnostic values do not match the evaluation grids.")
    if not (np.isfinite(w_interior).all() and np.isfinite(w_gamma).all()):
        raise ValueError("The lifting diagnostics contain NaN or infinite values.")
    i_min = int(np.argmin(w_interior))
    squared = w_gamma ** 2
    # Both segments have length one, so their quadrature spacing is 1/boundary_n.
    integral = float(np.sum(
        0.5 * squared[:, 0] + squared[:, 1:-1].sum(axis=1)
        + 0.5 * squared[:, -1]
    ) / boundary_n)
    return {
        "interior_grid_spacing": 1.0 / interior_n,
        "interior_points": int(len(interior)),
        "interior_min": float(w_interior[i_min]),
        "interior_min_point": interior[i_min].tolist(),
        "interior_negative_count": int(np.count_nonzero(w_interior < 0.0)),
        "interior_nonpositive_count": int(np.count_nonzero(w_interior <= 0.0)),
        "gamma_intervals_per_segment": int(boundary_n),
        "gamma_nodes_per_segment": int(boundary_n + 1),
        "gamma_quadrature": "Composite trapezoidal rule; two unit-length segments",
        "gamma_l2_residual": float(np.sqrt(integral)),
        "gamma_rms_residual": float(np.sqrt(integral / 2.0)),
        "gamma_max_abs_sampled": float(np.max(np.abs(w_gamma))),
        "interpretation": (
            "The interior minimum and boundary maximum are sampled extrema, "
            "not certified continuum bounds. The L2 residual is a quadrature estimate."
        ),
    }


def report_lifting_diagnostics(model_boundary, interior_n=256, boundary_n=4096,
                               batch_size=8192, output_dir="data"):
    """Evaluate w = (1-x^2)(1-y^2) N_b, print diagnostics, and save JSON.

    Can also be called on an already trained/loaded lifting network. Evaluation
    uses the network's device and dtype; reductions are accumulated in float64.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    interior, horizontal, vertical = lifting_diagnostic_points(interior_n, boundary_n)
    parameter = next(model_boundary.parameters())
    was_training = model_boundary.training
    model_boundary.eval()

    def evaluate(points):
        values = []
        with torch.no_grad():
            for start in range(0, len(points), batch_size):
                batch = torch.as_tensor(points[start:start + batch_size],
                                        device=parameter.device, dtype=parameter.dtype)
                values.append(boundary_function(batch, model_boundary)
                              .detach().cpu().numpy().reshape(-1))
        return np.concatenate(values)

    try:
        w_interior = evaluate(interior)
        w_gamma = np.stack((evaluate(horizontal), evaluate(vertical)))
    finally:
        model_boundary.train(was_training)

    diagnostics = summarize_lifting_diagnostics(
        interior, w_interior, w_gamma, interior_n, boundary_n)
    diagnostics["evaluation_dtype"] = str(parameter.dtype)
    diagnostics["evaluation_device"] = str(parameter.device)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "lifting_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nLearned lifting function diagnostics (after lifting training)")
    print(f"  Interior: {len(interior)} points, spacing h = 1/{interior_n}")
    print(f"  Sampled interior minimum: {diagnostics['interior_min']:.8e}")
    print(f"  Minimum location: {diagnostics['interior_min_point']}")
    print(f"  Negative interior samples: {diagnostics['interior_negative_count']}")
    print(f"  Nonpositive interior samples: {diagnostics['interior_nonpositive_count']}")
    print(f"  Gamma: {boundary_n + 1} nodes on each unit-length segment")
    print(f"  Gamma L2 residual (quadrature): {diagnostics['gamma_l2_residual']:.8e}")
    print(f"  Gamma RMS residual: {diagnostics['gamma_rms_residual']:.8e}")
    print(f"  Gamma maximum absolute residual (sampled): "
          f"{diagnostics['gamma_max_abs_sampled']:.8e}")
    print("  These finite-grid checks do not certify continuum positivity/feasibility.")
    print(f"  Saved diagnostics to {path}\n")
    return diagnostics



def generate_data(n_samples=50):
    n = n_samples + random.randint(1, 50)
    x_1 = torch.linspace(-1, 1, 2 * n + 1)
    y_1 = torch.linspace(1, 0, n + 1)
    X_1, Y_1 = torch.meshgrid(x_1, y_1, indexing='ij')
    XX_1 = torch.stack([X_1.reshape(-1), Y_1.reshape(-1)], dim=1)

    x_2 = torch.linspace(-1, 0, n + 1)
    y_2 = torch.linspace(-1 / n, -1, n)
    X_2, Y_2 = torch.meshgrid(x_2, y_2, indexing='ij')
    XX_2 = torch.stack([X_2.reshape(-1), Y_2.reshape(-1)], dim=1)

    XX = torch.cat([XX_1, XX_2], dim=0)
    return XX


def laplacian(y, x):
    grads = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]   # (N,2)
    y_x = grads[:, 0:1]
    y_y = grads[:, 1:2]
    y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0][:, 0:1]
    y_yy = torch.autograd.grad(y_y, x, torch.ones_like(y_y), create_graph=True)[0][:, 1:2]
    return y_xx + y_yy


def loss_function(y_pred, u, y_d):
    data_loss = 0.5 * torch.mean((y_pred - y_d) ** 2)
    reg_loss = 0.5 * alpha * torch.mean(u ** 2)
    total_loss = data_loss + reg_loss
    return total_loss, data_loss, reg_loss


def train_model(n_samples=50, epochs=5000, lr=1e-3,
                boundary_epochs=10000, lifting_only=False, plot_every=200):
    model_boundary = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        activation_fn=nn.Tanh,
    ).to(device)

    optimizer_b = torch.optim.Adam([
        {'params': model_boundary.parameters(), 'lr': 1e-3},])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=500, gamma=0.8)

    for epoch in range(boundary_epochs):
        x_train = generate_data(n_samples).to(device)
        dis_train = lshape_boundary_distance(x_train)
        boundry_nn = boundary_function(x_train,model_boundary)
        x_train_b = sample_boundary(100).to(device)
        dis_train_b = lshape_boundary_distance(x_train_b)
        boundry_nn_b = boundary_function(x_train_b,model_boundary)
        loss_b = torch.mean((boundry_nn_b - dis_train_b)**2)
        loss_in = torch.mean((boundry_nn - dis_train)**2)
        loss = loss_in + 1000*loss_b
        optimizer_b.zero_grad()
        loss.backward()
        optimizer_b.step()
        scheduler.step()

        if epoch % 200 == 0:
            print('loss_b',loss_b,'loss_in',loss_in)

    # The lifting is now fixed. Freezing parameters still permits spatial
    # derivatives with respect to x during the subsequent PDE training.
    model_boundary.eval()
    model_boundary.requires_grad_(False)
    model_boundary.zero_grad(set_to_none=True)
    diagnostics = report_lifting_diagnostics(model_boundary)
    torch.save({
        'model_state_dict': {key: value.detach().cpu()
                             for key, value in model_boundary.state_dict().items()},
        'architecture': {'input_dim': 2, 'output_dim': 1, 'hidden_dim': 64,
                         'num_layers': 2, 'activation_fn': 'Tanh'},
        'boundary_epochs': boundary_epochs,
        'diagnostics': diagnostics,
    }, "data/lifting_model.pth")
    if lifting_only:
        return None, model_boundary, None


    model = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    gamma = torch.tensor(0.0, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': gamma, 'lr': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)


    for epoch in range(epochs):
        model.train()
        x_train = generate_data(n_samples).to(device)
        x_d = x_train.detach().cpu().clone()
        x_train.requires_grad_(True)


        y_d_train= y_d(x_d).to(device)


        y_n = y_theta(x_train, model,model_boundary)
        lap_y_n = laplacian(y_n, x_train)
        y_pred = y_n + gamma * singular_part(x_d).to(device)

        u_pred = -lap_y_n -gamma * Deltap(x_d).to(device)-f_r(x_d).to(device)


        total_loss, data_loss, reg_loss = loss_function(y_pred, u_pred, y_d_train)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()



        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item():6f}, "
                  f"Data Loss: {data_loss.item():6f}, Reg Loss: {reg_loss.item():6f}, ")

            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:6f}')

            print(f'gamma"{gamma:6f}')

        if plot_every and epoch % plot_every == 0:
            y_pred, u_pred = evaluate_model(model, model_boundary, gamma)
            professional_plot(y_pred, u_pred)

    return model, model_boundary, gamma


def evaluate_model(model,model_b,gamma):
    model.eval()

    n = 128
    x_1 = torch.linspace(-1, 1 , 2 * n + 1)
    y_1 = torch.linspace(1 , 0 , n + 1)
    X_1, Y_1 = torch.meshgrid(x_1, y_1, indexing='ij')
    XX_1 = torch.stack([X_1.reshape(-1), Y_1.reshape(-1)], dim=1)

    x_2 = torch.linspace(-1 , 0 , n + 1)
    y_2 = torch.linspace(-1 / n, -1 , n)
    X_2, Y_2 = torch.meshgrid(x_2, y_2, indexing='ij')
    XX_2 = torch.stack([X_2.reshape(-1), Y_2.reshape(-1)], dim=1)

    XX = torch.cat([XX_1, XX_2], dim=0)
    x_test = XX.detach().clone().to(device).requires_grad_(True)
    x_cpu = XX.detach().cpu()

    with torch.enable_grad():

        y_n = y_theta(x_test, model,model_b)
        lap_y_n = laplacian(y_n, x_test)
        y_pred = y_n + gamma * singular_part(x_cpu).to(device)
        u_pred = -lap_y_n - gamma * Deltap(x_cpu).to(device)-f_r(x_cpu).to(device)



        return y_pred, u_pred



def professional_plot(y_pred, u_pred):

    n = 128
    x_1 = torch.linspace(-1, 1, 2 * n + 1)
    y_1 = torch.linspace(1, 0, n + 1)
    X_1, Y_1 = torch.meshgrid(x_1, y_1, indexing='ij')
    XX_1 = torch.stack([X_1.reshape(-1), Y_1.reshape(-1)], dim=1)

    x_2 = torch.linspace(-1, 0, n + 1)
    y_2 = torch.linspace(-1 / n, -1, n)
    X_2, Y_2 = torch.meshgrid(x_2, y_2, indexing='ij')
    XX_2 = torch.stack([X_2.reshape(-1), Y_2.reshape(-1)], dim=1)
    XX = torch.cat([XX_1, XX_2], dim=0)
    x_np = XX[:, 0].numpy()
    y_np = XX[:, 1].numpy()
    y_true = y_exact(XX)
    u_true = rate * y_true
    os.makedirs("data", exist_ok=True)
    filename = f"data/ERNM_singularity.pth"
    torch.save({
        'U_pred': u_pred.detach().cpu(),
        'Y_pred': y_pred.detach().cpu(),
        'U_exact': u_true.detach().cpu(),
        'Y_exact': y_true.detach().cpu(),
        'X': XX[:, 0],
        'Y': XX[:, 1],
    }, filename)


    y_pred = y_pred.detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy()
    y_true = y_true.numpy()
    u_true = u_true.numpy()

    error_y = (np.mean((y_pred - y_true) ** 2)/np.mean(y_true ** 2)) ** 0.5
    error_u = (np.mean((u_pred - u_true) ** 2) / np.mean(u_true ** 2))**0.5
    print(f'error_y: {error_y:.6f}, error_u: {error_u:.6f}')


    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(x_np, y_np,c=y_pred, cmap='jet',s=0.1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Predicted Y')
    ax1.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8)


    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(x_np, y_np, c=y_true, cmap='jet', s=0.1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('True Y')
    ax2.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    ax2.set_axis_off()

    ax2 = fig.add_subplot(2, 3, 3)
    sc2 = ax2.scatter(x_np, y_np, c=np.abs(y_true-y_pred), cmap='jet', s=0.1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Error')
    ax2.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    ax2.set_axis_off()



    ax1 = fig.add_subplot(2, 3, 4)
    sc1 = ax1.scatter(x_np, y_np, c=u_pred, cmap='jet', s=0.1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Predicted U')
    ax1.grid(True, linestyle='--', alpha=0.3)

    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(2, 3, 5)
    sc2 = ax2.scatter(x_np, y_np, c=u_true, cmap='jet', s=0.1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('True U')
    ax2.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    ax2.set_axis_off()

    ax2 = fig.add_subplot(2, 3, 6)
    sc2 = ax2.scatter(x_np, y_np, c=np.abs(u_true - u_pred), cmap='jet', s=0.1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Error')
    ax2.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    ax2.set_axis_off()


    os.makedirs("data", exist_ok=True)

    plt.savefig("data/irregular.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singularity-enriched neural method")
    parser.add_argument("--lifting-only", action="store_true",
                        help="Train the lifting and report diagnostics, then stop.")
    args = parser.parse_args()
    model, model_boundary, gamma = train_model(
        n_samples=50, epochs=20000, lr=1e-3, lifting_only=args.lifting_only
    )

    if model is not None:
        y_pred, u_pred = evaluate_model(model, model_boundary, gamma)
        # Saves the unstructured L-shaped grid and its predictions, without
        # attempting to reshape it into a full rectangular grid.
        professional_plot(y_pred, u_pred)

    print('finished')
