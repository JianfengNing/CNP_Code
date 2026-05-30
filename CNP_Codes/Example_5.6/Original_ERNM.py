import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import os


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



# Generate training data
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





def train_model(n_samples=10000, epochs=5000, lr=1e-3):
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

    for epoch in range(10000):
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


    model = ResNetFunction(
        input_dim=2,  # scalar input function
        output_dim=1,  # scalar output
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)


    for epoch in range(epochs):
        model.train()
        x_train = generate_data(n_samples).to(device)
        x_d = x_train.detach().cpu().clone()
        x_train.requires_grad_(True)


        y_d_train= y_d(x_d).to(device)


        y_pred = y_theta(x_train, model,model_boundary)
        lap_y_n = laplacian(y_pred, x_train)


        u_pred = -lap_y_n -f_r(x_d).to(device)


        # Compute loss
        total_loss, data_loss, reg_loss = loss_function(y_pred, u_pred, y_d_train)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()



        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item():6f}, "
                  f"Data Loss: {data_loss.item():6f}, Reg Loss: {reg_loss.item():6f}, ")

            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:6f}')

            y_pred, u_pred = evaluate_model(model,model_boundary)
            professional_plot(y_pred, u_pred)

    return model, train_losses, data_losses, reg_losses, control_errors


# Evaluate model
def evaluate_model(model,model_b):
    model.eval()

    n = 128
    x_1 = torch.linspace(-1, 1, 2 * n + 1)
    y_1 = torch.linspace(1, 0 , n + 1)
    X_1, Y_1 = torch.meshgrid(x_1, y_1, indexing='ij')
    XX_1 = torch.stack([X_1.reshape(-1), Y_1.reshape(-1)], dim=1)

    x_2 = torch.linspace(-1, 0 , n + 1)
    y_2 = torch.linspace(-1 / n, -1 , n)
    X_2, Y_2 = torch.meshgrid(x_2, y_2, indexing='ij')
    XX_2 = torch.stack([X_2.reshape(-1), Y_2.reshape(-1)], dim=1)

    XX = torch.cat([XX_1, XX_2], dim=0)
    x_test = torch.tensor(XX, requires_grad=True)

    # Use torch.enable_grad() to temporarily enable gradient computation
    with torch.enable_grad():

        y_pred = y_theta(x_test.to(device), model,model_b)
        lap_y_n = laplacian(y_pred, x_test)

        u_pred = -lap_y_n -f_r(x_test.detach().cpu())



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
    filename = f"data/ERNM_singularity_ori.pth"
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
    y_true = y_exact(XX).numpy()
    u_true = rate * y_true
    #yd = boundary_function(XX).numpy()

    error_y = (np.mean((y_pred - y_true) ** 2)/np.mean(y_true ** 2)) ** 0.5
    error_u = (np.mean((u_pred - u_true) ** 2) / np.mean(u_true ** 2))**0.5
    print(f'error_y: {error_y:.6f}, error_u: {error_u:.6f}')


    fig = plt.figure(figsize=(18, 10))

    # 左侧子图 (3D)
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(x_np, y_np,c=y_pred, cmap='jet',s=0.1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Predicted Y')
    ax1.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8)
    ax1.set_axis_off()



    # 右侧子图 (3D)
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

    # 为左侧散点图添加颜色条
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.8)
    ax1.set_axis_off()

    # 右侧子图 (3D)
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


    # 确保 data 文件夹存在
    os.makedirs("data", exist_ok=True)

    # 保存图片
    plt.savefig("data/irregular——original.png", dpi=150, bbox_inches='tight')
    plt.close()





# Main execution
if __name__ == "__main__":
    model, train_losses, data_losses, reg_losses, control_errors = train_model(
        n_samples=50, epochs=20000, lr=1e-3
    )

    y_pred, u_pred = evaluate_model(model)

    filename = f"data/CNP_state_bound.pth"
    torch.save({

        'U_pred': u_pred.reshape(256, 256).detach().cpu(),
        'Y_pred': y_pred.reshape(256, 256).detach().cpu(),
    }, filename)

    print('finished')
