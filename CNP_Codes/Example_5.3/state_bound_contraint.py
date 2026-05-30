import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
pi = math.pi
import random

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

torch.manual_seed(42)
if device.type == 'mps':
    torch.mps.manual_seed(42)
np.random.seed(45)


lambda_val = 0.1
d = 2
psi = 0.01
psi_root = psi ** 0.5


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
        x_original = x
        x = self.activation(self.input_layer(x))
        x = self.blocks(x)
        x = self.output_layer(x)
        boundary_factor = (1-x_original[:,0:1])* x_original[:,0:1] * (1-x_original[:,1:2])* x_original[:,1:2]
        x = x * boundary_factor
        return x



def yd(x):
    return 10*(torch.sin(2*pi*x[:,0:1])+x[:,1:2])



def generate_data(n_samples=50):
    n_plot = n_samples + random.randint(1, 50)
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return x


def laplacian(y, x):
    grads = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]   # (N,2)
    y_x = grads[:, 0:1]
    y_y = grads[:, 1:2]
    y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0][:, 0:1]
    y_yy = torch.autograd.grad(y_y, x, torch.ones_like(y_y), create_graph=True)[0][:, 1:2]
    return y_xx + y_yy




def loss_function(y_pred, u, y_d):
    data_loss = 0.5 * torch.mean((y_pred - y_d) ** 2)
    reg_loss = 0.5 * lambda_val * torch.mean(u ** 2)
    total_loss = data_loss + reg_loss
    return total_loss, data_loss, reg_loss



def train_model(n_samples=10000, epochs=5000, lr=1e-3):
    model = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    train_losses = []
    data_losses = []
    reg_losses = []
    control_errors = []

    print("Starting training...")
    print(f"Training data points: {n_samples}")

    for epoch in range(epochs):
        model.train()
        x_train = generate_data(n_samples).to(device)
        x_train.requires_grad_(True)

        y_d_train= yd(x_train).detach()

        NN = model(x_train) + psi_root
        y_n = psi-NN**2

        lap_y = laplacian(y_n, x_train)
        u_pred = -lap_y

        total_loss, data_loss, reg_loss = loss_function(y_n, u_pred, y_d_train)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(total_loss.item())
        data_losses.append(data_loss.item())
        reg_losses.append(reg_loss.item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item():6f}, "
                  f"Data Loss: {data_loss.item():6f}, Reg Loss: {reg_loss.item():6f}, ")

            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:6f}')

            y_pred, u_pred = evaluate_model(model)
            professional_plot(y_pred, u_pred)

    return model, train_losses, data_losses, reg_losses, control_errors


def evaluate_model(model):
    model.eval()

    n_plot = 256
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    x_test = torch.tensor(x, requires_grad=True)

    with torch.enable_grad():
        NN = model(x_test) + psi_root
        y_n = psi - NN ** 2

        lap_y = laplacian(y_n, x_test)

        u_pred = -lap_y

        return y_n, u_pred



def professional_plot(y_pred, u_pred):
    n_plot = 256
    y_pred_cpu = y_pred.cpu().detach().numpy().reshape(256, 256)
    u_pred_cpu = u_pred.cpu().detach().numpy().reshape(256, 256)

    x_plot = torch.linspace(0, 1, n_plot)
    y_plot = torch.linspace(0, 1, n_plot)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    X = X.numpy()
    Y = Y.numpy()

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_pred_cpu, cmap='viridis',
                             rstride=2, cstride=2, alpha=0.9)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Control Value')
    ax1.set_title('(b) Optimal Control', fontweight='bold')
    ax1.view_init(elev=30, azim=250)
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    surf2 = ax2.plot_surface(X, Y, y_pred_cpu, cmap='plasma',
                             rstride=1, cstride=1, alpha=0.9)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('State Value')
    ax2.set_title('(a) Optimal State', fontweight='bold')
    ax2.view_init(elev=30, azim=250)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)


    plt.tight_layout()
    plt.savefig("data/Professional_3D_View.png", dpi=150, bbox_inches='tight')
    plt.close()


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
