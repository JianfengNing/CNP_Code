import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
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

nu = 0.05

ratio = 100.0

torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, in_put_dim =2, hidden_layers=3, hidden_units=50):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_put_dim, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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


def y_exact(x):
    mask = (x[:, 0:1] < 0.5) & (x[:, 1:2] < 0.8)
    y = torch.zeros_like(x[:, 0:1])

    z1 = -4096 * (x[:, 0:1] ** 6) + 6144 * (x[:, 0:1] ** 5) - 3072 * (x[:, 0:1] ** 4) + 512 * (x[:, 0:1] ** 3)
    z2 = -244.140625 * (x[:, 1:2] ** 6) + 585.9375 * (x[:, 1:2] ** 5) - 468.75 * (x[:, 1:2] ** 4) + 125 * (
                x[:, 1:2] ** 3)

    y[mask] = z1[mask] * z2[mask]
    return y


def u_exact(x):
    return ratio*y_exact(x)




def ksi(x):
    a = -torch.abs(x[:,0:1]-0.8)-torch.abs((x[:,1:2]-0.2)*x[:,0:1]-0.3)+0.35
    return 25*2 * torch.maximum(0*x[:,0:1],a)

def f_yd(x):
    y = y_exact(x)
    laplace_y = laplacian(y, x)
    Ksi = ksi(x)
    uu = ratio * y
    f = -laplace_y  - uu - Ksi
    y_d = y + Ksi - nu * ratio * laplace_y


    return f.detach(), y_d.detach()


def boundary_fun(x):
    return torch.sin(pi * x[:,0:1])* torch.sin(pi * x[:,1:2])



def laplacian(y, x):
    grads = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]   # (N,2)
    y_x = grads[:, 0:1]
    y_y = grads[:, 1:2]
    y_xx = autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0][:, 0:1]
    y_yy = autograd.grad(y_y, x, torch.ones_like(y_y), create_graph=True)[0][:, 1:2]
    return y_xx + y_yy



def generate_data(n_samples=256):
    n_plot = 256
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return x


def generate_data_unifrom_random(n_samples=50):
    n_plot = n_samples+ random.randint(1, 50)
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return x

def main():

    nn_xi = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    nn_y = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)


    optimizer = torch.optim.Adam([
         {'params': nn_xi.parameters(), 'lr': 1e-3},
         {'params': nn_y.parameters(), 'lr': 1e-3}
     ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    num_epochs = 20000

    beta =  10**2
    loss_history = []
    tv_history = []
    data_history = []




    for epoch in range(num_epochs):


        x = generate_data_unifrom_random(50).to(device)
        x.requires_grad_(True)


        n_y = nn_y(x)
        N_y = (n_y**2) * boundary_fun(x)
        N_xi = nn_xi(x)**2


        integ = torch.mean(N_y*N_xi)



        f, y_d = f_yd(x)
        Delta_y = laplacian(N_y,x)
        u = -Delta_y-N_xi-f
        u_term = 0.5*nu*torch.mean(u ** 2)
        data_term = 0.5 * torch.mean((N_y - y_d) ** 2)
        loss = u_term + data_term + beta * integ

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        tv_history.append(u_term.item())
        data_history.append(data_term.item())

        if epoch % 200 == 0:
            print(f'Epoch {epoch:4d}, Loss: {loss.item():.6f}, '
                  f'TV: {u_term.item():.6f}, Data: {data_term.item():.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e},')


            Y_exact= y_exact(x)
            U_exact = u_exact(x)
            error_y = (torch.mean((N_y - Y_exact)**2)/torch.mean(Y_exact**2))**0.5
            error_u = (torch.mean((u - U_exact)**2)/torch.mean(U_exact**2))**0.5
            print(f'Error_y: {error_y:.6f}, '
                  f'Error_u: {error_u.item():.6f}, 'f'integral: {integ:.6f}, ')

            evaluate_model(nn_y, nn_xi, n_test=5000)


def evaluate_model(model_y,model_xi, n_test=5000):
    model_y.eval()
    model_xi.eval()

    n_plot = 256
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x_test = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    x_test = torch.tensor(x_test, requires_grad=True)

    with torch.enable_grad():
        f,yd = f_yd(x_test)
        n_y = model_y(x_test)
        y_pred = (n_y ** 2) * boundary_fun(x_test)
        N_xi = model_xi(x_test) ** 2

        Delta_y = laplacian(y_pred, x_test)



        u_pred = -Delta_y - N_xi - f

        Y_exact = y_exact(x_test)
        LapY_exact = laplacian(Y_exact, x_test).detach().cpu().numpy().reshape(n_plot, n_plot)
        f = f.detach().cpu().numpy().reshape(n_plot, n_plot)

    with torch.no_grad():
        y_exa = y_exact(x_test)
        u_exa =  ratio * y_exa
        xi_exa = ksi(x_test)

        filename = f"data/Direct_MPEC.pth"
        torch.save({
            'x_1': X,
            'x_2': Y,
            'U_pred': u_pred.cpu().detach(),
            'U_exa': u_exa.cpu().detach(),
            'Y_pred': y_pred.cpu().detach(),
            'Y_exa': y_exa.cpu().detach(),
            'Xi_pred':N_xi.cpu().detach(),
            "Xi_exa": xi_exa.cpu().detach(),
        }, filename)

        y_pred_cpu = y_pred.cpu().detach().numpy().reshape(n_plot, n_plot)
        u_pred_cpu = u_pred.cpu().detach().numpy().reshape(n_plot, n_plot)
        u_exa_cpu = u_exa.cpu().detach().numpy().reshape(n_plot, n_plot)
        y_exa_cpu = y_exa.cpu().detach().numpy().reshape(n_plot, n_plot)
        xi_pred_cpu =N_xi.cpu().detach().numpy().reshape(n_plot, n_plot)
        xi_exa_cpu = xi_exa.cpu().detach().numpy().reshape(n_plot, n_plot)


        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()

        fig = plt.figure(figsize=(16, 16))

        ax1 = fig.add_subplot(2, 2, 2, projection='3d')
        surf1 = ax1.plot_surface(X, Y, f, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_zlabel('Control Value')
        ax1.set_title('(b) f', fontweight='bold')
        ax1.view_init(elev=30, azim=250)
        ax1.grid(True, alpha=0.3)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)


        ax2 = fig.add_subplot(2, 2, 1, projection='3d')
        surf2 = ax2.plot_surface(X, Y, LapY_exact, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('State Value')
        ax2.set_title('(a) Exact y', fontweight='bold')
        ax2.view_init(elev=30, azim=250)
        ax2.grid(True, alpha=0.3)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        ax1 = fig.add_subplot(2, 2, 3, projection='3d')
        surf1 = ax1.plot_surface(X, Y, xi_exa_cpu, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_zlabel('Control Value')
        ax1.set_title('(c) Exact xi', fontweight='bold')
        ax1.view_init(elev=30, azim=250)
        ax1.grid(True, alpha=0.3)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # 第二个视角：强调细节特征
        ax2 = fig.add_subplot(2, 2, 4, projection='3d')
        surf2 = ax2.plot_surface(X, Y, xi_pred_cpu, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('State Value')
        ax2.set_title('(a) Prediction xi', fontweight='bold')
        ax2.view_init(elev=30, azim=250)
        ax2.grid(True, alpha=0.3)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        plt.savefig("data/Example_5.4.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_pred_cpu, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_zlabel('Control Value')
        ax1.set_title('(b) Predicted u', fontweight='bold')
        ax1.view_init(elev=30, azim=250)
        ax1.grid(True, alpha=0.3)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # 第二个视角：强调细节特征
        ax2 = fig.add_subplot(1, 2, 1, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u_exa_cpu, cmap='viridis',
                                 rstride=2, cstride=2, alpha=0.9)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('State Value')
        ax2.set_title('(a) Exact u', fontweight='bold')
        ax2.view_init(elev=30, azim=250)
        ax2.grid(True, alpha=0.3)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        plt.savefig("data/MPEC_control.png", dpi=150, bbox_inches='tight')
        plt.close()

        return y_pred, y_exact, u_pred, u_exact, x_test

if __name__ == "__main__":
    main()