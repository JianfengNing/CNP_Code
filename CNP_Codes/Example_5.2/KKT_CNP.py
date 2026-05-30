import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Parameters
lambda_val = 0.01
d = 10  # dimension
rate = 0
c_1 = 0.5 * ((4 / torch.pi) ** d)


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
        x = x * boundary_fun(x_original)
        return x







def exact_control(x):
    return d * ((torch.pi/2) ** 2) * torch.prod(torch.cos(0.5*torch.pi * x), dim=1, keepdim=True) + rate * d * (torch.pi ** 2) * torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True)


def exact_state(x):
    return (torch.prod(torch.cos(0.5*torch.pi * x), dim=1, keepdim=True) + rate * torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True))


def target_state(x):
    index = x[:, 0:1] < 0
    yd = (lambda_val * (d ** 2) * ((torch.pi/2) ** 4) + 1) * torch.prod(torch.cos(0.5*torch.pi * x), dim=1, keepdim=True) + rate*(lambda_val * (d ** 2) * (torch.pi ** 4) + 1) * torch.prod(torch.sin(torch.pi * x), dim=1, keepdim=True)
    yd[index] = yd[index] + 0.5
    return yd


def basis(x, d):
    return  2 * ((3/4)**d)*boundary_fun(x)

def boundary_fun(x):
    f1 = ((1-x[:,0:1]**2) * (1-x[:,1:2]**2) * (1-x[:,2:3]**2) * (1-x[:,3:4]**2)
           * (1-x[:,4:5]**2) * (1-x[:,5:6]**2) * (1-x[:,6:7]**2)*(1-x[:,7:8]**2)*(1-x[:,8:9]**2)*(1-x[:,9:10]**2))
    return f1


def generate_data(n_samples=10000):
    x = torch.rand(n_samples, d, device=device) * 2-1
    return x



def laplacian_y_theta(y, x):
    N_grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                 create_graph=True, retain_graph=True)[0]

    laplacian_N = torch.zeros_like(y)
    for i in range(d):
        N_grad2 = torch.autograd.grad(N_grad[:, i], x, grad_outputs=torch.ones_like(N_grad[:, i]),
                                      create_graph=True, retain_graph=True)[0][:, i:i + 1]
        laplacian_N += N_grad2

    return laplacian_N



def loss_function(y_pred, u_pred, y_d):
    data_loss = 0.5 * torch.mean((y_pred - y_d) ** 2)
    reg_loss = 0.5 * lambda_val * torch.mean(u_pred ** 2)
    total_loss = data_loss + reg_loss
    return total_loss, data_loss, reg_loss



def train_model(n_samples=10000, epochs=5000, lr=1e-3):
    model_y = ResNetFunction(
        input_dim=d,
        output_dim=1,
        hidden_dim=128,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    model_p = ResNetFunction(
        input_dim=d,
        output_dim=1,
        hidden_dim=128,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    lambda_param = torch.tensor(1.0, requires_grad=True)
    multiplier = torch.tensor(1.0, requires_grad=True)

    optimizer = torch.optim.Adam([
        {'params': model_y.parameters(), 'lr': 1e-3},
        {'params': model_p.parameters(), 'lr': 1e-3},
        {'params': lambda_param, 'lr': 1e-2},
        {'params': multiplier, 'lr': 1e-2},
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    train_losses = []
    print("Starting training...")
    print(f"Training data points: {n_samples}")

    for epoch in range(epochs):
        model_y.train()
        model_p.train()

        x_train = generate_data(n_samples)
        x_train_2 = generate_data(n_samples)
        x_train_2[:, 0:1] = 0.5 * x_train_2[:, 0:1] - 0.5
        x_train.requires_grad_(True)

        y_d_train = target_state(x_train).detach()

        y_net = model_y(x_train)

        integral_net = (2 ** (d - 1)) * torch.mean(model_y(x_train_2))

        basis_val = basis(x_train, d)
        y_pred = y_net - (integral_net - c_1 + lambda_param ** 2) * basis_val

        lap_y = laplacian_y_theta(y_pred, x_train)

        u_pred = -(1/lambda_val)*model_p(x_train)



        pde_loss = torch.mean((-lap_y-u_pred)**2)


        adjoint_residue = -lambda_val * laplacian_y_theta(u_pred, x_train) + y_pred - y_d_train + torch.where(x_train[:,0:1]<0,1,0)*(multiplier ** 2)
        adjoint_loss = torch.mean(adjoint_residue**2)

        multipler_loss = (multiplier* lambda_param)** 2

        total_loss = (pde_loss + adjoint_loss + 1.0*multipler_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(total_loss.item())


        if epoch % 100 == 0:
            integral =  (2 ** (d-1))*torch.mean(y_pred)
            print('integral', integral, 'c', c_1)
            print('lamda_para', lambda_param ** 2)
            print(f"multiplier: {multiplier**2:.6f}")
            u = exact_control(x_train)
            y = exact_state(x_train)
            control_error = torch.sqrt(torch.mean((u_pred - u) ** 2)) / torch.sqrt(
                torch.mean(u ** 2))
            state_error = torch.sqrt(torch.mean((y_pred - y) ** 2)) / torch.sqrt(
                torch.mean(y ** 2))
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item():.6e}, "
                  f"PDE Loss: {pde_loss.item():.6f}, Adjoint Loss: {adjoint_loss.item():.6f}, "
                  f"Control Error: {control_error.item():.6f}")
            print(f"State Error: {state_error.item():.6f}")

    return model_y, model_p, lambda_param



def evaluate_model(model_y,model_p, lambda_param):
    model_y.eval()
    model_p.eval()

    num_points = 256

    x1 = torch.linspace(-1, 1, num_points)
    x2 = torch.linspace(-1, 1, num_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')


    x_test_list = []
    for i in range(X1.numel()):
        point = torch.ones(d) * 0.0
        point[0] = X1.flatten()[i]
        point[1] = X2.flatten()[i]
        x_test_list.append(point)

    x_test = torch.stack(x_test_list).to(device)
    x_test.requires_grad_(True)
    x_int = generate_data(40000)
    x_int[:, 0:1] = 0.5 * x_int[:, 0:1] - 0.5

    y_net = model_y(x_test)
    y_int = model_y(x_int)

    integral_net = (2 ** (d - 1)) * torch.mean(y_int)

    basis_val = basis(x_test, d)
    y_pred = (y_net - (integral_net - c_1 + lambda_param ** 2) * basis_val).cpu().detach()


    u_pred = -(1/lambda_val)*model_p(x_test).cpu().detach()

    u_exact = exact_control(x_test).cpu().detach()
    y_exact = exact_state(x_test).cpu().detach()

    X1_np = X1.detach().numpy()
    X2_np = X2.detach().numpy()

    filename = f"data/KKT_CNP_10D.pth"
    torch.save({
        'x_1': X1.detach(),
        'x_2': X2.detach(),
        'U_pred': u_pred.reshape(num_points, num_points),
        'U_exa': u_exact.reshape(num_points, num_points),
        'Y_pred': y_pred.reshape(num_points, num_points),
        'Y_exa': y_exact.reshape(num_points, num_points),
    }, filename)

    fig = plt.figure(figsize=(16, 16))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    surf1 = ax1.plot_surface(X1_np, X2_np, u_exact.numpy().reshape(num_points, num_points), cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    #ax1.set_zlabel()
    ax1.set_title('(a) Exact Control', fontweight='bold')
    ax1.view_init(elev=35, azim=250)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X1_np, X2_np, u_pred.numpy().reshape(num_points, num_points), cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    #ax2.set_zlabel()
    ax2.set_title('(b) Computed Control', fontweight='bold')
    ax2.view_init(elev=35, azim=250)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(X1_np, X2_np, y_exact.numpy().reshape(num_points, num_points), cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    # ax3.set_zlabel()
    ax3.set_title('(c) Exact State', fontweight='bold')
    ax3.view_init(elev=35, azim=250)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(X1_np, X2_np, u_pred.numpy().reshape(num_points, num_points), cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    # ax4.set_zlabel()
    ax4.set_title('(d) Computed State', fontweight='bold')
    ax4.view_init(elev=35, azim=250)
    ax4.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig("data/KKT_CNP_figure.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



if __name__ == "__main__":

    if not torch.backends.mps.is_available():
        print("Warning: MPS not available, using CPU")
    else:
        print("MPS available, using Mac GPU")

    model_y, model_p, lambda_param = train_model(
        n_samples=30000, epochs=10000, lr=1e-3
    )

    y_pred, y_exact, u_pred, u_exact, x_test = evaluate_model(model_y,model_p,lambda_param)
