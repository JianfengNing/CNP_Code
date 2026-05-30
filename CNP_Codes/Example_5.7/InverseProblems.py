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

# Set random seeds
torch.manual_seed(42)
if device.type == 'mps':
    torch.mps.manual_seed(42)
np.random.seed(45)

d=2

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

        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.Sequential(
            *[FCResBlock(hidden_dim, self.activation) for _ in range(num_layers)]
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.blocks(x)
        x = self.output_layer(x)
        return x



def boundary_function(x):
    return (1-x[:,0:1]**2) * (1-x[:,1:2]**2)




def exact_u(x):
    term1 = torch.exp((x[:,0:1]**2+x[:,1:2]**2)/torch.pi) * torch.sin(torch.pi * x[:,0:1]) * torch.sin(torch.pi * x[:,1:2])
    bd = Dirich_boundary(x)
    return term1 + bd



def exact_c(x):
    c1 = 1 +  2.0*torch.prod(torch.sin(torch.pi * x), dim=-1, keepdim=True)-0.2
    c2 = c1 - torch.relu(c1-2.0)
    c3 = c2 + torch.relu(1.0-c2)
    return c3



def f(x):
    x.requires_grad_(True)
    u = exact_u(x)
    lap_u = laplacian(u,x)
    return -0.1*lap_u+exact_c(x)*exact_u(x)






def generate_data(n_samples=10000):
    n_plot = n_samples
    x_plot = torch.linspace(-1, 1, n_plot, device=device)
    y_plot = torch.linspace(-1, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return x

def Dirich_boundary(x):
    return torch.sin(x[:,0:1] + x[:,1:2]) * 0 + 1


def neumann_boundary_loss(model_u, x_0,x_1,y_0,y_1,neuman_meas):

    u_x_0 = model_u(x_0) * boundary_function(x_0)+ Dirich_boundary(x_0)
    u_x_0_x = torch.autograd.grad(u_x_0, x_0, torch.ones_like(u_x_0), create_graph=True)[0][:, 0:1]

    u_x_1 = model_u(x_1) * boundary_function(x_1)+ Dirich_boundary(x_1)
    u_x_1_x = torch.autograd.grad(u_x_1, x_1, torch.ones_like(u_x_1), create_graph=True)[0][:, 0:1]

    u_y_0 = model_u(y_0) * boundary_function(y_0)+ Dirich_boundary(y_0)
    u_y_0_y = torch.autograd.grad(u_y_0, y_0, torch.ones_like(u_y_0), create_graph=True)[0][:, 1:2]

    u_y_1 = model_u(y_1) * boundary_function(y_1)+ Dirich_boundary(y_1)
    u_y_1_y = torch.autograd.grad(u_y_1, y_1, torch.ones_like(u_y_1), create_graph=True)[0][:, 1:2]

    loss = (torch.mean((u_x_0_x-neuman_meas[0])**2) + torch.mean((u_x_1_x-neuman_meas[1])**2)
            + torch.mean((u_y_0_y-neuman_meas[2])**2) + torch.mean((u_y_1_y-neuman_meas[3])**2))

    return 0.25*loss




def laplacian(y, x):
    grads = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]   # (N,2)
    y_x = grads[:, 0:1]
    y_y = grads[:, 1:2]
    y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0][:, 0:1]
    y_yy = torch.autograd.grad(y_y, x, torch.ones_like(y_y), create_graph=True)[0][:, 1:2]
    return y_xx + y_yy





def train_model(n_samples=10000, epochs=5000, lr=1e-3):

    model_u = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    model_c = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model_u.parameters(), 'lr': lr},
        {'params': model_c.parameters(), 'lr': lr},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    x_train = generate_data(n_samples).to(device)
    x_train.requires_grad_(True)
    noise_level = 0.005


    measurement = exact_c(x_train) * exact_u(x_train)
    measurement = (measurement + torch.randn_like(measurement) * measurement * noise_level).detach()

    n_plot = 128
    zero_plot = torch.linspace(-1, 1, n_plot, device=device) * 0-1.0
    one_plot = zero_plot +  2.0
    line_plot = torch.linspace(-1, 1, n_plot, device=device)

    X_0 = torch.stack([zero_plot.reshape(-1), line_plot.reshape(-1)], dim=1)
    X_1 = torch.stack([one_plot.reshape(-1), line_plot.reshape(-1)], dim=1)
    Y_0 = torch.stack([line_plot.reshape(-1), zero_plot.reshape(-1)], dim=1)
    Y_1 = torch.stack([line_plot.reshape(-1), one_plot.reshape(-1)], dim=1)



    Measure_Neu = []

    X_0.requires_grad_(True)
    X_1.requires_grad_(True)
    Y_0.requires_grad_(True)
    Y_1.requires_grad_(True)

    u_x_0 =exact_u(X_0)
    meaure_neuman_X_0 = torch.autograd.grad(u_x_0, X_0, torch.ones_like(u_x_0), create_graph=True)[0][:, 0:1]
    u_x_1 = exact_u(X_1)
    meaure_neuman_X_1 = torch.autograd.grad(u_x_1, X_1, torch.ones_like(u_x_1), create_graph=True)[0][:, 0:1]
    u_y_0 = exact_u(Y_0)
    meaure_neuman_Y_0 = torch.autograd.grad(u_y_0, Y_0, torch.ones_like(u_y_0), create_graph=True)[0][:, 1:2]
    u_y_1 = exact_u(Y_1)
    meaure_neuman_Y_1 = torch.autograd.grad(u_y_1, Y_1, torch.ones_like(u_y_1), create_graph=True)[0][:, 1:2]


    meaure_neuman_X_0 = (meaure_neuman_X_0 + torch.randn_like(meaure_neuman_X_0) * meaure_neuman_X_0 * noise_level).detach()
    Measure_Neu.append(meaure_neuman_X_0)
    meaure_neuman_X_1 = (meaure_neuman_X_1 + torch.randn_like(meaure_neuman_X_1) * meaure_neuman_X_1 * noise_level).detach()
    Measure_Neu.append(meaure_neuman_X_1)
    meaure_neuman_Y_0 = (meaure_neuman_Y_0 + torch.randn_like(meaure_neuman_Y_0) * meaure_neuman_Y_0 * noise_level).detach()
    Measure_Neu.append(meaure_neuman_Y_0)
    meaure_neuman_Y_1 = (meaure_neuman_Y_1 + torch.randn_like(meaure_neuman_Y_1) * meaure_neuman_Y_1 * noise_level).detach()
    Measure_Neu.append(meaure_neuman_Y_1)


    print("Starting training...")
    print(f"Training data points: {n_samples}")

    for epoch in range(epochs):

        model_u.train()
        model_c.train()


        x_train.requires_grad_(True)


        X_0.requires_grad_(True)
        X_1.requires_grad_(True)
        Y_0.requires_grad_(True)
        Y_1.requires_grad_(True)



        u_pred = model_u(x_train) * boundary_function(x_train) + Dirich_boundary(x_train)
        c_pred = 1+torch.sin(model_c(x_train) * boundary_function(x_train))**2


        data_loss = torch.mean((u_pred * c_pred - measurement) ** 2)
        loss_neumann = neumann_boundary_loss(model_u,X_0,X_1,Y_0,Y_1, Measure_Neu)

        if epoch % 500 == 0:
            f_pred = -0.1 * laplacian(u_pred, x_train) + c_pred * u_pred

        total_loss = data_loss + loss_neumann


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()



        if epoch % 500 == 0:
            u_exact = exact_u(x_train)
            c_exact = exact_c(x_train)
            f_exact = f(x_train)

            error_u = (torch.mean((u_exact - u_pred) ** 2) / torch.mean(u_exact ** 2)) ** 0.5
            error_c = (torch.mean((c_exact - c_pred) ** 2) / torch.mean(c_exact ** 2)) ** 0.5
            error_f = (torch.mean((f_exact - f_pred) ** 2) / torch.mean(f_exact ** 2)) ** 0.5
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item():2e}, "
                  f"TotalLoss: {total_loss.item():.6f}, loss_neumann: {loss_neumann.item():.6f},"
                  f"Error_u: {error_u.item():.6f}, Error_c: {error_c.item():.6f},Error_f: {error_f.item():.6f},")


            evaluate_model(model_u, model_c)


    return model_y,model_p,model_u, train_losses, control_errors




def evaluate_model(model_u,model_c):
    model_u.eval()
    model_c.eval()

    n_plot = 256
    dd =1.0
    x_plot = torch.linspace(-dd, dd, n_plot, device=device)
    y_plot = torch.linspace(-dd, dd, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    x_test = torch.tensor(x, requires_grad=True)
    f_exact = f(x_test)

    with torch.enable_grad():
        u_pred = model_u(x_test) * boundary_function(x_test) + Dirich_boundary(x_test)
        c_pred = 1 + torch.sin(model_c(x_test)*boundary_function(x_test))**2
        f_pred = -0.1 * laplacian(u_pred, x_test) + c_pred * u_pred


    with torch.no_grad():
        u_exact = exact_u(x_test)
        c_exact = exact_c(x_test)

        f_error = torch.sqrt(torch.mean((f_pred - f_exact) ** 2)) / torch.sqrt(torch.mean(f_exact ** 2))
        c_error = torch.sqrt(torch.mean((c_pred - c_exact) ** 2)) / torch.sqrt(torch.mean(c_exact ** 2))

        filename = f"data/hybrid_inverse_problem.pth"
        torch.save({
            'x_1': X.detach().cpu(),
            'x_2': Y.detach().cpu(),
            'C_pred': c_pred.reshape(256, 256).detach().cpu(),
            'C_exa': c_exact.reshape(256, 256).detach().cpu(),
            'F_pred': f_pred.reshape(256, 256).detach().cpu(),
            'F_exa': f_exact.reshape(256, 256).detach().cpu(),

        }, filename)

        f_exact = f_exact.detach().cpu().numpy().reshape(n_plot, n_plot)
        c_exact = c_exact.detach().cpu().numpy().reshape(n_plot, n_plot)
        f_pred = f_pred.detach().cpu().numpy().reshape(n_plot, n_plot)
        c_pred = c_pred.detach().cpu().numpy().reshape(n_plot, n_plot)
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()

        print(f"f L2 Error: {f_error.item():.6e}")
        print(f"sigma L2 Error: {c_error.item():.6e}")

    fig = plt.figure(figsize=(12, 12))


    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, f_exact, cmap='inferno',
                     rstride=2, cstride=2, alpha=0.9)


    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('(a) exact f', fontweight='bold')
    ax1.view_init(elev=40, azim=250)
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)


    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, f_pred, cmap='inferno',
                     rstride=2, cstride=2, alpha=0.9)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('(b) pred f', fontweight='bold')
    ax2.view_init(elev=40, azim=250)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    ax1 = fig.add_subplot(2, 2, 3, projection='3d')
    ax1.plot_surface(X, Y, c_exact, cmap='inferno',
                     rstride=2, cstride=2, alpha=0.9)


    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('(a) exact c', fontweight='bold')
    ax1.view_init(elev=40, azim=250)
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)


    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    ax2.plot_surface(X, Y, c_pred, cmap='inferno',
                     rstride=2, cstride=2, alpha=0.9)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('(b) pred c', fontweight='bold')
    ax2.view_init(elev=40, azim=250)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()

    plt.savefig("data/inverse_problem.png", dpi=200, bbox_inches='tight')
    plt.close()
    print('figure saved')





def plot_results(train_losses, control_errors, y_pred, y_exact, u_pred, u_exact,x_test):


    norms = torch.norm(x_test, p=2, dim=1)

    mask = norms > 1
    u_pred[mask] = torch.nan
    y_exact[mask] = torch.nan
    u_exact[mask] = torch.nan
    y_pred[mask] = torch.nan

    n_plot = 256
    x_plot = torch.linspace(-1, 1, n_plot)
    y_plot = torch.linspace(-1, 1, n_plot)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    X = X.numpy()
    Y = Y.numpy()



    x_test = x_test.cpu().detach().numpy()
    u_pred_cpu = u_pred.cpu().detach().numpy().reshape(n_plot, n_plot)
    u_exact_cpu = u_exact.cpu().numpy().reshape(n_plot, n_plot)


    fig = plt.figure(figsize=(16, 6))



    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    print(x_test[:,0].shape,u_exact_cpu.shape)
    ax1.plot_surface(X, Y, u_pred_cpu, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)


    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('(b) Numerical Optimal Control', fontweight='bold')
    ax1.view_init(elev=40, azim=250)
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2.plot_surface(X, Y, u_exact_cpu, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('(a) Exact Optimal Control', fontweight='bold')
    ax2.view_init(elev=40, azim=250)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()

    plt.savefig("data/PINN-Prejection_Improved.png", dpi=200, bbox_inches='tight')
    plt.close()
    print('figure saved')




if __name__ == "__main__":

    if not torch.backends.mps.is_available():
        print("Warning: MPS not available, using CPU")
    else:
        print("MPS available, using Mac GPU")

    model_y,model_p,model_u, train_losses, control_errors = train_model(
        n_samples=128, epochs=20000, lr=1e-3
    )

    y_pred, y_exact, u_pred, u_exact, x_test = evaluate_model(model_y,model_p,model_u)

    filename = f"data/PINN_Proj_rate{rate}.pth"
    torch.save({
        'control_error': control_errors,
        'u_pred': u_pred,
        'u_exact': u_exact,
        'y_pred': y_pred,
        'y_exact': y_exact
    }, filename)

    # Plot results
    plot_results(train_losses, control_errors, y_pred, y_exact, u_pred, u_exact)