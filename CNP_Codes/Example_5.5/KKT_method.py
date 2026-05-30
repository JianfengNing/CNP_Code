import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
pi = math.pi
import torch.autograd as autograd
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


d=2
mu = torch.tensor(0.5)
lambda_val = 0.1
frequency_u = 8.0

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




def boundary_function(x):
    return  torch.sin(torch.pi * x[:,0:1]) * torch.sin(torch.pi * x[:,1:2])

def exact_y(x):
    y1 = torch.exp(-0.05*mu)*(torch.sin(pi*x[:,0:1])**2)*torch.sin(pi*x[:,1:2])*torch.cos(pi*x[:,1:2])
    y2 =  -torch.exp(-0.05*mu)*(torch.sin(pi*x[:,1:2])**2)*torch.sin(pi*x[:,0:1])*torch.cos(pi*x[:,0:1])
    return y1,y2

def exact_control(x):
    u1 = (torch.exp(-0.05*mu)-torch.exp(-mu))*(torch.sin(frequency_u*pi*x[:,0:1])**2)*torch.sin(frequency_u*pi*x[:,1:2])*torch.cos(frequency_u*pi*x[:,1:2])
    u2 = -(torch.exp(-0.05*mu)-torch.exp(-mu))*(torch.sin(frequency_u*pi*x[:,1:2])**2)*torch.sin(frequency_u*pi*x[:,0:1])*torch.cos(frequency_u*pi*x[:,0:1])
    return -u1/lambda_val, -u2/lambda_val


def f(x):

    x = x.clone().requires_grad_(True)

    y1, y2 = exact_y(x)

    grad_y1 = torch.autograd.grad(y1.sum(), x, create_graph=True)[0]
    grad_y1_x = grad_y1[:, 0:1]
    grad_y1_y = grad_y1[:, 1:2]

    grad_y2 = torch.autograd.grad(y2.sum(), x, create_graph=True)[0]
    grad_y2_x = grad_y2[:, 0:1]
    grad_y2_y = grad_y2[:, 1:2]

    laplace_y1_x = torch.autograd.grad(grad_y1_x.sum(), x, create_graph=True)[0][:, 0:1]
    laplace_y1_y = torch.autograd.grad(grad_y1_y.sum(), x, create_graph=True)[0][:, 1:2]
    laplace_y1 = laplace_y1_x + laplace_y1_y

    laplace_y2_x = torch.autograd.grad(grad_y2_x.sum(), x, create_graph=True)[0][:, 0:1]
    laplace_y2_y = torch.autograd.grad(grad_y2_y.sum(), x, create_graph=True)[0][:, 1:2]
    laplace_y2 = laplace_y2_x + laplace_y2_y

    conv1 = y1 * grad_y1_x + y2 * grad_y1_y
    conv2 = y1 * grad_y2_x + y2 * grad_y2_y

    u1, u2 = exact_control(x)

    f1 = -mu * laplace_y1 + conv1 - u1
    f2 = -mu * laplace_y2 + conv2 - u2

    return f1.detach(), f2.detach()




def yd(x):

    x = x.clone().requires_grad_(True)
    y1, y2 = exact_y(x)
    u1, u2 = exact_control(x)

    lambda1 = -u1 * lambda_val
    lambda2 = -u2 * lambda_val

    grad_y1 = torch.autograd.grad(y1.sum(), x, create_graph=True)[0]
    grad_y2 = torch.autograd.grad(y2.sum(), x, create_graph=True)[0]

    grad_lambda1 = torch.autograd.grad(lambda1.sum(), x, create_graph=True)[0]
    grad_lambda2 = torch.autograd.grad(lambda2.sum(), x, create_graph=True)[0]

    grad_lambda1_x = grad_lambda1[:, 0:1]
    grad_lambda1_y = grad_lambda1[:, 1:2]

    laplace_lambda1_x = torch.autograd.grad(grad_lambda1_x.sum(), x, create_graph=True)[0][:, 0:1]
    laplace_lambda1_y = torch.autograd.grad(grad_lambda1_y.sum(), x, create_graph=True)[0][:, 1:2]
    laplace_lambda1 = laplace_lambda1_x + laplace_lambda1_y

    grad_lambda2_x = grad_lambda2[:, 0:1]
    grad_lambda2_y = grad_lambda2[:, 1:2]

    laplace_lambda2_x = torch.autograd.grad(grad_lambda2_x.sum(), x, create_graph=True)[0][:, 0:1]
    laplace_lambda2_y = torch.autograd.grad(grad_lambda2_y.sum(), x, create_graph=True)[0][:, 1:2]
    laplace_lambda2 = laplace_lambda2_x + laplace_lambda2_y

    grad_y_T_lambda1 = grad_y1[:, 0:1] * lambda1 + grad_y2[:, 0:1] * lambda2
    grad_y_T_lambda2 = grad_y1[:, 1:2] * lambda1 + grad_y2[:, 1:2] * lambda2

    y_dot_grad_lambda1 = y1 * grad_lambda1_x + y2 * grad_lambda1_y
    y_dot_grad_lambda2 = y1 * grad_lambda2_x + y2 * grad_lambda2_y

    yd1 = y1 + mu * laplace_lambda1 + y_dot_grad_lambda1 - grad_y_T_lambda1
    yd2 = y2 + mu * laplace_lambda2 + y_dot_grad_lambda2 - grad_y_T_lambda2

    return yd1.detach(), yd2.detach()




def generate_data(n_samples=10000):
    n_plot = 50+random.randint(1, 50)
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return x





def compute_u(x_train,model_psi,model_p,f_train,mu):
    psi = model_psi(x_train)*(boundary_function(x_train)**2)
    grads_psi = torch.autograd.grad(psi, x_train, torch.ones_like(psi), create_graph=True)[0]
    y1 = grads_psi[:,1:2]
    y2 = -grads_psi[:,0:1]


    grads1 = torch.autograd.grad(y1, x_train, torch.ones_like(y1), create_graph=True)[0]  # (N,2)
    y1_x = grads1[:, 0:1]
    y1_y = grads1[:, 1:2]
    y1_xx = torch.autograd.grad(y1_x, x_train, torch.ones_like(y1_x), create_graph=True)[0][:, 0:1]
    y1_yy = torch.autograd.grad(y1_y, x_train, torch.ones_like(y1_y), create_graph=True)[0][:, 1:2]

    grads2 = torch.autograd.grad(y2, x_train, torch.ones_like(y2), create_graph=True)[0]  # (N,2)
    y2_x = grads2[:, 0:1]
    y2_y = grads2[:, 1:2]
    y2_xx = torch.autograd.grad(y2_x, x_train, torch.ones_like(y2_x), create_graph=True)[0][:, 0:1]
    y2_yy = torch.autograd.grad(y2_y, x_train, torch.ones_like(y2_y), create_graph=True)[0][:, 1:2]

    p = model_p(x_train)
    grads_p = torch.autograd.grad(p, x_train, torch.ones_like(p), create_graph=True)[0]  # (N,2)
    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]


    f1 = f_train[0]
    f2 = f_train[1]
    u1 = -mu*(y1_xx + y1_yy) + (y1 * y1_x + y2 * y1_y) + p_x -f1
    u2 = -mu*(y2_xx + y2_yy) + (y1 * y2_x + y2 * y2_y) + p_y -f2
    u_pred = torch.cat((u1, u2), dim=1)
    y_pred = torch.cat((y1, y2), dim=1)

    return  u_pred,y_pred




def laplacian(y, x):
    grads = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]   # (N,2)
    y_x = grads[:, 0:1]
    y_y = grads[:, 1:2]
    y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0][:, 0:1]
    y_yy = torch.autograd.grad(y_y, x, torch.ones_like(y_y), create_graph=True)[0][:, 1:2]
    return y_xx + y_yy



def loss_function(x_train,model_y,model_p,model_w,model_e,f_train,yd_train, mu,lambda_val):
    psi = model_y(x_train) * (boundary_function(x_train) ** 2)
    grads_psi = torch.autograd.grad(psi, x_train, torch.ones_like(psi), create_graph=True)[0]
    y1 = grads_psi[:, 1:2]
    y2 = -grads_psi[:, 0:1]

    grads1 = torch.autograd.grad(y1, x_train, torch.ones_like(y1), create_graph=True)[0]  # (N,2)
    y1_x = grads1[:, 0:1]
    y1_y = grads1[:, 1:2]
    y1_xx = torch.autograd.grad(y1_x, x_train, torch.ones_like(y1_x), create_graph=True)[0][:, 0:1]
    y1_yy = torch.autograd.grad(y1_y, x_train, torch.ones_like(y1_y), create_graph=True)[0][:, 1:2]

    grads2 = torch.autograd.grad(y2, x_train, torch.ones_like(y2), create_graph=True)[0]  # (N,2)
    y2_x = grads2[:, 0:1]
    y2_y = grads2[:, 1:2]
    y2_xx = torch.autograd.grad(y2_x, x_train, torch.ones_like(y2_x), create_graph=True)[0][:, 0:1]
    y2_yy = torch.autograd.grad(y2_y, x_train, torch.ones_like(y2_y), create_graph=True)[0][:, 1:2]

    p = model_p(x_train)
    grads_p = torch.autograd.grad(p, x_train, torch.ones_like(p), create_graph=True)[0]  # (N,2)
    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]

    phi = model_w(x_train) * (boundary_function(x_train) ** 2)
    grads_phi = torch.autograd.grad(phi, x_train, torch.ones_like(phi), create_graph=True)[0]
    w1 = grads_phi[:, 1:2]
    w2 = -grads_phi[:, 0:1]

    grads1w = torch.autograd.grad(w1, x_train, torch.ones_like(w1), create_graph=True)[0]  # (N,2)
    w1_x = grads1w[:, 0:1]
    w1_y = grads1w[:, 1:2]
    w1_xx = torch.autograd.grad(w1_x, x_train, torch.ones_like(w1_x), create_graph=True)[0][:, 0:1]
    w1_yy = torch.autograd.grad(w1_y, x_train, torch.ones_like(w1_y), create_graph=True)[0][:, 1:2]

    grads2w = torch.autograd.grad(w2, x_train, torch.ones_like(w2), create_graph=True)[0]  # (N,2)
    w2_x = grads2w[:, 0:1]
    w2_y = grads2w[:, 1:2]
    w2_xx = torch.autograd.grad(w2_x, x_train, torch.ones_like(w2_x), create_graph=True)[0][:, 0:1]
    w2_yy = torch.autograd.grad(w2_y, x_train, torch.ones_like(w2_y), create_graph=True)[0][:, 1:2]

    e = model_e(x_train)
    grads_e = torch.autograd.grad(e, x_train, torch.ones_like(e), create_graph=True)[0]  # (N,2)
    e_x = grads_e[:, 0:1]
    e_y = grads_e[:, 1:2]

    f1 = f_train[0]
    f2 = f_train[1]
    yd1 = yd_train[0]
    yd2 = yd_train[1]

    state_residue1 =  -mu * (y1_xx + y1_yy) + (y1 * y1_x + y2 * y1_y) + p_x -f1 - (-1/lambda_val)*w1
    state_residue2 =  -mu * (y2_xx + y2_yy) + (y1 * y2_x + y2 * y2_y) + p_y - f2 - (-1/lambda_val)*w2

    adjoint_residue1 = mu * (w1_xx + w1_yy) + (y1 * w1_x + y2 * w1_y) - (y1_x * w1 + y2_x * w2) - e_x + y1 - yd1
    adjoint_residue2 = mu * (w2_xx + w2_yy) + (y1 * w2_x + y2 * w2_y) - (y1_y * w1 + y2_y * w2) - e_y + y2 - yd2

    loss_state = torch.mean(state_residue1**2+state_residue2**2)
    loss_adjoint = torch.mean(adjoint_residue1**2+adjoint_residue2**2)
    loss_total = loss_state + loss_adjoint

    u_pred = torch.cat(((-1/lambda_val)*w1,(-1/lambda_val)*w2),dim=1)
    y_pred = torch.cat((y1,y2),dim=1)

    return loss_total, loss_state, loss_adjoint,u_pred,y_pred


def train_model(n_samples=10000, epochs=5000, lr=1e-3):
    model_y = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    model_p = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    model_w = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)
    model_e = ResNetFunction(
        input_dim=2,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        activation_fn=nn.Tanh,
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model_y.parameters(), 'lr': lr},
        {'params': model_p.parameters(), 'lr': lr},
        {'params': model_w.parameters(), 'lr': lr},
        {'params': model_e.parameters(), 'lr': lr}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)


    train_losses = []

    control_errors = []

    print("Starting training...")
    print(f"Training data points: {n_samples}")

    for epoch in range(epochs):
        model_y.train()
        model_p.train()
        model_w.train()
        model_e.train()

        x_train = generate_data(n_samples)
        x_train.requires_grad_(True)

        y_d_train = yd(x_train)
        f_train = f(x_train)


        total_loss, loss_state, loss_adjoint,u_pred,y_pred = loss_function(x_train,model_y,model_p,model_w,model_e,f_train,y_d_train,mu,lambda_val)


        u_exact_train = exact_control(x_train)
        u_exact_train = torch.cat((u_exact_train[0],u_exact_train[1]),dim=1)
        control_error = torch.sqrt(torch.mean((u_pred - u_exact_train) ** 2)) / torch.sqrt(
            torch.mean(u_exact_train ** 2))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(total_loss.item())
        control_errors.append(control_error.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d}, Total Loss: {total_loss.item()}, "
                  f"state loss: {loss_state.item():.6f}, adjoint_loss: {loss_adjoint.item():.6f}, "
                  f"Control Error: {control_error.item():.6f}")

            y_pred, y_exact, u_pred, u_exact, x_test = evaluate_model(model_y,model_w)

            # Plot results
            plot_results(train_losses, control_errors, y_pred, y_exact, u_pred, u_exact)

    return model_y,model_w, train_losses, control_errors


def evaluate_model(model_y,model_w, n_test=5000):
    model_y.eval()
    model_w.eval()

    n_plot = 256
    x_plot = torch.linspace(0, 1, n_plot, device=device)
    y_plot = torch.linspace(0, 1, n_plot, device=device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    x_test = torch.tensor(x, requires_grad=True)

    with torch.enable_grad():
        psi = model_y(x_test) * (boundary_function(x_test) ** 2)
        grads_psi = torch.autograd.grad(psi, x_test, torch.ones_like(psi), create_graph=True)[0]
        y1 = grads_psi[:, 1:2]
        y2 = -grads_psi[:, 0:1]
        phi = model_w(x_test) * (boundary_function(x_test) ** 2)
        grads_phi = torch.autograd.grad(phi, x_test, torch.ones_like(phi), create_graph=True)[0]
        w1 = grads_phi[:, 1:2]
        w2 = -grads_phi[:, 0:1]
        u_pred = torch.cat(((-1 / lambda_val) * w1, (-1 / lambda_val) * w2), dim=1)
        y_pred = torch.cat((y1, y2), dim=1)

    with torch.no_grad():
        y_exact = exact_y(x_test)
        y_exact = torch.cat((y_exact[0],y_exact[1]),dim=1)
        u_exact = exact_control(x_test)
        u_exact = torch.cat((u_exact[0],u_exact[1]),dim=1)

        state_error = torch.sqrt(torch.mean((y_pred - y_exact) ** 2)) / torch.sqrt(torch.mean(y_exact ** 2))
        control_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)) / torch.sqrt(torch.mean(u_exact ** 2))



        print(f"\nEvaluation Results:")
        print(f"State Relative L2 Error: {state_error.item():.6e}")
        print(f"Control Relative L2 Error: {control_error.item():.6e}")

        return y_pred, y_exact, u_pred, u_exact, x_test


def plot_results(train_losses, control_errors, y_pred, y_exact, u_pred, u_exact):
    # Move to CPU for plotting
    n_plot = 256
    x_plot = torch.linspace(0, 1, n_plot)
    y_plot = torch.linspace(0, 1, n_plot)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    filename = f"data/Adjoint_Navier_Stoke.pth"
    torch.save({
        'x_1': X,
        'x_2': Y,
        'U_pred': u_pred,
        'U_exa': u_exact,
        'Y_pred': y_pred,
        'Y_exa': y_exact,
        'error_history': control_errors,
    }, filename)
    X = X.numpy()
    Y = Y.numpy()
    u_pred_cpu = u_pred[:,0].cpu().detach().numpy().reshape(n_plot,n_plot)
    u_exact_cpu = u_exact[:,0].cpu().numpy().reshape(n_plot,n_plot)
    u_pred_cpu2 = u_pred[:, 1].cpu().detach().numpy().reshape(n_plot, n_plot)
    u_exact_cpu2 = u_exact[:, 1].cpu().numpy().reshape(n_plot, n_plot)



    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].semilogy(train_losses, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[1, 0].semilogy(control_errors, label='Control Error', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Relative L2 Error')
    axes[1, 0].set_title('Control Error During Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True)


    im = axes[0, 1].imshow(u_exact_cpu, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('u1_exact')
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[1, 1].imshow(u_pred_cpu, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('u1_pred')
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[0, 2].imshow(u_exact_cpu2, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_title('u2_exact')
    plt.colorbar(im, ax=axes[0, 2])


    im = axes[1, 2].imshow(u_pred_cpu2, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    axes[1, 2].set_title('u2_pred')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('NavierStokes_adjoint.png')
    plt.close()
    #plt.show()

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_pred_cpu, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('(b) Numerical Optimal Control', fontweight='bold')
    ax1.view_init(elev=40, azim=250)  # 较低视角突出高度变化
    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_exact_cpu, cmap='inferno',
                             rstride=2, cstride=2, alpha=0.9)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('(a) Exact Optimal Control', fontweight='bold')
    ax2.view_init(elev=40, azim=250)
    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("data/adjoint_method_3D_View.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    model_y,model_w,train_losses, control_errors = train_model(
        n_samples=30000, epochs=20001, lr=1e-3
    )

    plot_results(train_losses, control_errors, y_pred, y_exact, u_pred, u_exact)