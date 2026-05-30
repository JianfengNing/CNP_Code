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

# Fully-Connected ResNet
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



def u_exact(x):
    return torch.cos(np.pi*x) + torch.cos(6*np.pi*x) + 2


def u_d(x):
    return ((1/100) * (np.pi**2)+1) * torch.cos(np.pi*x)+((36/100) * (np.pi**2)+1) * torch.cos(6*np.pi*x) + 0.5*0.5*x+2

def compute_derivative(u, x):

    ones = torch.ones_like(u)
    gradients = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return gradients



def train_model(n_samples=500, integral=1.0, epochs=5000, lr=1e-3):
    model = ResNetFunction(
        input_dim=1,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        activation_fn=nn.Tanh,
    ).to(device)
    lambda_param = nn.Parameter(torch.tensor([1.0, 1.0], device=device))

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': lambda_param, 'lr': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)


    num_epochs = epochs
    loss_history = []
    error_history = []
    tv_history = []
    data_history = []
    c_1 = integral

    alpha = 0.01

    c_1 = 3.0
    c_2 = 1.0 - 2/(np.pi**2)

    b = 12.0 * c_2 - 6.0 * c_1
    a = c_1 - 0.5 * b

    for epoch in range(num_epochs):
        N = n_samples + random.randint(0, 200)
        x = torch.linspace(0, 1, N, device=device).view(-1, 1).requires_grad_(True)
        optimizer.zero_grad()
        u_n = model(x)

        lambda_sq = lambda_param ** 2
        integral_in = 12 * torch.mean(u_n * (0.5 - x))

        u = (u_n - torch.mean(u_n) - integral_in * (0.5 - x) + (a + b * x) +
             lambda_sq[0] * (1 - 2 * x) + lambda_sq[1] * (-2 + 3 * x))

        u_x = compute_derivative(u_n, x) + integral_in + b - 2 * lambda_sq[0] + 3 * lambda_sq[1]

        tv_term = torch.mean(u_x ** 2)
        data_term = torch.mean((u - u_d(x)) ** 2)

        loss = alpha * tv_term + data_term
        loss.backward()
        optimizer.step()
        scheduler.step()


        loss_history.append(loss.item())
        tv_history.append(tv_term.item())
        data_history.append(data_term.item())

        u_exa = u_exact(x)
        l2error = (torch.mean((u - u_exa) ** 2) / torch.mean(u_exa ** 2)) ** 0.5
        error_history.append(l2error.item())


        if epoch % 200 == 0:
            print(f'Epoch {epoch:4d}, Loss: {loss.item():.6f}, '
                  f'TV: {tv_term.item():.6f}, Data: {data_term.item():.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            print(f'RelL2err: {l2error:.6f}, ')

    return model, lambda_param, loss_history, tv_history, data_history, error_history, a, b

def evaluate_model(model, lambda_param,error_history, loss_history, a, b):
    x_test = torch.linspace(0, 1, 1000, device=device).view(-1, 1).requires_grad_(True)

    u_nn = model(x_test)
    lambda_sq = lambda_param ** 2

    integral_in = 12 * torch.mean(u_nn * (0.5 - x_test))

    u_pred = (u_nn - torch.mean(u_nn) - integral_in * (0.5 - x_test) + (a + b * x_test) +
         lambda_sq[0] * (1 - 2 * x_test) + lambda_sq[1] * (-2 + 3 * x_test))
    v_val = u_d(x_test)
    u_deriv = compute_derivative(u_nn, x_test) + integral_in + b - 2 * lambda_sq[0] + 3 * lambda_sq[1]
    u_exa = u_exact(x_test)
    u_exa_deriv = compute_derivative(u_exa, x_test)
    rel2error = (torch.mean((u_pred - u_exa) ** 2) / torch.mean(u_exa ** 2)) ** 0.5
    rel2error_der = (torch.mean((u_deriv - u_exa_deriv) ** 2) / torch.mean(u_exa_deriv ** 2)) ** 0.5
    print(f'RelL2err: {rel2error:.6f}, RelL2err_deri: {rel2error_der:.6f},')

    filename = f"data/my_example2.pth"
    torch.save({
        'loss_history': loss_history,
        'error_history': error_history,
        'x_test': x_test,
        'u_pred': u_pred.detach().cpu(),
        'u_deriv': u_deriv.detach().cpu(),
        'u_exact': u_exa.detach().cpu(),
        'u_exa_deri': u_exa_deriv,
    }, filename)




    with torch.no_grad():
        x_test_cpu = x_test.detach().cpu().numpy()
        u_pred_cpu = u_pred.detach().cpu().numpy()
        u_exa_cpu = u_exa.detach().cpu().numpy()
        v_val_cpu = v_val.detach().cpu().numpy()
        u_deriv_cpu = u_deriv.detach().cpu().numpy()
        u_exa_deriv = u_exa_deriv.detach().cpu().numpy()

    final_tv = np.mean(u_deriv_cpu ** 2)
    final_data = np.mean((u_pred_cpu - v_val_cpu) ** 2)
    final_loss = 0.1 * final_tv + final_data

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(loss_history, label='Total Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title('Training Loss History')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(x_test_cpu, u_exa_cpu, 'r-', linewidth=2, label='Exact solution')
    plt.plot(x_test_cpu, u_pred_cpu, 'b--', linewidth=2, label='Computed Solution')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Solutions')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(x_test_cpu, u_exa_deriv, 'r-', linewidth=2, label='Exact derivative')
    plt.plot(x_test_cpu, u_deriv_cpu, 'b--', linewidth=2, label='Computed derivative')
    plt.xlabel('x')
    plt.ylabel('du/dx')
    plt.legend()
    plt.title('Derivative of Solutions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('my_method.png')
    plt.show()



if __name__ == "__main__":
    integral_value = 1.0
    model, lambda_param, loss_history, tv_history, data_history,error_history,a,b\
        =train_model(n_samples=1000,integral=integral_value, epochs=5000, lr=1e-3)

    evaluate_model(model, lambda_param, error_history, loss_history, a, b)

