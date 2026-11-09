"""Standalone Example 5.1 CNP method.

Run: python CNP.py
Reevaluate: python CNP.py --checkpoint data/my_example2.pth
Optional plots: add --show (no image file is created).
Saves ONE file: data/my_example2.pth, including plotting data, histories,
network weights, frozen projection, final metrics, and evaluation samples.
Dependencies: NumPy, PyTorch, Matplotlib. No local helper module is needed.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import hashlib
from numpy.polynomial.legendre import leggauss


ALPHA = 0.01


C1 = 3.0


C2 = 1.0 - 2.0 / np.pi**2


def exact_solution(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cos(np.pi*x) + np.cos(6.0*np.pi*x) + 2.0


def exact_derivative(x):
    x = np.asarray(x, dtype=np.float64)
    return -np.pi*np.sin(np.pi*x) - 6.0*np.pi*np.sin(6.0*np.pi*x)


def desired_solution(x):
    x = np.asarray(x, dtype=np.float64)
    return ((1.0 + ALPHA*np.pi**2)*np.cos(np.pi*x)
            + (1.0 + 36.0*ALPHA*np.pi**2)*np.cos(6.0*np.pi*x)
            + 2.0 + 0.25*x)


def exact_objective():
    return float(0.5*ALPHA**2*np.pi**4*1297.0
                 + 0.5*ALPHA*np.pi**2*37.0 + 1.0/48.0 - ALPHA)


def evaluation_rule(n_intervals=1024, order=10):
    """Composite Gauss--Legendre rule on one fixed uniform partition."""
    if int(n_intervals) != n_intervals or int(order) != order:
        raise ValueError("n_intervals and order must be positive integers.")
    n_intervals, order = int(n_intervals), int(order)
    if n_intervals < 1 or order < 1:
        raise ValueError("n_intervals and order must be positive integers.")
    edges = np.linspace(0.0, 1.0, n_intervals+1, dtype=np.float64)
    z, w = leggauss(order)
    widths = np.diff(edges)
    x = (edges[:-1, None] + widths[:, None]*(z[None, :]+1.0)/2.0).ravel()
    weights = (widths[:, None]*w[None, :]/2.0).ravel()
    return x, weights


def rule_fingerprint(x, weights):
    """A reproducibility identifier for the exact points AND exact weights."""
    digest = hashlib.sha256()
    for values in (x, weights):
        array = np.ascontiguousarray(values, dtype="<f8").reshape(-1)
        digest.update(np.asarray([array.size], dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_metrics(x, weights, u, du):
    """The same weighted definitions for NN, direct FE, and iterative FE."""
    x, weights, u, du = [np.asarray(v, dtype=np.float64).reshape(-1)
                          for v in (x, weights, u, du)]
    if len(x) == 0 or not (len(x) == len(weights) == len(u) == len(du)):
        raise ValueError("Coordinates, weights, u, and du must have the same positive length.")
    if not all(np.all(np.isfinite(v)) for v in (x, weights, u, du)):
        raise ValueError("Evaluation arrays contain non-finite values.")
    if np.any(weights <= 0) or not np.isclose(np.sum(weights), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Expected positive integration weights on [0,1] summing to one.")
    exact_u = exact_solution(x)
    exact_du = exact_derivative(x)
    target = desired_solution(x)
    integrate = lambda values: float(np.dot(weights, values))
    u_error_sq = integrate((u-exact_u)**2)
    du_error_sq = integrate((du-exact_du)**2)
    denominator_u = integrate(exact_u**2)
    denominator_du = integrate(exact_du**2)
    if denominator_u <= 0 or denominator_du <= 0:
        raise ValueError("Reference norms must be positive on the evaluation rule.")
    final_tv = integrate(du**2)
    final_data = integrate((u-target)**2)
    objective = ALPHA*final_tv + final_data
    moment0, moment1 = integrate(u), integrate(x*u)
    residual0, residual1 = moment0-C1, moment1-C2
    reference_objective = exact_objective()
    return {
        "relative_l2_u": float(np.sqrt(u_error_sq/denominator_u)),
        "relative_l2_du": float(np.sqrt(du_error_sq/denominator_du)),
        "final_tv": final_tv,
        "final_data": final_data,
        "final_objective": objective,
        "moment0": moment0,
        "moment1": moment1,
        "residual0": residual0,
        "residual1": residual1,
        "positive_violation0": max(0.0, residual0),
        "positive_violation1": max(0.0, residual1),
        "exact_objective": reference_objective,
        "objective_gap_to_continuous_optimum": objective-reference_objective,
        "squared_l2_error_u": u_error_sq,
        "squared_l2_error_du": du_error_sq,
        "reference_squared_l2_u": denominator_u,
        "reference_squared_l2_du": denominator_du,
        "continuum_objective_identity_error": abs(
            objective-reference_objective-u_error_sq-ALPHA*du_error_sq+0.5*residual1),
    }


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4,
                 activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[FCResBlock(hidden_dim, self.activation) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output_layer(self.blocks(self.activation(self.input_layer(x))))


def u_exact(x):
    return torch.cos(np.pi * x) + torch.cos(6 * np.pi * x) + 2


def u_exact_derivative(x):
    return -np.pi * torch.sin(np.pi * x) - 6 * np.pi * torch.sin(6 * np.pi * x)


def u_d(x, alpha=ALPHA):
    return ((alpha * np.pi**2 + 1) * torch.cos(np.pi * x)
            + (36 * alpha * np.pi**2 + 1) * torch.cos(6 * np.pi * x)
            + 0.25 * x + 2)


def compute_derivative(u, x, create_graph=True):
    # Apply this to the pointwise network, not the batch-coupled projection.
    return torch.autograd.grad(
        outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
        create_graph=create_graph, retain_graph=create_graph,
    )[0]


def simpson_grid(n_points, device, dtype):
    """Endpoint-inclusive grid with an odd number of points on [0, 1]."""
    n_points = max(3, int(n_points))
    if n_points % 2 == 0:
        n_points += 1
    x = torch.linspace(0, 1, n_points, device=device, dtype=dtype).view(-1, 1)
    weights = torch.ones_like(x)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights = weights / (3 * (n_points - 1))
    return x, weights


def integrate(values, weights):
    return torch.sum(weights * values)


def projection_coefficients(raw, x, weights):
    # Keep these connected to the parameter graph during training.
    m0 = integrate(raw, weights)
    correction = 12 * integrate(raw * (0.5 - x), weights)
    return m0, correction


def constrained_values(raw, x, lambda_sq, a, b, m0, correction):
    return (raw - m0 - correction * (0.5 - x) + a + b * x
            + lambda_sq[0] * (1 - 2 * x)
            + lambda_sq[1] * (-2 + 3 * x))


def constrained_derivative(raw, x, lambda_sq, b, correction, create_graph=True):
    # The global moments are constant with respect to the evaluation coordinate.
    # Their dependence on network parameters is retained in the training graph.
    return (compute_derivative(raw, x, create_graph=create_graph)
            + correction + b - 2 * lambda_sq[0] + 3 * lambda_sq[1])


def train_model(n_samples=1000, epochs=5000, lr=1e-3, device=None):
    device = select_device() if device is None else torch.device(device)
    model = ResNetFunction(1, 1, hidden_dim=64, num_layers=2,
                          activation_fn=nn.Tanh).to(device)
    dtype = next(model.parameters()).dtype
    # These are auxiliary CNP parameters (gamma_i in the paper), not KKT multipliers.
    lambda_param = nn.Parameter(torch.tensor([1.0, 1.0], device=device, dtype=dtype))
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": [lambda_param], "lr": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    history = {key: [] for key in ("loss", "error", "tv", "data", "moment0", "moment1")}
    b = 12 * C2 - 6 * C1
    a = C1 - 0.5 * b
    # Keep the original approximate range of grid sizes, but require odd counts.
    base_points = max(3, int(n_samples))
    if base_points % 2 == 0:
        base_points += 1

    for epoch in range(epochs):
        n_points = base_points + 2 * random.randint(0, 100)
        x, weights = simpson_grid(n_points, device, dtype)
        x.requires_grad_(True)
        optimizer.zero_grad()
        raw = model(x)
        lambda_sq = lambda_param.square()
        m0, correction = projection_coefficients(raw, x, weights)
        u = constrained_values(raw, x, lambda_sq, a, b, m0, correction)
        ux = constrained_derivative(raw, x, lambda_sq, b, correction)
        tv_term = integrate(ux.square(), weights)
        data_term = integrate((u - u_d(x)).square(), weights)
        loss = ALPHA * tv_term + data_term

        # All history entries refer to the same pre-update iterate.
        with torch.no_grad():
            exact = u_exact(x)
            relative_error = torch.sqrt(
                integrate((u - exact).square(), weights)
                / integrate(exact.square(), weights)
            )
            moment0 = integrate(u, weights)
            moment1 = integrate(x * u, weights)
            for key, value in (("loss", loss), ("error", relative_error),
                               ("tv", tv_term), ("data", data_term),
                               ("moment0", moment0), ("moment1", moment1)):
                history[key].append(value.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}, "
                  f"RelL2err: {relative_error.item():.6f}, "
                  f"Q(u): {moment0.item():.8f}, Q(xu): {moment1.item():.8f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    return model, lambda_param, history, a, b



def freeze_projection(model, lambda_param, a, b, n_projection):
    """Use the original projection rule/dtype once after training."""
    parameter = next(model.parameters())
    xq, wq = simpson_grid(n_projection, parameter.device, parameter.dtype)
    with torch.no_grad():
        rawq = model(xq)
        m0, correction = projection_coefficients(rawq, xq, wq)
        lambda_sq = lambda_param.detach().square()
        uq = constrained_values(rawq, xq, lambda_sq, a, b, m0, correction)
        projection = {
            "rule": "composite Simpson", "nodes": len(xq),
            "m0": m0.detach().cpu().clone(),
            "correction": correction.detach().cpu().clone(),
            "moment0": integrate(uq, wq).item(),
            "moment1": integrate(xq*uq, wq).item(),
            "coefficient_dtype": str(parameter.dtype),
        }
    return projection, lambda_sq.detach().cpu().clone()


def load_checkpoint(path):
    """Load your own full checkpoint saved by this or the supplied old script."""
    # The full checkpoint includes NumPy diagnostics, not just tensor weights.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model_state_dict", "lambda_param", "architecture", "a", "b", "projection")
    if any(key not in saved for key in required):
        raise ValueError("Expected a complete CNP .pth file with network weights and saved projection coefficients.")
    for key, value in (("alpha", ALPHA), ("c1", C1), ("c2", C2)):
        if not np.isclose(float(saved.get(key, value)), value, rtol=0, atol=1e-14):
            raise ValueError(f"Checkpoint {key} does not match this Example 5.1 problem.")
    architecture = dict(saved["architecture"])
    if architecture.pop("activation", "Tanh") != "Tanh":
        raise ValueError("Expected the supplied Tanh network architecture.")
    dtype = saved["model_state_dict"]["input_layer.weight"].dtype
    model = ResNetFunction(**architecture).to(device="cpu", dtype=dtype)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    lambda_param = torch.as_tensor(saved["lambda_param"]).detach().cpu().clone()
    projection = dict(saved["projection"])
    for key in ("m0", "correction"):
        if key not in projection:
            raise ValueError("Saved projection coefficients are required; they are never recomputed here.")
        projection[key] = torch.as_tensor(projection[key]).detach().cpu().clone()
    # Older checkpoints retain these diagnostics in metrics, rather than projection.
    for key in ("moment0", "moment1"):
        old_value = saved.get("metrics", {}).get("projection_"+key)
        if key not in projection and old_value is not None:
            projection[key] = float(old_value)
    saved["projection"] = projection
    # Square in the original dtype BEFORE conversion to float64. New checkpoints
    # also save the already squared coefficients of the final represented function.
    saved["lambda_sq"] = (torch.as_tensor(saved["lambda_sq"]).detach().cpu().clone()
                          if "lambda_sq" in saved else lambda_param.square())
    return model, lambda_param, saved


def evaluate_model(model, lambda_param, history, a, b, data_dir="data",
                   n_projection=4001, eval_intervals=1024, eval_order=10,
                   check_order=5, show=False, config=None,
                   projection=None, lambda_sq=None, training_seconds=None):
    model.eval()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    reused_projection = projection is not None
    if projection is None:
        projection, lambda_sq = freeze_projection(model, lambda_param, a, b, n_projection)
    elif lambda_sq is None:
        lambda_sq = lambda_param.detach().square().cpu()

    # The original model and its stored coefficients retain their original dtype.
    # Only this evaluation copy uses CPU float64, so all methods use identical x.
    evaluation_model = copy.deepcopy(model).cpu().double().eval()
    evaluation_model.requires_grad_(False)
    m0 = torch.as_tensor(projection["m0"], dtype=torch.float64).detach().cpu()
    correction = torch.as_tensor(projection["correction"], dtype=torch.float64).detach().cpu()
    squared64 = torch.as_tensor(lambda_sq, dtype=torch.float64).detach().cpu()

    def evaluate_fixed(nodes):
        with torch.enable_grad():
            x = torch.as_tensor(np.asarray(nodes, dtype=np.float64),
                                dtype=torch.float64, device="cpu").reshape(-1, 1)
            x = x.clone().detach().requires_grad_(True)
            raw = evaluation_model(x)
            u = constrained_values(raw, x, squared64, a, b, m0, correction)
            ux = constrained_derivative(raw, x, squared64, b, correction, create_graph=False)
        return u.detach().numpy().ravel(), ux.detach().numpy().ravel()

    x, weights = evaluation_rule(eval_intervals, eval_order)
    u, du = evaluate_fixed(x)
    metrics = evaluate_metrics(x, weights, u, du)
    xc, wc = evaluation_rule(eval_intervals, check_order)
    uc, duc = evaluate_fixed(xc)
    coarse = evaluate_metrics(xc, wc, uc, duc)
    checked_keys = ("relative_l2_u", "relative_l2_du", "final_objective", "moment0", "moment1")
    metrics.update({
        "alpha": ALPHA, "c1": C1, "c2": C2,
        "evaluation_rule": "composite Gauss-Legendre", "evaluation_intervals": eval_intervals,
        "evaluation_order": eval_order, "evaluation_points": len(x),
        "rule_sha256": rule_fingerprint(x, weights),
        "evaluation_device": "cpu", "evaluation_dtype": "float64",
        "stored_parameter_dtype": str(next(model.parameters()).dtype),
        "projection_nodes": projection.get("nodes"),
        "projection_coefficients_reused": reused_projection,
        "algebraic_target0": C1-0.5*float(squared64[1]),
        "algebraic_target1": C2-float(squared64[0])/6.0,
        "independent_moment0": metrics["moment0"], "independent_moment1": metrics["moment1"],
        "independent_residual0": metrics["residual0"], "independent_residual1": metrics["residual1"],
        "training_seconds": training_seconds,
        "training_epochs_completed": len(history["loss"]) if "loss" in history else None,
        "quadrature_check": {
            "order": check_order, "points": len(xc), "rule_sha256": rule_fingerprint(xc, wc),
            "metrics": {key: coarse[key] for key in checked_keys},
            "absolute_differences": {key: abs(metrics[key]-coarse[key]) for key in checked_keys},
        },
    })
    for index in (0, 1):
        key = f"moment{index}"
        if key in projection:
            value = float(projection[key])
            metrics["projection_"+key] = value
            metrics[f"discrete_identity_residual{index}"] = value-metrics[f"algebraic_target{index}"]
            metrics[f"quadrature_gap{index}"] = metrics[key]-value

    # Preserve the original (1000, 1) plotting-data format. These points do not
    # replace the common Gauss rule used for the final metrics above.
    plot_x = np.linspace(0, 1, 1000, dtype=np.float64)
    plot_u, plot_du = evaluate_fixed(plot_x)
    saved_data = {
        "loss_history": history.get("loss", []),
        "error_history": history.get("error", []),
        "tv_history": history.get("tv", []),
        "data_history": history.get("data", []),
        "objective_history": history.get("loss", []),
        "history": history,
        "metrics": metrics,
        "x_test": torch.from_numpy(plot_x.reshape(-1, 1)),
        "u_pred": torch.from_numpy(plot_u.reshape(-1, 1)),
        "u_deriv": torch.from_numpy(plot_du.reshape(-1, 1)),
        "u_exact": torch.from_numpy(exact_solution(plot_x).reshape(-1, 1)),
        "u_exa_deri": torch.from_numpy(exact_derivative(plot_x).reshape(-1, 1)),
    }
    # One file retains the original plotting keys plus the complete checkpoint.
    data_path = data_dir/"my_example2.pth"
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "lambda_param": lambda_param.detach().cpu(),
        "lambda_sq": torch.as_tensor(lambda_sq).detach().cpu(),
        "architecture": {"input_dim": model.input_layer.in_features,
                         "output_dim": model.output_layer.out_features,
                         "hidden_dim": model.input_layer.out_features,
                         "num_layers": len(model.blocks), "activation": "Tanh"},
        "config": dict(config or {}), "history": history,
        "alpha": ALPHA, "c1": C1, "c2": C2, "a": a, "b": b,
        "projection": projection, "metrics": metrics, "training_seconds": training_seconds,
        "training_epochs_completed": metrics["training_epochs_completed"],
        **saved_data,
        "x_eval": torch.from_numpy(x.reshape(-1, 1)),
        "weights_eval": torch.from_numpy(weights.reshape(-1, 1)),
        "u_eval": torch.from_numpy(u.reshape(-1, 1)),
        "du_eval": torch.from_numpy(du.reshape(-1, 1)),
        "u_exact_eval": torch.from_numpy(exact_solution(x).reshape(-1, 1)),
        "du_exact_eval": torch.from_numpy(exact_derivative(x).reshape(-1, 1)),
        "target_eval": torch.from_numpy(desired_solution(x).reshape(-1, 1)),
        "x_check": torch.from_numpy(xc.reshape(-1, 1)),
        "weights_check": torch.from_numpy(wc.reshape(-1, 1)),
        "u_check": torch.from_numpy(uc.reshape(-1, 1)),
        "du_check": torch.from_numpy(duc.reshape(-1, 1)),
    }
    torch.save(checkpoint, data_path)
    print(f"Saved all data and model: {data_path.resolve()}")
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        if history.get("loss"):
            axes[0].plot(history["loss"], label="Total loss", linewidth=2)
            axes[0].set_yscale("log")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
        axes[1].plot(plot_x, exact_solution(plot_x), "r-", label="Exact solution")
        axes[1].plot(plot_x, plot_u, "b--", label="Computed solution")
        axes[1].set(xlabel="x", ylabel="u", title="Solutions")
        axes[2].plot(plot_x, exact_derivative(plot_x), "r-", label="Exact derivative")
        axes[2].plot(plot_x, plot_du, "b--", label="Computed derivative")
        axes[2].set(xlabel="x", ylabel="du/dx", title="Derivatives")
        for axis in axes:
            if axis.lines:
                axis.legend()
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Evaluate your saved checkpoint without training.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--data-dir", default="data", help="Directory for the single my_example2.pth result file.")
    parser.add_argument("--projection-nodes", type=int, default=4001)
    parser.add_argument("--eval-intervals", type=int, default=1024)
    parser.add_argument("--eval-order", type=int, default=10)
    parser.add_argument("--check-order", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    # Validate the rule before spending time on training.
    evaluation_rule(args.eval_intervals, args.eval_order)
    evaluation_rule(args.eval_intervals, args.check_order)
    config = vars(args).copy()
    if args.checkpoint:
        model, lambda_param, saved = load_checkpoint(args.checkpoint)
        history, a, b = saved.get("history", {}), float(saved["a"]), float(saved["b"])
        projection, lambda_sq = saved["projection"], saved["lambda_sq"]
        training_seconds = saved.get("training_seconds")
        config.update(mode="checkpoint_evaluation", source_config=saved.get("config", {}))
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = select_device() if args.device is None else torch.device(args.device)
        print(f"Training device: {device}; final evaluation device/dtype: CPU/float64")
        synchronize(device)
        begin = time.perf_counter()
        model, lambda_param, history, a, b = train_model(
            n_samples=args.n_samples, epochs=args.epochs, lr=args.lr, device=device)
        synchronize(device)
        training_seconds = time.perf_counter()-begin
        projection, lambda_sq = None, None
        config.update(mode="training_and_evaluation", training_device=str(device),
                      timing_scope="Model/optimizer initialization and training, including training diagnostics; final evaluation excluded.")
    evaluate_model(model, lambda_param, history, a, b, data_dir=args.data_dir,
                   n_projection=args.projection_nodes, eval_intervals=args.eval_intervals,
                   eval_order=args.eval_order, check_order=args.check_order,
                   show=args.show, config=config,
                   projection=projection, lambda_sq=lambda_sq, training_seconds=training_seconds)


if __name__ == "__main__":
    main()
