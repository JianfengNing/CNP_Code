"""Standalone Example 5.1 penalty method.

Run: python penalty.py --penalty-parameter 10
Reevaluate: python penalty.py --checkpoint data/penalty_10_example2.pth
Optional plots: add --show (no image file is created).
Saves ONE file: data/penalty_<rho>_example2.pth, including plotting data,
histories, network weights, final metrics, and evaluation samples.
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
    return torch.cos(np.pi * x) + torch.cos(6 * np.pi * x) + 2


def u_d(x, alpha=ALPHA):
    return ((alpha * np.pi**2 + 1) * torch.cos(np.pi * x)
            + (36 * alpha * np.pi**2 + 1) * torch.cos(6 * np.pi * x)
            + 0.25 * x + 2)


def compute_derivative(u, x, create_graph=True):
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


def train_model(n_samples=1000, epochs=5000, lr=1e-3,
                penalty_para=10.0, device=None):
    """Train the raw NN with the original squared integral-violation penalty."""
    penalty_para = float(penalty_para)
    if not np.isfinite(penalty_para) or penalty_para < 0:
        raise ValueError("penalty_para must be finite and nonnegative.")
    device = select_device() if device is None else torch.device(device)
    model = ResNetFunction(1, 1, hidden_dim=64, num_layers=2,
                          activation_fn=nn.Tanh).to(device)
    dtype = next(model.parameters()).dtype
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": lr}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    history = {key: [] for key in (
        "loss", "objective", "error", "tv", "data", "penalty",
        "moment0", "moment1", "violation0", "violation1",
    )}
    # Same randomized training grids and weights as the current CNP code.
    base_points = max(3, int(n_samples))
    if base_points % 2 == 0:
        base_points += 1

    for epoch in range(epochs):
        n_points = base_points + 2 * random.randint(0, 100)
        x, weights = simpson_grid(n_points, device, dtype)
        x.requires_grad_(True)
        optimizer.zero_grad()
        u = model(x)
        u_x = compute_derivative(u, x)
        tv_term = integrate(u_x.square(), weights)
        data_term = integrate((u-u_d(x)).square(), weights)
        objective = ALPHA*tv_term + data_term

        # Integrate first, then penalize the two scalar violations. These
        # moments stay connected to the parameter graph for backpropagation.
        moment0 = integrate(u, weights)
        moment1 = integrate(x*u, weights)
        violation0 = torch.relu(moment0-C1)
        violation1 = torch.relu(moment1-C2)
        penalty = penalty_para*(violation0.square()+violation1.square())
        loss = objective + penalty

        # All history entries refer to this same pre-update iterate.
        with torch.no_grad():
            exact = u_exact(x)
            relative_error = torch.sqrt(
                integrate((u-exact).square(), weights)/integrate(exact.square(), weights))
            for key, value in (
                ("loss", loss), ("objective", objective), ("error", relative_error),
                ("tv", tv_term), ("data", data_term), ("penalty", penalty),
                ("moment0", moment0), ("moment1", moment1),
                ("violation0", violation0), ("violation1", violation1),
            ):
                history[key].append(value.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}, "
                  f"J: {objective.item():.6f}, Penalty: {penalty.item():.6f}, "
                  f"RelL2err: {relative_error.item():.6f}, "
                  f"Violations: ({violation0.item():.3e}, {violation1.item():.3e}), "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    return model, penalty_para, history


def penalty_diagnostics(metrics, penalty_para):
    """Keep the original objective and penalized objective separately named."""
    squared_violation = (metrics["positive_violation0"]**2
                         + metrics["positive_violation1"]**2)
    weighted_penalty = float(penalty_para)*squared_violation
    return {
        "penalty_parameter": float(penalty_para),
        "squared_constraint_violation": float(squared_violation),
        "final_penalty": float(weighted_penalty),
        "final_penalized_objective": float(metrics["final_objective"]+weighted_penalty),
    }


def load_checkpoint(path):
    """Load the full penalty checkpoint written by this revised script."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = ("model_state_dict", "architecture", "penalty_para")
    if not isinstance(saved, dict) or any(key not in saved for key in required):
        raise ValueError(
            "A complete penalty .pth file with network weights is required. "
            "The original script saved predictions/history only; those files "
            "cannot reconstruct the trained network at new evaluation points.")
    if saved.get("method", "penalty_nn") != "penalty_nn":
        raise ValueError("Expected a penalty-method checkpoint.")
    for key, value in (("alpha", ALPHA), ("c1", C1), ("c2", C2)):
        if not np.isclose(float(saved.get(key, value)), value, rtol=0, atol=1e-14):
            raise ValueError(f"Checkpoint {key} does not match Example 5.1.")
    architecture = dict(saved["architecture"])
    if architecture.pop("activation", "Tanh") != "Tanh":
        raise ValueError("Expected the supplied Tanh network architecture.")
    dtype = saved["model_state_dict"]["input_layer.weight"].dtype
    model = ResNetFunction(**architecture).to(device="cpu", dtype=dtype)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    penalty_para = float(saved["penalty_para"])
    if not np.isfinite(penalty_para) or penalty_para < 0:
        raise ValueError("Saved penalty coefficient must be finite and nonnegative.")
    return model, penalty_para, saved


def evaluate_model(model, penalty_para, history, data_dir="data",
                   eval_intervals=1024, eval_order=10, check_order=5,
                   show=False, config=None, training_seconds=None):
    model.eval()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Evaluation precision matches the CNP code. The trained model is retained.
    evaluation_model = copy.deepcopy(model).cpu().double().eval()
    evaluation_model.requires_grad_(False)

    def evaluate_fixed(nodes):
        with torch.enable_grad():
            x = torch.as_tensor(np.asarray(nodes, dtype=np.float64),
                                dtype=torch.float64, device="cpu").reshape(-1, 1)
            x = x.clone().detach().requires_grad_(True)
            u = evaluation_model(x)
            du = compute_derivative(u, x, create_graph=False)
        return u.detach().numpy().ravel(), du.detach().numpy().ravel()

    x, weights = evaluation_rule(eval_intervals, eval_order)
    u, du = evaluate_fixed(x)
    metrics = evaluate_metrics(x, weights, u, du)
    metrics.update(penalty_diagnostics(metrics, penalty_para))
    xc, wc = evaluation_rule(eval_intervals, check_order)
    uc, duc = evaluate_fixed(xc)
    coarse = evaluate_metrics(xc, wc, uc, duc)
    coarse.update(penalty_diagnostics(coarse, penalty_para))
    checked_keys = (
        "relative_l2_u", "relative_l2_du", "final_objective",
        "moment0", "moment1", "final_penalty", "final_penalized_objective",
    )
    metrics.update({
        "method": "penalty_nn", "alpha": ALPHA, "c1": C1, "c2": C2,
        "evaluation_rule": "composite Gauss-Legendre", "evaluation_intervals": eval_intervals,
        "evaluation_order": eval_order, "evaluation_points": len(x),
        "rule_sha256": rule_fingerprint(x, weights),
        "evaluation_device": "cpu", "evaluation_dtype": "float64",
        "stored_parameter_dtype": str(next(model.parameters()).dtype),
        "independent_moment0": metrics["moment0"], "independent_moment1": metrics["moment1"],
        "independent_residual0": metrics["residual0"], "independent_residual1": metrics["residual1"],
        "max_constraint_violation": max(metrics["positive_violation0"], metrics["positive_violation1"]),
        "training_seconds": training_seconds,
        "training_epochs_completed": len(history["loss"]) if "loss" in history else None,
        "numpy_version": str(np.__version__), "torch_version": str(torch.__version__),
        "quadrature_check": {
            "order": check_order, "points": len(xc), "rule_sha256": rule_fingerprint(xc, wc),
            "metrics": {key: coarse[key] for key in checked_keys},
            "absolute_differences": {key: abs(metrics[key]-coarse[key]) for key in checked_keys},
        },
    })

    # Preserve the original (1000, 1) plotting-data format. These points do not
    # replace the common Gauss rule used for the final metrics above.
    plot_x = np.linspace(0, 1, 1000, dtype=np.float64)
    plot_u, plot_du = evaluate_fixed(plot_x)
    saved_data = {
        # The original loss_history is the penalized training loss.
        "loss_history": history.get("loss", []),
        "error_history": history.get("error", []),
        "tv_history": history.get("tv", []),
        "data_history": history.get("data", []),
        "objective_history": history.get("objective", []),
        "penalty_history": history.get("penalty", []),
        "penalty_para": float(penalty_para),
        "history": history,
        "metrics": metrics,
        "x_test": torch.from_numpy(plot_x.reshape(-1, 1)),
        "u_pred": torch.from_numpy(plot_u.reshape(-1, 1)),
        "u_deriv": torch.from_numpy(plot_du.reshape(-1, 1)),
        "u_exact": torch.from_numpy(exact_solution(plot_x).reshape(-1, 1)),
        "u_exa_deri": torch.from_numpy(exact_derivative(plot_x).reshape(-1, 1)),
    }
    # One file retains the original plotting keys plus the complete checkpoint.
    data_path = data_dir/f"penalty_{penalty_para:g}_example2.pth"
    checkpoint = {
        "method": "penalty_nn",
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "architecture": {"input_dim": model.input_layer.in_features,
                         "output_dim": model.output_layer.out_features,
                         "hidden_dim": model.input_layer.out_features,
                         "num_layers": len(model.blocks), "activation": "Tanh"},
        "penalty_para": float(penalty_para), "alpha": ALPHA, "c1": C1, "c2": C2,
        "config": dict(config or {}), "history": history, "metrics": metrics,
        "training_seconds": training_seconds,
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
            axes[0].plot(history["loss"], label="Penalized training loss", linewidth=2)
            if history.get("objective"):
                axes[0].plot(history["objective"], label="Original objective", linewidth=1.5)
            axes[0].set_yscale("log")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training history")
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
    parser.add_argument("--checkpoint", default=None, help="Reevaluate a complete penalty checkpoint without training.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--penalty-parameter", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--data-dir", default="data", help="Directory for the single penalty_<rho>_example2.pth result file.")
    parser.add_argument("--eval-intervals", type=int, default=1024)
    parser.add_argument("--eval-order", type=int, default=10)
    parser.add_argument("--check-order", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    evaluation_rule(args.eval_intervals, args.eval_order)
    evaluation_rule(args.eval_intervals, args.check_order)
    config = vars(args).copy()
    if args.checkpoint:
        model, penalty_para, saved = load_checkpoint(args.checkpoint)
        history = saved.get("history", {})
        training_seconds = saved.get("training_seconds")
        config.update(mode="checkpoint_evaluation", source_config=saved.get("config", {}),
                      penalty_parameter=penalty_para)
    else:
        if args.epochs < 1:
            parser.error("--epochs must be positive for a new training run.")
        if not np.isfinite(args.penalty_parameter) or args.penalty_parameter < 0:
            parser.error("--penalty-parameter must be finite and nonnegative.")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = select_device() if args.device is None else torch.device(args.device)
        print(f"Training device: {device}; final evaluation device/dtype: CPU/float64")
        synchronize(device)
        begin = time.perf_counter()
        model, penalty_para, history = train_model(
            n_samples=args.n_samples, epochs=args.epochs, lr=args.lr,
            penalty_para=args.penalty_parameter, device=device)
        synchronize(device)
        training_seconds = time.perf_counter()-begin
        config.update(mode="training_and_evaluation", training_device=str(device),
                      training_quadrature="composite Simpson on randomized odd equispaced grids",
                      timing_scope="Model/optimizer initialization and training including diagnostics; final evaluation excluded.")
    evaluate_model(model, penalty_para, history, data_dir=args.data_dir,
                   eval_intervals=args.eval_intervals, eval_order=args.eval_order,
                   check_order=args.check_order,
                   show=args.show, config=config, training_seconds=training_seconds)


if __name__ == "__main__":
    main()
