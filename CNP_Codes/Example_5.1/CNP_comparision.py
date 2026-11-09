"""Example 5.1: scalar-map and auxiliary-initialization sensitivity study.

Run: python CNP_sensitivity_with_trial_settings.py
Default: 24 runs, 5,000 Adam epochs each, width 64 and two residual blocks.
  1. h(gamma)=gamma**2, gamma=(1,1).
  2. h(gamma)=exp(gamma), gamma=(1,1).
  3. Ten square-map runs, each gamma_i independently Uniform[0.5,1.5].
  4. Twelve exponential-map runs, each gamma_i independently Normal(0,1).
Set epochs and initial learning rates separately for each map in make_parser():
  --square-epochs, --square-lr (network), --square-gamma-lr (auxiliary variables)
  --exp-epochs,    --exp-lr (network),    --exp-gamma-lr (auxiliary variables)
Set random trial counts using --square-trials (default 10) and --exp-trials
(default 12). Total runs = square-trials + exp-trials + two fixed-initialization runs.
Set the square-map uniform interval using --uniform-low and --uniform-high.
Set the two baseline initial values using --square-gamma-init GAMMA1 GAMMA2
and --exp-gamma-init GAMMA1 GAMMA2 (both default to 1 1). These are initial
values only: gamma remains trainable. Random trials use their own sampled values.
Each map's training settings apply to its baseline AND all its random trials.
The learning rates retain the original StepLR decay: factor 0.8 every 200 epochs.
ALL runs reuse identical initial neural weights and a common training-grid
sequence, truncated to each map's epoch count. gamma remains trainable.
Use --random-mode shared if the same random scalar should initialize both
auxiliary variables in each random trial. The default is independent draws.
Normal(0,1) means mean zero and standard deviation one. Sampling happens
once per trial, at initialization; gamma is then optimized normally.

Prints relative L2 errors for u and u', signed constraint slacks, initial/final
gamma, and timing. Writes results.csv, results.json, and a distribution figure
(PNG/PDF). Optional --save-models retains each trained network and projection.
All output defaults to a new cnp_sensitivity_results folder beside this script.

Positive slack means feasible: s1=3-integral(u), s2=C2-integral(x*u).
The exact optimum has (s1,s2)=(1,0). The implemented rays give algebraic slacks
(h(gamma_2)/2, h(gamma_1)/6). The primary reported slacks use independent
Gauss quadrature of a FIXED represented function; small negative values are
reported without clipping. Finite quadrature is not a continuous-feasibility
certificate. Simpson projection coefficients are frozen once after training.

At the default gamma=(1,1), square and exponential maps have DIFFERENT initial
coefficients (1 versus e). Setting equal gamma does not match initial slacks.
No clipping of exp(gamma), or post-training feasibility correction, is used.

Quick execution check (not paper results):
  python CNP_sensitivity_with_trial_settings.py --square-epochs 3 --exp-epochs 4 --n-samples 31 --grid-jitter 2 --width 8 --blocks 1 --projection-nodes 101 --eval-intervals 16 --log-every 1 --output quick_check
Dependencies: NumPy, PyTorch, Matplotlib. No local helper module is needed.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from numpy.polynomial.legendre import leggauss

ALPHA = 0.01
C1 = 3.0
C2 = 1.0 - 2.0 / np.pi**2
SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_GROUPS = (
    ("random_square", "square"),
    ("random_exp", "exp"),
)

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


def constrained_values(raw, x, coefficients, a, b, m0, correction):
    return (raw - m0 - correction * (0.5 - x) + a + b * x
            + coefficients[0] * (1 - 2 * x)
            + coefficients[1] * (-2 + 3 * x))


def constrained_derivative(raw, x, coefficients, b, correction, create_graph=True):
    # The global moments are constant with respect to the evaluation coordinate.
    # Their dependence on network parameters is retained in the training graph.
    return (compute_derivative(raw, x, create_graph=create_graph)
            + correction + b - 2 * coefficients[0] + 3 * coefficients[1])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def map_coefficients(gamma, map_name):
    if map_name == "square":
        return gamma.square()
    if map_name == "exp":
        return torch.exp(gamma)
    raise ValueError(f"Unknown scalar map: {map_name}")


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed, device):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def make_experiments(args):
    rng = np.random.default_rng(args.gamma_seed)
    shape = (args.square_trials, 2 if args.random_mode == "independent" else 1)
    random_gamma = rng.uniform(args.uniform_low, args.uniform_high, size=shape)
    # Separate RNG preserves the previous square-map draws exactly.
    exp_rng = np.random.default_rng(args.exp_gamma_seed)
    exp_shape = (args.exp_trials, 2 if args.random_mode == "independent" else 1)
    exp_gamma = exp_rng.normal(loc=0.0, scale=1.0, size=exp_shape)
    if args.random_mode == "shared":
        random_gamma = np.repeat(random_gamma, 2, axis=1)
        exp_gamma = np.repeat(exp_gamma, 2, axis=1)
    experiments = [
        {"run": "square_fixed_init", "group": "map_comparison", "map": "square",
         "gamma_init": list(args.square_gamma_init)},
        {"run": "exp_fixed_init", "group": "map_comparison", "map": "exp",
         "gamma_init": list(args.exp_gamma_init)},
    ]
    experiments += [
        {"run": f"square_random_{i+1:02d}", "group": "random_square", "map": "square",
         "gamma_init": values.tolist()}
        for i, values in enumerate(random_gamma)
    ]
    experiments += [
        {"run": f"exp_random_{i+1:02d}", "group": "random_exp", "map": "exp",
         "gamma_init": values.tolist()}
        for i, values in enumerate(exp_gamma)
    ]
    return experiments


def build_training_cache(args, schedule, device, dtype):
    """Cache only deterministic, detached quadrature/target data (about 1.3 MB)."""
    cache = {}
    with torch.no_grad():
        for size in sorted(set(schedule)):
            x, weights = simpson_grid(size, device, dtype)
            cache[size] = (x, weights, u_d(x))
    return cache


def training_settings(args, map_name):
    """Resolve the separate parser settings for one scalar map."""
    return {
        "epochs": getattr(args, f"{map_name}_epochs"),
        "network_lr": getattr(args, f"{map_name}_lr"),
        "gamma_lr": getattr(args, f"{map_name}_gamma_lr"),
    }


def train_one(args, experiment, base_state, schedule, cache, device, dtype):
    settings = training_settings(args, experiment["map"])
    epochs = settings["epochs"]
    if len(schedule) != epochs:
        raise ValueError("Training-grid schedule must match this map's epoch count")
    model = ResNetFunction(1, 1, hidden_dim=args.width, num_layers=args.blocks).to(device=device, dtype=dtype)
    model.load_state_dict(base_state)
    gamma = nn.Parameter(torch.tensor(experiment["gamma_init"], device=device, dtype=dtype))
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": settings["network_lr"]},
        {"params": [gamma], "lr": settings["gamma_lr"]},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    b = 12*C2-6*C1
    a = C1-0.5*b
    history = []
    training_seconds = 0.0
    synchronize(device)
    begin = time.perf_counter()
    for epoch, size in enumerate(schedule, 1):
        coords, weights, target = cache[size]
        x = coords.detach().clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        raw = model(x)
        coefficients = map_coefficients(gamma, experiment["map"])
        m0, correction = projection_coefficients(raw, x, weights)
        u = constrained_values(raw, x, coefficients, a, b, m0, correction)
        du = constrained_derivative(raw, x, coefficients, b, correction)
        loss = integrate((u-target).square(), weights)+ALPHA*integrate(du.square(), weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % args.log_every == 0 or epoch == epochs:
            synchronize(device)
            training_seconds += time.perf_counter()-begin
            loss_value = float(loss.detach().cpu())
            gamma_values = gamma.detach().cpu().tolist()
            if not math.isfinite(loss_value) or not all(math.isfinite(v) for v in gamma_values):
                raise FloatingPointError(f"Nonfinite loss/parameters in {experiment['run']} at epoch {epoch}")
            history.append({"epoch": epoch, "pre_update_loss": loss_value,
                            "post_update_gamma": gamma_values, "training_seconds": training_seconds,
                            "next_network_lr": optimizer.param_groups[0]["lr"],
                            "next_gamma_lr": optimizer.param_groups[1]["lr"]})
            print(f"  epoch {epoch:5d}/{epochs}: loss={loss_value:.6e}, "
                  f"gamma=({gamma_values[0]:+.6f}, {gamma_values[1]:+.6f})", flush=True)
            synchronize(device)
            begin = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return model, gamma, a, b, training_seconds, history


def freeze_projection(model, gamma, map_name, a, b, nodes):
    parameter = next(model.parameters())
    x, weights = simpson_grid(nodes, parameter.device, parameter.dtype)
    with torch.no_grad():
        raw = model(x)
        m0, correction = projection_coefficients(raw, x, weights)
        coefficients = map_coefficients(gamma, map_name)
        if not torch.isfinite(torch.cat((m0.reshape(1), correction.reshape(1), coefficients))).all().item():
            raise FloatingPointError("Nonfinite final cone or projection coefficients")
        u = constrained_values(raw, x, coefficients, a, b, m0, correction)
        projection = {
            "rule": "composite Simpson", "nodes": len(x),
            "m0": float(m0.cpu()), "correction": float(correction.cpu()),
            "moment0": float(integrate(u, weights).cpu()),
            "moment1": float(integrate(x*u, weights).cpu()),
            "dtype": str(parameter.dtype),
        }
    # Evaluate the map BEFORE conversion to float64, preserving the coefficients
    # of the represented function constructed in the training dtype.
    return projection, coefficients.detach().cpu().clone()


def evaluate_fixed(model, projection, coefficients, a, b, nodes, batch_size):
    """Evaluate one fixed projected function; never reproject individual batches."""
    values, derivatives = [], []
    for start in range(0, len(nodes), batch_size):
        x = torch.tensor(np.array(nodes[start:start+batch_size], copy=True),
                         dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
        with torch.enable_grad():
            raw = model(x)
            du = constrained_derivative(raw, x, coefficients, b, projection["correction"], create_graph=False)
        with torch.no_grad():
            u = constrained_values(raw.detach(), x, coefficients, a, b,
                                   projection["m0"], projection["correction"])
        values.append(u.detach().numpy().ravel())
        derivatives.append(du.detach().numpy().ravel())
    return np.concatenate(values), np.concatenate(derivatives)


def evaluate_one(args, model, gamma, map_name, a, b, main_rule, check_rule):
    projection, coefficients = freeze_projection(model, gamma, map_name, a, b, args.projection_nodes)
    evaluation_model = copy.deepcopy(model).cpu().double().eval()
    evaluation_model.requires_grad_(False)
    coefficients64 = coefficients.double()
    x, weights = main_rule
    u, du = evaluate_fixed(evaluation_model, projection, coefficients64, a, b, x, args.eval_batch_size)
    metrics = evaluate_metrics(x, weights, u, du)
    xc, wc = check_rule
    uc, duc = evaluate_fixed(evaluation_model, projection, coefficients64, a, b, xc, args.eval_batch_size)
    check_metrics = evaluate_metrics(xc, wc, uc, duc)
    if not all(math.isfinite(v) for m in (metrics, check_metrics) for v in m.values()):
        raise FloatingPointError("Nonfinite final evaluation metrics")
    algebraic_slack1 = float(coefficients64[1])/2
    algebraic_slack2 = float(coefficients64[0])/6
    metrics.update({
        "slack1": C1-metrics["moment0"],
        "slack2": C2-metrics["moment1"],
        "algebraic_slack1": algebraic_slack1,
        "algebraic_slack2": algebraic_slack2,
        "projection_slack1": C1-projection["moment0"],
        "projection_slack2": C2-projection["moment1"],
        "quadrature_gap_moment0": metrics["moment0"]-projection["moment0"],
        "quadrature_gap_moment1": metrics["moment1"]-projection["moment1"],
        "max_constraint_violation": max(metrics["positive_violation0"], metrics["positive_violation1"]),
        "quadrature_check": {
            "order": args.check_order,
            "absolute_differences": {key: abs(metrics[key]-check_metrics[key]) for key in (
                "relative_l2_u", "relative_l2_du", "moment0", "moment1", "final_objective")},
        },
        "exp_coefficient_underflow": map_name == "exp" and bool((coefficients == 0).any()),
    })
    return metrics, projection, coefficients


def compact_row(record):
    init = record["gamma_init"]
    final = record.get("gamma_final", [None, None])
    hfinal = record.get("coefficients_final", [None, None])
    metrics = record.get("metrics", {})
    row = {"run": record["run"], "group": record["group"], "map": record["map"],
           "status": record["status"], **record["training_settings"],
           "gamma1_init": init[0], "gamma2_init": init[1],
           "initial_algebraic_slack1": record["initial_algebraic_slack1"],
           "initial_algebraic_slack2": record["initial_algebraic_slack2"],
           "gamma1_final": final[0], "gamma2_final": final[1],
           "h_gamma1_final": hfinal[0], "h_gamma2_final": hfinal[1]}
    for key in ("relative_l2_u", "relative_l2_du", "slack1", "slack2",
                "algebraic_slack1", "algebraic_slack2", "projection_slack1", "projection_slack2",
                "max_constraint_violation", "final_objective"):
        row[key] = metrics.get(key)
    row["training_seconds"] = record.get("training_seconds")
    row["failure"] = record.get("failure", "")
    return row


def random_statistics(records, group, requested):
    attempted = [r for r in records if r["group"] == group]
    successful = [r for r in attempted if r["status"] == "success"]
    statistics = {"requested_runs": requested, "successful_runs": len(successful),
                  "failed_runs": len(attempted)-len(successful), "pending_runs": requested-len(attempted)}
    if not successful:
        return statistics
    for key in ("relative_l2_u", "relative_l2_du", "slack1", "slack2"):
        values = np.asarray([r["metrics"][key] for r in successful])
        statistics[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                           "sample_std": float(np.std(values, ddof=1)) if len(values)>1 else None,
                           "minimum": float(values.min()), "maximum": float(values.max())}
    return statistics


def save_results(output, configuration, records):
    rows = [compact_row(r) for r in records]
    if rows:
        with (output/"results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {"configuration": configuration, "runs": records,
               "random_initialization_statistics": {
                   map_name: random_statistics(
                       records, group, configuration["random_initialization"][map_name]["trials"])
                   for group, map_name in RANDOM_GROUPS}}
    with (output/"results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def print_table(records, configuration):
    print("\nFINAL RESULTS: signed slacks s1=C1-integral(u), s2=C2-integral(x*u)")
    print("Exact optimal slacks: s1=1 (inactive), s2=0 (active).")
    print(f"{'Run':<20} {'Epochs':>7} {'gamma1(0)':>10} {'gamma2(0)':>10} {'rel L2(u)':>13} "
          f"{'rel L2(du)':>13} {'slack1':>13} {'slack2':>13}")
    for record in records:
        g = record["gamma_init"]
        epochs = record["training_settings"]["epochs"]
        if record["status"] != "success":
            print(f"{record['run']:<20} {epochs:7d} {g[0]:10.6f} {g[1]:10.6f} FAILED: {record['failure']}")
            continue
        m = record["metrics"]
        print(f"{record['run']:<20} {epochs:7d} {g[0]:10.6f} {g[1]:10.6f} {m['relative_l2_u']:13.6e} "
              f"{m['relative_l2_du']:13.6e} {m['slack1']:13.6e} {m['slack2']:13.6e}")
    for group, map_name in RANDOM_GROUPS:
        count = configuration["random_initialization"][map_name]["trials"]
        statistics = random_statistics(records, group, count)
        print(f"\nRandom {map_name}-map trials: {statistics['successful_runs']}/{count} successful")
        for key in ("relative_l2_u", "relative_l2_du", "slack1", "slack2"):
            if key in statistics:
                s = statistics[key]
                standard_deviation = f"{s['sample_std']:.6e}" if s["sample_std"] is not None else "N/A"
                print(f"  {key}: mean={s['mean']:.6e}, std={standard_deviation}, "
                      f"median={s['median']:.6e}, range=[{s['minimum']:.6e}, {s['maximum']:.6e}]")
    print("Slacks are independently quadrature-evaluated; small negative values are not clipped.")


def plot_error_distribution(records, output, configuration, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {group: [r for r in records if r["group"] == group and r["status"] == "success"]
              for group, _ in RANDOM_GROUPS}
    if not any(groups.values()):
        print("No successful random trial: cannot plot an error distribution.")
        return
    baselines = {r["map"]: r for r in records if r["group"] == "map_comparison" and r["status"] == "success"}
    uniform = configuration["random_initialization"]["square"]
    epochs_by_map = {r["map"]: r["training_settings"]["epochs"] for r in records}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, key, title in zip(axes, ("relative_l2_u", "relative_l2_du"), ("Solution error", "Derivative error")):
        all_values = []
        for position, (group, _), color, fill in zip(
                (1, 2), RANDOM_GROUPS, ("#245782", "#b85d29"), ("#d7e5f4", "#f7dfca")):
            values = np.asarray([r["metrics"][key] for r in groups[group]])
            if not len(values):
                continue
            ax.boxplot([values], positions=[position], widths=0.45, patch_artist=True,
                       showfliers=False,
                       boxprops={"facecolor": fill, "edgecolor": color, "alpha": 0.8},
                       medianprops={"color": color, "linewidth": 2})
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(position+jitter, values, color=color, s=30, zorder=3)
            all_values.extend(values.tolist())
        for name, color, title_name in (("square", "#23856a", "Square"),
                                        ("exp", "#b85d29", "Exponential")):
            if name in baselines:
                gamma1, gamma2 = baselines[name]["gamma_init"]
                label = title_name + rf": $\gamma(0)=({gamma1:g},{gamma2:g})$"
                baseline = baselines[name]["metrics"][key]
                ax.axhline(baseline, color=color, linestyle="--", linewidth=1.4, label=label)
                all_values.append(baseline)
        if min(all_values)>0:
            ax.set_yscale("log")
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([
            f"Square ({epochs_by_map['square']} epochs)\n"
            + rf"$\gamma_i(0)\sim U({uniform['low']:g},{uniform['high']:g})$",
            f"Exponential ({epochs_by_map['exp']} epochs)\n" + r"$\gamma_i(0)\sim\mathcal{N}(0,1)$",
        ])
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.legend(loc="best", fontsize=8)
    counts = configuration["random_initialization"]
    completed = "; ".join(f"{name}: {len(groups[group])}/{counts[name]['trials']} successful"
                          for group, name in RANDOM_GROUPS)
    fig.suptitle(f"Sensitivity to auxiliary initialization\n{completed}", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"random_initialization_errors.{extension}", dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # NUMBER OF RANDOM TRIALS: change these two defaults.
    # Two additional baseline runs use the initial values specified below.
    parser.add_argument("--square-trials", type=int, default=10)
    parser.add_argument("--exp-trials", type=int, default=10)

    # UNIFORM SAMPLING INTERVAL for the random square-map trials.
    parser.add_argument("--uniform-low", type=float, default=0.5)
    parser.add_argument("--uniform-high", type=float, default=1.0)

    # DETERMINISTIC INITIALIZATION: (gamma_1, gamma_2) for each baseline run.
    # These values are trainable after initialization, not held fixed.
    parser.add_argument("--square-gamma-init", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--exp-gamma-init", type=float, nargs=2, default=[0.0, 0.0])

    # EDIT THESE DEFAULTS to configure the square-map experiments.
    parser.add_argument("--square-epochs", type=int, default=5000,
                        help="Training epochs for every square-map run")
    parser.add_argument("--square-lr", type=float, default=1e-3,
                        help="Initial network learning rate for the square map")
    parser.add_argument("--square-gamma-lr", type=float, default=1e-2,
                        help="Initial auxiliary-variable learning rate for the square map")

    # EDIT THESE DEFAULTS to configure the exponential-map experiments.
    parser.add_argument("--exp-epochs", type=int, default=5000,
                        help="Training epochs for every exponential-map run")
    parser.add_argument("--exp-lr", type=float, default=1e-3,
                        help="Initial network learning rate for the exponential map")
    parser.add_argument("--exp-gamma-lr", type=float, default=0.5,
                        help="Initial auxiliary-variable learning rate for the exponential map")

    # Shared architecture, quadrature, reproducibility and output settings.
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Base node count, rounded UP to an odd number for Simpson quadrature")
    parser.add_argument("--grid-jitter", type=int, default=100,
                        help="Add 2*k nodes each epoch, k uniform on {0,...,grid-jitter}")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Common neural-network initialization seed")
    parser.add_argument("--data-seed", type=int, default=0, help="Common training-grid schedule seed")
    parser.add_argument("--gamma-seed", type=int, default=2026, help="Seed for the uniform square-map gamma pairs")
    parser.add_argument("--exp-gamma-seed", type=int, default=2027,
                        help="Seed for the standard-normal exponential-map gamma pairs")
    parser.add_argument("--random-mode", choices=("independent", "shared"), default="independent")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="CPU training/evaluation threads; 0 retains the runtime default")
    parser.add_argument("--projection-nodes", type=int, default=4001)
    parser.add_argument("--eval-intervals", type=int, default=1024)
    parser.add_argument("--eval-order", type=int, default=10)
    parser.add_argument("--check-order", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR/"cnp_sensitivity_results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.uniform_low, args.uniform_high))
            or args.uniform_low >= args.uniform_high):
        parser.error("Uniform bounds must be finite and satisfy uniform-low < uniform-high")
    for name, values in (("square", args.square_gamma_init), ("exp", args.exp_gamma_init)):
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            parser.error(f"{name}-gamma-init must contain two finite values: gamma_1 gamma_2")
    if min(args.square_trials, args.exp_trials) < 1:
        parser.error("Both random trial counts must be positive integers")
    if min(args.square_epochs, args.exp_epochs, args.width, args.blocks, args.eval_batch_size, args.log_every) < 1:
        parser.error("Epochs, architecture dimensions, evaluation batch size, and log interval must be positive")
    if args.n_samples < 3 or args.projection_nodes < 3 or min(args.grid_jitter, args.cpu_threads) < 0:
        parser.error("Invalid quadrature-node, grid-jitter or thread setting")
    rates = (args.square_lr, args.square_gamma_lr, args.exp_lr, args.exp_gamma_lr)
    if not all(math.isfinite(rate) and rate > 0 for rate in rates):
        parser.error("All network and auxiliary learning rates must be finite and positive")
    if min(args.seed, args.data_seed, args.gamma_seed, args.exp_gamma_seed) < 0:
        parser.error("Seeds must be nonnegative")
    main_rule = evaluation_rule(args.eval_intervals, args.eval_order)
    check_rule = evaluation_rule(args.eval_intervals, args.check_order)
    if (args.output/"results.json").exists() and not args.overwrite:
        parser.error("Results already exist; choose a new --output directory or pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        parser.error("Use float32 for MPS, or choose CPU/CUDA for float64")
    seed_torch(args.seed, device)
    base_model = ResNetFunction(1, 1, hidden_dim=args.width, num_layers=args.blocks).to(dtype=dtype)
    base_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
    parameter_count = sum(p.numel() for p in base_model.parameters())+2
    base_nodes = args.n_samples if args.n_samples % 2 else args.n_samples+1
    settings_by_map = {name: training_settings(args, name) for name in ("square", "exp")}
    sampler = random.Random(args.data_seed)
    longest_schedule = [base_nodes+2*sampler.randint(0, args.grid_jitter)
                        for _ in range(max(args.square_epochs, args.exp_epochs))]
    schedules = {name: longest_schedule[:settings["epochs"]]
                 for name, settings in settings_by_map.items()}
    cache = build_training_cache(args, longest_schedule, device, dtype)
    experiments = make_experiments(args)
    configuration = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device_used": str(device), "training_dtype": str(dtype), "optimizer": "Adam", "weight_decay": 0.0,
        "training_by_map": settings_by_map,
        "lr_decay_step": 200, "lr_decay_factor": 0.8, "trainable_parameters_including_gamma": parameter_count,
        "problem": {"alpha": ALPHA, "c1": C1, "c2": C2, "exact_slacks": [1.0, 0.0]},
        "training_quadrature": {"rule": "composite Simpson", "min_nodes": base_nodes,
                                "max_nodes": base_nodes+2*args.grid_jitter,
                                "schedule_note": "Each map uses the first N epochs of one common grid sequence",
                                "schedule_sha256_by_map": {
                                    name: hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
                                    for name, schedule in schedules.items()}},
        "evaluation_quadrature": {"rule": "composite Gauss-Legendre", "intervals": args.eval_intervals,
                                  "order": args.eval_order, "check_order": args.check_order,
                                  "points": len(main_rule[0]), "sha256": rule_fingerprint(*main_rule),
                                  "device": "cpu", "dtype": "float64"},
        "timing_scope": "Synchronized training only; excludes model initialization, logging, final evaluation and saving",
        "comparison_design": "Identical initial network weights and common grid sequence; map, auxiliary initialization, and configured per-map learning rates/epochs may differ",
        "random_initialization": {
            "square": {"distribution": "uniform", "low": args.uniform_low, "high": args.uniform_high,
                       "trials": args.square_trials, "seed": args.gamma_seed, "mode": args.random_mode},
            "exp": {"distribution": "normal", "mean": 0.0, "standard_deviation": 1.0,
                    "trials": args.exp_trials, "seed": args.exp_gamma_seed, "mode": args.random_mode},
        },
        "fixed_initialization": {"square": list(args.square_gamma_init), "exp": list(args.exp_gamma_init)},
        "map_initialization_note": "Baseline gamma values are configurable and remain trainable; equal gamma need not give equal coefficients or initial slacks",
        "python": platform.python_version(), "pytorch": str(torch.__version__), "platform": platform.platform(),
    }
    print(f"Device: {device}; training dtype: {dtype}; Adam; trainable parameters: {parameter_count:,}")
    print(f"{len(experiments)} runs. Same initial neural weights; each map uses its prefix of the common grid sequence.")
    for name, settings in settings_by_map.items():
        print(f"  {name}: epochs={settings['epochs']}; network LR={settings['network_lr']:g}; "
              f"auxiliary LR={settings['gamma_lr']:g}")
    print("Both learning rates decay by 0.8 every 200 epochs within each run.")
    print(f"Random gamma initialization: {args.random_mode}; both coordinates remain trainable.")
    print(f"  Square: {args.square_trials} trials, Uniform[{args.uniform_low:g},{args.uniform_high:g}], seed {args.gamma_seed}.")
    print(f"  Exponential: {args.exp_trials} trials, Normal(0,1), seed {args.exp_gamma_seed}.")
    print(f"Baseline gamma initialization: square={args.square_gamma_init}; exponential={args.exp_gamma_init}.")
    print("The scalar map converts gamma to cone coefficients; equal gamma need not give equal initial slacks.")
    records = []
    for i, experiment in enumerate(experiments, 1):
        initial_gamma = np.asarray(experiment["gamma_init"], dtype=np.float64)
        initial_h = initial_gamma**2 if experiment["map"] == "square" else np.exp(initial_gamma)
        settings = settings_by_map[experiment["map"]]
        record = {**experiment, "training_settings": dict(settings),
                  "initial_coefficients": initial_h.tolist(),
                  "initial_algebraic_slack1": float(initial_h[1]/2),
                  "initial_algebraic_slack2": float(initial_h[0]/6)}
        print(f"\nRun {i}/{len(experiments)}: {record['run']}; gamma(0)={record['gamma_init']}", flush=True)
        print(f"  epochs={settings['epochs']}; network LR={settings['network_lr']:g}; "
              f"auxiliary LR={settings['gamma_lr']:g}", flush=True)
        model = gamma = None
        try:
            model, gamma, a, b, seconds, history = train_one(
                args, experiment, base_state, schedules[experiment["map"]], cache, device, dtype)
            metrics, projection, coefficients = evaluate_one(args, model, gamma, experiment["map"], a, b, main_rule, check_rule)
            record.update(status="success", gamma_final=gamma.detach().cpu().tolist(),
                          coefficients_final=coefficients.tolist(), training_seconds=seconds,
                          metrics=metrics, projection=projection, training_history=history)
            print(f"  FINAL: relL2(u)={metrics['relative_l2_u']:.6e}; relL2(du)={metrics['relative_l2_du']:.6e}; "
                  f"slack1={metrics['slack1']:+.6e}; slack2={metrics['slack2']:+.6e}", flush=True)
            print(f"  gamma(final)={record['gamma_final']}; h(gamma)={record['coefficients_final']}")
            if metrics["exp_coefficient_underflow"]:
                print("  WARNING: an exponential coefficient underflowed to zero in floating-point arithmetic.")
            if args.save_models:
                model_dir = args.output/"models"
                model_dir.mkdir(exist_ok=True)
                torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "gamma": gamma.detach().cpu(), "map": experiment["map"], "coefficients": coefficients,
                            "projection": projection, "a": a, "b": b, "training_settings": dict(settings),
                            "architecture": {"input_dim": 1, "output_dim": 1, "hidden_dim": args.width,
                                             "num_layers": args.blocks}, "configuration": configuration, "metrics": metrics},
                           model_dir/f"{record['run']}.pth")
        except FloatingPointError as error:
            record.update(status="failed", failure=str(error))
            print(f"  FAILED: {error}", flush=True)
        records.append(record)
        save_results(args.output, configuration, records)
        del model, gamma
        gc.collect()
    print_table(records, configuration)
    plot_error_distribution(records, args.output, configuration, args.show)
    print(f"\nSaved results.csv, results.json and distribution figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()

