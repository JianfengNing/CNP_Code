"""Navier--Stokes benchmark: Adam, zero weight decay, 20,000 epochs.

Keep reduced_method.py, KKT_method.py, and CreateFigures.py in one directory.
Run the two training scripts sequentially, then run CreateFigures.py.
Outputs are under results/<method>_adam_seed42 beside these scripts.
Both methods use the same initialization/data seeds and evaluation quadrature.

Efficiency: detached analytic-data cache; reused stream-function derivatives;
weighted microbatch gradients with one Adam step per epoch; batched evaluation;
no plotting inside training. --train-batch-size 0 uses a full grid when memory
permits. Timings exclude setup, error monitoring and saving, all reported
separately. The default evaluation grid has 256 nodes/axis (spacing 1/255).
--selection-note records the author's actual hyperparameter-selection rationale.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

DEFAULT_METHOD = "reduced"
SCRIPT_DIR = Path(__file__).resolve().parent
MU = 0.5
ALPHA = 0.1
FREQUENCY = 8.0


class FCResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(x))) + x)


class ResNetFunction(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.blocks = nn.Sequential(*[FCResBlock(hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.output_layer(self.blocks(torch.tanh(self.input_layer(x))))


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_torch(seed, device):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def build_models(method, width, blocks, seed, device, dtype):
    names = ["state_stream", "pressure"]
    if method == "kkt":
        names += ["adjoint_stream", "adjoint_pressure"]
    models = {}
    for index, name in enumerate(names):
        # Common state/pressure initialization across methods. Data sampling
        # uses an independent random generator and is not affected by this.
        seed_torch(seed + index, device)
        models[name] = ResNetFunction(width, blocks).to(device=device, dtype=dtype)
    return models


def parameter_counts(models):
    counts = {name: sum(p.numel() for p in model.parameters() if p.requires_grad)
              for name, model in models.items()}
    return {"by_network": counts, "total": sum(counts.values()),
            "auxiliary_trainable_scalars": 0,
            "definition": "All requires_grad parameters, including biases"}


def uniform_grid(n, device, dtype):
    axis = torch.linspace(0, 1, n, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def mode_with_derivatives(x, frequency, amplitude):
    """Analytic solenoidal mode, Jacobian [component, coordinate], Laplacian."""
    k = frequency * math.pi
    sx, sy = torch.sin(k*x[:, 0]), torch.sin(k*x[:, 1])
    cx, cy = torch.cos(k*x[:, 0]), torch.cos(k*x[:, 1])
    v = amplitude * torch.stack((sx.square()*sy*cy, -sy.square()*sx*cx), dim=1)
    jac = amplitude*k*torch.stack((
        torch.stack((2*sx*cx*sy*cy, sx.square()*(cy.square()-sy.square())), dim=1),
        torch.stack((-sy.square()*(cx.square()-sx.square()), -2*sy*cy*sx*cx), dim=1)
    ), dim=1)
    lap = amplitude*k*k*torch.stack((
        (4*(cx.square()-sx.square())-2)*sy*cy,
        -(4*(cy.square()-sy.square())-2)*sx*cx), dim=1)
    return v, jac, lap


@torch.no_grad()
def manufactured_data(x):
    """Same frequency-8 manufactured problem as the original AD data routines.

    The unused analytic f()/yd() in the original reduced script used frequency
    one for the adjoint; they must not be substituted for its compute_f/compute_yd.
    """
    a = math.exp(-0.05*MU)
    y, jy, ly = mode_with_derivatives(x, 1.0, a)
    adj, ja, la = mode_with_derivatives(x, FREQUENCY, a-math.exp(-MU))
    u = -adj/ALPHA
    f = -MU*ly + torch.einsum("nij,nj->ni", jy, y) - u
    yd = y + MU*la + torch.einsum("nij,nj->ni", ja, y) \
        - torch.einsum("nji,nj->ni", jy, adj)
    return {"f": f, "yd": yd, "y": y, "u": u}


def gradient(value, x, create_graph=True, retain_graph=None):
    return torch.autograd.grad(value.sum(), x, create_graph=create_graph,
                               retain_graph=retain_graph)[0]


def stream_value(model, x):
    w = torch.sin(math.pi*x[:, 0])*torch.sin(math.pi*x[:, 1])
    return model(x).squeeze(1)*w.square()


def stream_velocity(model, x):
    # Inference for KKT state/control needs only one spatial derivative.
    g = gradient(stream_value(model, x), x, create_graph=False)
    return torch.stack((g[:, 1], -g[:, 0]), dim=1).detach()


def stream_jet(model, x, training=True):
    """Velocity, its Jacobian, and Laplacian from a scalar stream function.

    Delta(curl psi) = curl(Delta psi): one third-derivative call replaces
    four separate third-derivative calls in the original code. Intermediate
    spatial derivatives still require create_graph=True, even in inference.
    """
    g = gradient(stream_value(model, x), x)
    h0 = gradient(g[:, 0], x)
    h1 = gradient(g[:, 1], x)
    glap = gradient(h0[:, 0]+h1[:, 1], x, create_graph=training)
    v = torch.stack((g[:, 1], -g[:, 0]), dim=1)
    jac = torch.stack((h1, -h0), dim=1)
    lap = torch.stack((glap[:, 1], -glap[:, 0]), dim=1)
    if not training:
        return v.detach(), jac.detach(), lap.detach()
    return v, jac, lap


def reduced_fields(models, x, forcing, training=True):
    y, jy, ly = stream_jet(models["state_stream"], x, training)
    gp = gradient(models["pressure"](x), x, create_graph=training)
    u = -MU*ly + torch.einsum("nij,nj->ni", jy, y) + gp - forcing
    return y, u


def loss_terms(method, models, x, data):
    if method == "reduced":
        y, u = reduced_fields(models, x, data["f"])
        # Preserve the original mean over BOTH points and two components.
        # On this unit square this training loss approximates J/2.
        first = 0.5*(y-data["yd"]).square().mean()
        second = 0.5*ALPHA*u.square().mean()
    else:
        y, jy, ly = stream_jet(models["state_stream"], x)
        adj, ja, la = stream_jet(models["adjoint_stream"], x)
        gp = gradient(models["pressure"](x), x)
        ge = gradient(models["adjoint_pressure"](x), x)
        state = -MU*ly + torch.einsum("nij,nj->ni", jy, y) + gp-data["f"]+adj/ALPHA
        adjoint = MU*la + torch.einsum("nij,nj->ni", ja, y) \
            - torch.einsum("nji,nj->ni", jy, adj)-ge+y-data["yd"]
        # Equal, fixed residual weights, as in the original KKT script.
        first = state.square().sum(dim=1).mean()
        second = adjoint.square().sum(dim=1).mean()
    return first+second, first, second


def backward_epoch(method, models, coords, data, batch_size):
    """Accumulate the full-grid gradient, then let the caller take ONE step."""
    n = len(coords)
    chunk = batch_size or n
    terms = torch.zeros(3, device=coords.device, dtype=coords.dtype)
    for start in range(0, n, chunk):
        stop = min(start+chunk, n)
        x = coords[start:stop].detach().clone().requires_grad_(True)
        batch = {key: value[start:stop] for key, value in data.items()}
        values = loss_terms(method, models, x, batch)
        fraction = (stop-start)/n
        (values[0]*fraction).backward()
        # Store detached scalars only; no graphs are retained between chunks.
        terms += torch.stack([v.detach() for v in values])*fraction
        del x, batch, values
    return terms


def cache_training_data(sizes, device, dtype, cache_location):
    cache_device = device if cache_location == "device" else torch.device("cpu")
    cache = {}
    for n in sorted(set(sizes)):
        x = uniform_grid(n, cache_device, dtype)
        data = manufactured_data(x)
        cache[n] = (x, {key: data[key] for key in ("f", "yd")})
    return cache


def evaluate(method, models, device, dtype, n=256, batch_size=2048, save_fields=False):
    """Common post-update errors, using CPU float64 quadrature accumulation."""
    states = {name: model.training for name, model in models.items()}
    flags = [(p, p.requires_grad) for model in models.values() for p in model.parameters()]
    for model in models.values():
        model.eval()
    for p, _ in flags:
        p.requires_grad_(False)
    # CPU coordinates/weights: move only one derivative batch onto the device.
    grid = uniform_grid(n, torch.device("cpu"), dtype)
    w = torch.ones(n, dtype=torch.float64)/(n-1)
    w[0] *= 0.5
    w[-1] *= 0.5
    weights = torch.outer(w, w).reshape(-1)
    totals = torch.zeros(7, dtype=torch.float64)
    legacy = torch.zeros(4, dtype=torch.float64)
    fields = {key: [] for key in ("y", "u", "y_exact", "u_exact")} if save_fields else None
    try:
        for start in range(0, len(grid), batch_size):
            x = grid[start:start+batch_size].to(device).detach().requires_grad_(True)
            data = manufactured_data(x)
            with torch.enable_grad():
                if method == "reduced":
                    y, u = reduced_fields(models, x, data["f"], training=False)
                else:
                    y = stream_velocity(models["state_stream"], x)
                    u = -stream_velocity(models["adjoint_stream"], x)/ALPHA
            yc, uc = y.detach().cpu().double(), u.detach().cpu().double()
            ye, ue = data["y"].cpu().double(), data["u"].cpu().double()
            yd = data["yd"].cpu().double()
            integrands = torch.stack(((yc-ye).square().sum(1), ye.square().sum(1),
                                      (uc-ue).square().sum(1), ue.square().sum(1),
                                      (yc-yd).square().sum(1), uc.square().sum(1),
                                      (ye-yd).square().sum(1)), dim=1)
            totals += (weights[start:start+len(x), None]*integrands).sum(0)
            legacy += integrands[:, :4].sum(0)
            if fields is not None:
                for key, value in zip(fields, (yc, uc, ye, ue)):
                    fields[key].append(value.to(dtype=dtype))
            del x, y, u, data, integrands
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for name, model in models.items():
            model.train(states[name])
    result = {
        "state_relative_l2": math.sqrt(float(totals[0]/totals[1])),
        "control_relative_l2": math.sqrt(float(totals[2]/totals[3])),
        "objective_J": float(0.5*totals[4]+0.5*ALPHA*totals[5]),
        "exact_reference_objective_J": float(0.5*totals[6]+0.5*ALPHA*totals[3]),
        "legacy_equal_weight_state_error": math.sqrt(float(legacy[0]/legacy[1])),
        "legacy_equal_weight_control_error": math.sqrt(float(legacy[2]/legacy[3])),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite evaluation result; inspect training stability.")
    if fields is not None:
        fields = {key: torch.cat(value) for key, value in fields.items()}
        fields["x"] = grid
    return result, fields


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_info(device, dtype):
    name = platform.processor() or platform.machine()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        # macOS exposes the Apple chip here; fallback remains honest if absent.
        try:
            import subprocess
            name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            name = "Apple MPS (chip model not detected)"
    return {"device": str(device), "device_name": name, "dtype": str(dtype),
            "platform": platform.platform(), "python": platform.python_version(),
            "pytorch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda}


def run(args, method=DEFAULT_METHOD):
    started = time.perf_counter()
    device = select_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use float32 or another device.")
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.output) if args.output else SCRIPT_DIR/"results"/f"{method}_adam_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out/"report.json").exists() and not args.overwrite:
        raise FileExistsError(f"{out}/report.json exists; choose another --output or pass --overwrite.")
    models = build_models(method, args.width, args.blocks, args.seed, device, dtype)
    counts = parameter_counts(models)
    # Fixed explicitly in both scripts: no AdamW fallback is present.
    optimizer_name = "adam"
    weight_decay = 0.0
    parameters = [p for model in models.values() for p in model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    generator = random.Random(args.data_seed)
    sizes = [generator.randint(args.grid_min, args.grid_max) for _ in range(args.epochs)]
    cache = cache_training_data(sizes, device, dtype, args.cache)
    cache_bytes = sum(x.numel()*x.element_size()+sum(v.numel()*v.element_size() for v in data.values())
                      for x, data in cache.values())
    schedule_hash = hashlib.sha256(json.dumps(sizes).encode()).hexdigest()
    environment = environment_info(device, dtype)
    synchronize(device)
    setup_seconds = time.perf_counter()-started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Method: {method}; device: {device}; optimizer: {optimizer_name}; weight decay: {weight_decay}")
    print(f"Trainable parameters: {counts['total']:,}; by network: {counts['by_network']}")
    print(f"Training grids: {args.grid_min}--{args.grid_max} nodes/axis; "
          f"cache: {cache_bytes/2**20:.2f} MiB; microbatch: {args.train_batch_size or 'full grid'}")
    training_seconds = 0.0
    monitoring_seconds = 0.0
    training_log, error_history = [], []
    synchronize(device)
    segment_start = time.perf_counter()
    for epoch, n in enumerate(sizes, 1):
        coords, data = cache[n]
        if coords.device != device:
            coords = coords.to(device)
            data = {key: value.to(device) for key, value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        lr_used = optimizer.param_groups[0]["lr"]
        terms = backward_epoch(method, models, coords, data, args.train_batch_size)
        optimizer.step()
        scheduler.step()
        log_now = epoch % args.log_every == 0 or epoch == args.epochs
        eval_now = args.eval_every > 0 and epoch % args.eval_every == 0 and epoch < args.epochs
        if log_now or eval_now:
            synchronize(device)
            training_seconds += time.perf_counter()-segment_start
            monitor_start = time.perf_counter()
            if log_now:
                values = terms.cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError("Nonfinite training loss.")
                training_log.append({"epoch": epoch, "nodes_per_axis": n, "points": n*n,
                                     "learning_rate": lr_used, "loss": values[0],
                                     "first_loss_term": values[1], "second_loss_term": values[2],
                                     "training_seconds": training_seconds})
                print(f"epoch={epoch:6d} loss={values[0]:.6e} lr={lr_used:.3e} "
                      f"training_s={training_seconds:.3f}")
            if eval_now:
                metrics, _ = evaluate(method, models, device, dtype, args.eval_grid, args.eval_batch_size)
                error_history.append({"epoch": epoch, "training_seconds": training_seconds, **metrics})
                print(f"  fixed-grid control error={metrics['control_relative_l2']:.6e}")
            synchronize(device)
            monitoring_seconds += time.perf_counter()-monitor_start
            segment_start = time.perf_counter()
    # Training peak includes optimizer state/cache and optional monitoring, as stated.
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    evaluation_start = time.perf_counter()
    metrics, fields = evaluate(method, models, device, dtype, args.eval_grid, args.eval_batch_size,
                               save_fields=True)
    synchronize(device)
    evaluation_seconds = time.perf_counter()-evaluation_start
    error_history.append({"epoch": args.epochs, "training_seconds": training_seconds, **metrics})
    report = {
        "schema_version": 2, "method": method, "parameters": counts, "environment": environment,
        "configuration": {**vars(args), "optimizer": optimizer_name, "weight_decay": weight_decay,
                          "mu": MU, "alpha": ALPHA, "adjoint_frequency": FREQUENCY,
                          "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8,
                          "initialization": "PyTorch Linear defaults; per-network seed=seed+index",
                          "stopping_rule": "Fixed number of epochs; no early stopping",
                          "kkt_residual_weights": {"state": 1.0, "adjoint": 1.0} if method == "kkt" else None,
                          "penalty_parameters": "No additional PDE/boundary penalty in reduced method; "
                                                "KKT residual weights are fixed at one",
                          "hyperparameter_selection": args.selection_note or
                              "Not supplied by the author; values recorded, selection rationale unknown"},
        "training_grid": {"domain": "[0,1]^2", "distribution": "Uniform integer nodes per axis, inclusive",
                          "minimum_nodes_per_axis": args.grid_min, "maximum_nodes_per_axis": args.grid_max,
                          "quadrature": "Equal node weights, including boundary nodes, as in original scripts",
                          "mean_points_per_epoch": sum(n*n for n in sizes)/len(sizes),
                          "grid_size_histogram": {str(n): sizes.count(n) for n in sorted(set(sizes))},
                          "schedule_sha256": schedule_hash, "cache_bytes": cache_bytes,
                          "microbatch_semantics": "Weighted accumulation; one optimizer/scheduler step per full grid"},
        "evaluation": {"nodes_per_axis": args.eval_grid, "points": args.eval_grid**2,
                       "spacing": 1/(args.eval_grid-1), "includes_boundary": True,
                       "quadrature": "Tensor-product composite trapezoidal rule on [0,1]^2",
                       "relative_error": "sqrt(sum_i q_i |v_i-v_exact_i|_2^2 / sum_i q_i |v_exact_i|_2^2)",
                       "accumulation": "CPU float64; field/derivative dtype recorded in environment",
                       "legacy_errors": "Unweighted full-grid ratios, matching original evaluation definitions"},
        "timing": {"setup_seconds": setup_seconds, "training_seconds": training_seconds,
                   "mean_epoch_seconds": training_seconds/args.epochs,
                   "monitoring_seconds": monitoring_seconds, "final_evaluation_seconds": evaluation_seconds,
                   "training_definition": "Synchronized wall time; includes transfers from cache, forward/spatial "
                                          "derivatives, backward, optimizer and scheduler; excludes setup/data "
                                          "precomputation, logging, error evaluation, plotting and file writing",
                   "warmup": "No epochs discarded; initial optimizer/kernel setup costs included"},
        "memory": {"cuda_peak_allocated_bytes": peak_bytes,
                   "scope": "CUDA tensor allocation peak during training/monitoring, including cache and optimizer; "
                            "not process RSS or total driver allocation; null on CPU/MPS"},
        "final_metrics": metrics,
    }
    output_start = time.perf_counter()
    write_csv(out/"training_history.csv", training_log)
    write_csv(out/"error_history.csv", error_history)
    summary = {"method": method, "seed": args.seed, "parameters": counts["total"],
               "epochs": args.epochs, "optimizer": optimizer_name, "weight_decay": weight_decay,
               "training_seconds": training_seconds, "mean_epoch_seconds": training_seconds/args.epochs,
               **metrics}
    write_csv(out/"summary.csv", [summary])
    torch.save({"models": {name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
                           for name, model in models.items()}, "configuration": report["configuration"]},
               out/"model_weights.pth")
    # Always save CPU fields for CreateFigures.py. No derivatives are recomputed
    # for plotting, and no live autograd graph is serialized.
    field_payload = {
        **fields, "method": method, "optimizer": "adam", "weight_decay": 0.0,
        "epochs": args.epochs, "seed": args.seed, "nodes_per_axis": args.eval_grid,
        "schema_version": 2, "final_metrics": metrics,
        "quadrature": report["evaluation"]["quadrature"],
        "training_seconds": training_seconds,
        "error_epochs": [row["epoch"] for row in error_history],
        "error_history": [row["control_relative_l2"] for row in error_history],
        "error_training_seconds": [row["training_seconds"] for row in error_history],
    }
    torch.save(field_payload, out/"evaluation_fields.pth")
    report["timing"]["output_seconds_before_report"] = time.perf_counter()-output_start
    report["timing"]["elapsed_seconds_before_report"] = time.perf_counter()-started
    with (out/"report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"\nTraining time: {training_seconds:.3f} s; mean/epoch: {training_seconds/args.epochs:.6f} s")
    print(f"State relative L2 error: {metrics['state_relative_l2']:.6e}")
    print(f"Control relative L2 error: {metrics['control_relative_l2']:.6e}")
    print(f"Saved report and results to {out.resolve()}")
    return report


def make_parser():
    parser = argparse.ArgumentParser(description=f"{DEFAULT_METHOD} Navier--Stokes benchmark")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=("adam",), default="adam",
                        help="Fixed to Adam for the paper comparison")
    parser.add_argument("--weight-decay", type=float, choices=(0.0,), default=0.0,
                        help="Fixed to zero for both methods")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step", type=int, default=500)
    parser.add_argument("--lr-gamma", type=float, default=0.8)
    parser.add_argument("--grid-min", type=int, default=51)
    parser.add_argument("--grid-max", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=2048,
                        help="Microbatch size; 0 uses the full grid (may be faster but needs more memory)")
    parser.add_argument("--eval-grid", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Evaluate on fixed grid this often; 0 means final only")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--cache", choices=("cpu", "device"), default="device")
    parser.add_argument("--threads", type=int, default=0, help="0 retains PyTorch CPU thread default")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-note", default="",
                        help="Author-supplied factual explanation of hyperparameter selection")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    positive = (args.epochs, args.width, args.blocks, args.lr_step, args.eval_batch_size, args.log_every)
    if min(positive) < 1 or args.grid_min < 2 or args.grid_max < args.grid_min or args.eval_grid < 2:
        parser.error("Invalid positive count or grid range")
    if args.eval_grid < 4*FREQUENCY+1:
        parser.error("Use --eval-grid >= 33 to avoid undersampling the frequency-8 reference control")
    if min(args.train_batch_size, args.eval_every, args.threads) < 0 or args.lr <= 0 or args.lr_gamma <= 0:
        parser.error("Invalid batch size, interval, thread count or learning-rate setting")
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error("weight-decay must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
