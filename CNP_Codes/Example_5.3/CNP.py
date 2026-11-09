"""Example 5.3: CNP training with reproducible benchmark diagnostics.

Default training preserves the supplied model and objective: a width-64 ResNet
with three residual blocks, 20,000 Adam steps, and an endpoint-inclusive
tensor grid with 51--100 points per coordinate at each step. Training integrals
use tensor-product composite trapezoidal quadrature on this existing grid.
Training uses explicit float32 tensors, including on Apple MPS.
Reusable training grids, targets, and quadrature weights are cached without
autograd graphs (at most 50 grids; about 4.51 MiB with the default settings).
One optimizer step still uses the entire original grid, with the same random
grid sequence, objective, Adam settings, and learning-rate schedule.
Per-epoch scalar histories are buffered on the device and transferred every
1,000 epochs (or at the final epoch), avoiding per-step host synchronization.
Non-finite losses are detected when that buffer is transferred.
Progress messages also report elapsed training time and the average time per
completed epoch. A final summary separates training, evaluation, checkpoint
writing, and total workflow time. Accelerator timing is synchronized.

Final evaluation first copies the model to CPU and only then converts that
copy to float64; no float64 tensor is sent to MPS. It uses the same quadrature/FD diagnostic
definitions as the companion classical_copy.py. The control remains the actual
automatic-differentiation value -Delta(y); it is not replaced by an FD value.

Examples:
    python state_bound_contraint_2.py
    python state_bound_contraint_2.py --checkpoint data/CNP_state_bound.pth
    python state_bound_contraint_2.py --epochs 100 --eval-cells 32 --skip-check

Exactly one data file is written: data/CNP_state_bound.pth by default.
The old Y_pred and U_pred plotting keys (256 by 256) are retained. A full
model checkpoint, final metrics, timings, and diagnostic arrays are added.
Plotting is off by default; --plot-every 200 enables the original periodic
surface plot, whose time is reported separately from training.
Run python Create_figures.py after final evaluation to generate the CNP
and classical surface figures from their saved result files.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn

# BEGIN SHARED NUMERICS (identical in the two standalone scripts)
"""Identical post-processing for Example 5.3 (inlined into both scripts).

The classical solver uses a five-point finite-difference state equation.
Integral comparisons use separately reconstructed state and control fields.
KKT diagnostics are evaluated on each native FD grid, never by applying
the FD operator to a coarse solution interpolated onto a finer grid.

For the Lagrangian J + <p,A y-u>_h + <mu,y-psi>_h, use p=alpha*u
and the nonnegative candidate mu=max(yd-y-A*p,0). Then report the
remaining adjoint, complementarity and primal residuals. Control
stationarity and dual feasibility hold by this recovery; their zeros
are not independent optimality evidence. PDAS native multipliers are
checked separately as well. These are discrete diagnostics, not claims
of pointwise continuous KKT residuals for measure-valued multipliers.
"""

import hashlib
import json
import platform
import sys

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

ALPHA = 0.1
PSI = 0.01
DEFAULT_ACTIVE_TOLERANCES = (1e-5, 1e-4, 1e-3)
PROTOCOL_VERSION = "example53-common-v1"


def desired(points):
    points = np.asarray(points, dtype=np.float64)
    return 10.0 * (np.sin(2.0 * np.pi * points[:, 0]) + points[:, 1])


def evaluation_rule(cells=256, order=2):
    """Tensor composite Gauss rule; 256 cells/order 2 gives 512**2 points."""
    cells, order = int(cells), int(order)
    if cells < 1 or order < 1:
        raise ValueError("Quadrature cells and order must be positive.")
    z, w = np.polynomial.legendre.leggauss(order)
    axis = ((np.arange(cells)[:, None] + (z[None, :] + 1.0) / 2.0)
            / cells).reshape(-1)
    weights_1d = np.tile(w / (2.0 * cells), cells)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    weights = np.multiply.outer(weights_1d, weights_1d).reshape(-1)
    return points, weights


def evaluation_fingerprint(points, weights):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes())
    return digest.hexdigest()


def fd_system(intervals):
    """N intervals, (N-1)**2 interior unknowns, h=1/N."""
    intervals = int(intervals)
    if intervals < 2:
        raise ValueError("At least two intervals per coordinate are required.")
    h = 1.0 / intervals
    n = intervals - 1
    e = np.ones(n)
    t = sp.diags((-e, 2.0 * e, -e), (-1, 0, 1),
                 shape=(n, n), format="csr")
    identity = sp.eye(n, format="csr")
    operator = (sp.kron(identity, t, format="csr")
                + sp.kron(t, identity, format="csr")) / h**2
    axis = np.arange(1, intervals, dtype=np.float64) * h
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return operator.tocsr(), points, h


def reconstruct_fd(intervals, values, points):
    """Bilinear field reconstruction, using zero values at boundary nodes.

    This is also the explicitly chosen reconstruction of the FD control.
    It is not an additional control boundary constraint in the PDE problem.
    The represented control is NOT obtained by differentiating this state
    interpolant. The pair satisfies the native FD equation, not generally
    the continuous PDE after interpolation.
    """
    intervals = int(intervals)
    values = np.asarray(values, dtype=np.float64)
    if values.size == (intervals - 1)**2:
        field = np.zeros((intervals + 1, intervals + 1), dtype=np.float64)
        field[1:-1, 1:-1] = values.reshape(intervals - 1, intervals - 1)
    elif values.shape == (intervals + 1, intervals + 1):
        field = values
    else:
        raise ValueError("FD values do not match the stated mesh size.")
    axis = np.linspace(0.0, 1.0, intervals + 1)
    interpolant = RegularGridInterpolator(
        (axis, axis), field, method="linear", bounds_error=True)
    return interpolant(np.asarray(points, dtype=np.float64)).reshape(-1)


def _finite_vectors(*arrays):
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1)
              for value in arrays]
    if len({value.size for value in arrays}) != 1:
        raise ValueError("Metric arrays must have equal lengths.")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Metric arrays contain NaN or infinity.")
    return arrays


def pair_metrics(points, weights, y, u):
    weights, y, u, yd = _finite_vectors(weights, y, u, desired(points))
    if np.any(weights < 0.0) or not np.sum(weights) > 0.0:
        raise ValueError("Positive quadrature weights are required.")
    positive = np.maximum(y - PSI, 0.0)
    tracking = 0.5 * float(np.dot(weights, (y - yd)**2))
    regularization = 0.5 * ALPHA * float(np.dot(weights, u**2))
    return {
        "final_objective": tracking + regularization,
        "tracking_term": tracking,
        "control_regularization_term": regularization,
        "state_violation_linf_at_evaluation_points": float(positive.max()),
        "state_violation_l1_quadrature": float(np.dot(weights, positive)),
        "state_violation_l2_quadrature": float(np.sqrt(np.dot(weights, positive**2))),
        "state_l2_quadrature": float(np.sqrt(np.dot(weights, y**2))),
        "control_l2_quadrature": float(np.sqrt(np.dot(weights, u**2))),
        "minimum_state_at_evaluation_points": float(y.min()),
        "maximum_state_at_evaluation_points": float(y.max()),
        "evaluation_point_count": int(y.size),
        "quadrature_weight_sum": float(np.sum(weights)),
    }


def near_contact(y, tolerance):
    """Numerical contact set; report upper-bound violation independently."""
    if not float(tolerance) > 0.0:
        raise ValueError("The near-contact tolerance must be positive.")
    return np.abs(np.asarray(y, dtype=np.float64).reshape(-1) - PSI) <= tolerance


def compare_fields(y, u, ref_y, ref_u, weights,
                   active_tolerances=DEFAULT_ACTIVE_TOLERANCES):
    y, u, ref_y, ref_u, weights = _finite_vectors(y, u, ref_y, ref_u, weights)
    result = {}
    for name, candidate, reference in (("state", y, ref_y), ("control", u, ref_u)):
        error_squared = float(np.dot(weights, (candidate - reference)**2))
        reference_squared = float(np.dot(weights, reference**2))
        result[f"absolute_l2_{name}_error_against_reference"] = float(np.sqrt(error_squared))
        result[f"relative_l2_{name}_error_against_reference"] = (
            float(np.sqrt(error_squared / reference_squared))
            if reference_squared > 0.0 else None)
    contact_results = {}
    for tolerance in active_tolerances:
        mask = near_contact(y, float(tolerance))
        reference_mask = near_contact(ref_y, float(tolerance))
        union = float(np.dot(weights, mask | reference_mask))
        intersection = float(np.dot(weights, mask & reference_mask))
        contact_results[f"{float(tolerance):.12g}"] = {
            "state_tolerance": float(tolerance),
            "candidate_contact_area": float(np.dot(weights, mask)),
            "reference_contact_area": float(np.dot(weights, reference_mask)),
            "symmetric_difference_area": float(np.dot(weights, mask ^ reference_mask)),
            "false_positive_area": float(np.dot(weights, mask & ~reference_mask)),
            "false_negative_area": float(np.dot(weights, ~mask & reference_mask)),
            "intersection_over_union": intersection / union if union > 0.0 else 1.0,
        }
    result["near_contact_comparison"] = contact_results
    return result


def kkt_diagnostics(intervals, y, u, native_multiplier=None):
    """Return metrics, p=alpha*u and the recovered nonnegative multiplier.

    ||v||_h = h*||v||_2. Weak residuals use the inverse zero-Dirichlet
    FD Laplacian. Complementarity is h**2*sum(abs(mu*(psi-y))).
    No multiplier L2 convergence is assumed. For the CNP pair, A*y-u
    includes FD truncation error; its actual AD control is retained.
    """
    operator, points, h = fd_system(intervals)
    y, u = _finite_vectors(y, u)
    if y.size != operator.shape[0]:
        raise ValueError("KKT values must be supplied on the native interior grid.")
    yd = desired(points)
    p = ALPHA * u
    g = y - yd + operator @ p
    recovered_mu = np.maximum(-g, 0.0)
    primal = operator @ y - u
    gap = PSI - y
    positive = np.maximum(-gap, 0.0)
    lu = spla.splu(operator.tocsc())
    state_weak = y - lu.solve(u)

    def l2(value):
        return float(h * np.linalg.norm(value))

    def linf(value):
        return float(np.max(np.abs(value)))

    result = {
        "diagnostic_intervals": int(intervals),
        "diagnostic_interior_points": int(y.size),
        "grid_spacing": h,
        "discrete_pde_residual_l2": l2(primal),
        "discrete_pde_residual_linf": linf(primal),
        "discrete_pde_residual_weak_l2": l2(state_weak),
        "state_violation_linf_at_native_nodes": float(positive.max()),
        "state_violation_l2_at_native_nodes": l2(positive),
        "control_stationarity_l2_by_recovery": 0.0,
        "native_unpenalized_interior_objective":
            0.5 * h**2 * float(np.dot(y - yd, y - yd) + ALPHA * np.dot(u, u)),
    }

    def add_multiplier_metrics(prefix, mu):
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        if mu.size != y.size or not np.all(np.isfinite(mu)):
            raise ValueError("Invalid multiplier values.")
        adjoint = g + mu
        weak_adjoint = lu.solve(adjoint)
        complementarity = h**2 * float(np.sum(np.abs(mu * gap)))
        dual_violation = float(np.maximum(-mu, 0.0).max())
        result.update({
            f"{prefix}_adjoint_residual_l2": l2(adjoint),
            f"{prefix}_adjoint_residual_linf": linf(adjoint),
            f"{prefix}_adjoint_residual_weak_l2": l2(weak_adjoint),
            f"{prefix}_complementarity_l1": complementarity,
            f"{prefix}_complementarity_linf": linf(mu * gap),
            f"{prefix}_dual_violation_linf": dual_violation,
            f"{prefix}_multiplier_mass": h**2 * float(np.sum(mu)),
            f"{prefix}_kkt_max_residual": max(
                l2(state_weak), l2(weak_adjoint), float(positive.max()),
                complementarity, dual_violation),
        })

    add_multiplier_metrics("recovered", recovered_mu)
    if native_multiplier is not None:
        add_multiplier_metrics("native", native_multiplier)
    return result, p, recovered_mu


def machine_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "protocol_version": PROTOCOL_VERSION,
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def dumps_json(value, **kwargs):
    return json.dumps(value, default=json_default, allow_nan=False, **kwargs)

# END SHARED NUMERICS


ARCHITECTURE = {"input_dim": 2, "output_dim": 1, "hidden_dim": 64, "num_layers": 3}


class FCResBlock(nn.Module):
    def __init__(self, dim, activation, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(dim, dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x):
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out + identity)


class ResNetFunction(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=64, num_layers=3,
                 activation_fn=nn.Tanh, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.activation = activation_fn()
        self.input_layer = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.blocks = nn.Sequential(
            *[FCResBlock(hidden_dim, self.activation, dtype=dtype, device=device)
              for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=dtype, device=device)
        self.architecture = dict(input_dim=input_dim, output_dim=output_dim,
                                 hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x):
        original = x
        value = self.activation(self.input_layer(x))
        value = self.blocks(value)
        value = self.output_layer(value)
        boundary_factor = ((1.0 - original[:, 0:1]) * original[:, 0:1]
                           * (1.0 - original[:, 1:2]) * original[:, 1:2])
        return value * boundary_factor


def state_from_model(model, x):
    """Keep the supplied square parameterization unchanged."""
    shifted_network = model(x) + math.sqrt(PSI)
    return PSI - shifted_network.square()


def desired_torch(x):
    return 10.0 * (torch.sin(2.0 * math.pi * x[:, 0:1]) + x[:, 1:2])


def laplacian(y, x, create_graph=True):
    # All three vector-Jacobian products have the same (number_of_points, 1)
    # cotangent; reuse it rather than allocating it three times.
    ones = torch.ones_like(y)
    first = torch.autograd.grad(y, x, ones,
                                create_graph=True, retain_graph=True)[0]
    terms = []
    for coordinate in range(2):
        component = first[:, coordinate:coordinate + 1]
        second = torch.autograd.grad(
            component, x, ones,
            create_graph=create_graph,
            retain_graph=(create_graph or coordinate == 0),
        )[0][:, coordinate:coordinate + 1]
        terms.append(second)
    return terms[0] + terms[1]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed=42, numpy_seed=45, grid_seed=42):
    random.seed(grid_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            and hasattr(torch.mps, "manual_seed")):
        torch.mps.manual_seed(seed)


def torch_machine_info(device):
    result = machine_info()
    result.update(torch=str(torch.__version__), device=str(device),
                  torch_threads=int(torch.get_num_threads()),
                  torch_interop_threads=int(torch.get_num_interop_threads()))
    if device.type == "cuda":
        result["accelerator"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        result["accelerator"] = "Apple Metal Performance Shaders"
    return result


def _tensor_grid(points_per_axis, device=None, dtype=torch.float32):
    axis = torch.linspace(0.0, 1.0, points_per_axis, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def generate_data(n_samples=50, device=None, dtype=torch.float32):
    """n_samples is the original base count PER AXIS, not the total count."""
    points_per_axis = int(n_samples) + random.randint(1, 50)
    return _tensor_grid(points_per_axis, device, dtype), points_per_axis


def _trapezoidal_weights(n, device, dtype):
    w = torch.ones(n, device=device, dtype=dtype)
    w[0] = 0.5
    w[-1] = 0.5
    return torch.outer(w, w) / (n - 1)**2


def _cached_training_data(cache, count, device, dtype):
    """Cache only fixed data, never a network output or derivative graph."""
    if count not in cache:
        with torch.no_grad():
            points = _tensor_grid(count, device, dtype)
            cache[count] = (points, desired_torch(points),
                            _trapezoidal_weights(count, device, dtype))
    return cache[count]


def loss_function(y, u, target, weights=None):
    """Tensor trapezoid on the endpoint-inclusive grid from generate_data."""
    n_points = y.shape[0]
    n = math.isqrt(n_points)
    if n < 2 or n * n != n_points:
        raise ValueError("Expected an n-by-n tensor grid.")
    if weights is None:
        weights = _trapezoidal_weights(n, y.device, y.dtype)
    tracking_integrand = (y - target).square().reshape(n, n)
    control_integrand = u.square().reshape(n, n)
    tracking = 0.5 * torch.sum(weights * tracking_integrand)
    regularization = 0.5 * ALPHA * torch.sum(weights * control_integrand)
    return tracking + regularization, tracking, regularization


def make_evaluation_model(model):
    """Make a CPU float64 copy without converting any MPS tensor to float64."""
    # Keep device transfer and dtype conversion separate. The original model
    # and its optimizer stay in float32 on the training device.
    evaluation_model = copy.deepcopy(model).cpu()
    evaluation_model = evaluation_model.double()
    evaluation_model.eval()
    for parameter in evaluation_model.parameters():
        parameter.requires_grad_(False)
    return evaluation_model


def evaluate_points(model, points, batch_size=2048, label=None):
    """Memory-bounded spatial differentiation; returns CPU float64 arrays."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Evaluation points must have shape (number_of_points, 2).")
    parameter = next(model.parameters())
    if parameter.device.type != "cpu" or parameter.dtype != torch.float64:
        raise ValueError(
            "evaluate_points requires the CPU float64 evaluation copy; "
            "call make_evaluation_model(model) first.")
    ys = np.empty(points.shape[0], dtype=np.float64)
    us = np.empty_like(ys)
    if label:
        print(f"{label}: {points.shape[0]:,} points, batches of {batch_size}", flush=True)
    with torch.enable_grad():
        for start in range(0, points.shape[0], batch_size):
            stop = min(start + batch_size, points.shape[0])
            # CPU float64 input already has the required dtype. Share the
            # NumPy slice instead of copying every evaluation batch.
            batch = points[start:stop]
            if not batch.flags.writeable or any(stride < 0 for stride in batch.strides):
                batch = batch.copy()
            x = torch.from_numpy(batch).requires_grad_(True)
            y = state_from_model(model, x)
            u = -laplacian(y, x, create_graph=False)
            ys[start:stop] = y.detach().cpu().numpy().reshape(-1)
            us[start:stop] = u.detach().cpu().numpy().reshape(-1)
            # Release the old spatial graph before constructing the next one.
            del x, y, u
    if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(us)):
        raise FloatingPointError("Non-finite CNP state or control during evaluation.")
    return ys, us


def plotting_grid():
    axis = np.linspace(0.0, 1.0, 256)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return axis, np.column_stack((xx.reshape(-1), yy.reshape(-1)))


def professional_plot(y, u, filename):
    """Optional plot; writes one repeatedly overwritten image only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axis = np.linspace(0.0, 1.0, 256)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    fig = plt.figure(figsize=(16, 6))
    for panel, values, title, cmap in (
        (1, y, "(a) State", "plasma"),
        (2, u, "(b) Control", "viridis"),
    ):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.plot_surface(xx, yy, np.asarray(values).reshape(256, 256),
                        cmap=cmap, rstride=2, cstride=2, alpha=0.9)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Value")
        ax.set_title(title)
        ax.view_init(elev=30, azim=250)
    fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_model(n_samples=50, epochs=20000, lr=1e-3, device=None,
                plot_every=0, batch_size=2048, data_dir=Path("data"),
                log_every=1000):
    if epochs < 1 or log_every < 1:
        raise ValueError("epochs and log_every must be positive")
    device = choose_device("auto") if device is None else torch.device(device)
    synchronize(device)
    setup_start = time.perf_counter()
    # Explicit CPU construction prevents an ambient default dtype/device from
    # creating float64 parameters on MPS. Only float32 parameters move to it.
    model = ResNetFunction(**ARCHITECTURE, dtype=torch.float32, device="cpu").to(device)
    training_dtype = next(model.parameters()).dtype
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    parameters = tuple(model.parameters())
    model.train()
    grid_cache = {}
    scalar_buffer = torch.empty((min(log_every, epochs), 3),
                                device=device, dtype=training_dtype)
    buffered = 0
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    history = {name: [] for name in
               ("loss", "tracking", "regularization", "learning_rate",
                "points_per_axis", "total_training_points")}
    plotting_seconds = 0.0
    synchronize(device)
    loop_start = time.perf_counter()
    for epoch in range(epochs):
        # Exactly the same single Python RNG draw as generate_data().
        count = int(n_samples) + random.randint(1, 50)
        points, target, weights = _cached_training_data(
            grid_cache, count, device, training_dtype)
        # A fresh coordinate leaf shares cached storage; no coordinate copy.
        x = points.detach().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        y = state_from_model(model, x)
        u = -laplacian(y, x, create_graph=True)
        total, tracking, regularization = loss_function(y, u, target, weights)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        # Spatial derivatives are needed, but a final gradient for the input
        # coordinates is not. Accumulate only trainable-parameter gradients.
        total.backward(inputs=parameters)
        optimizer.step()
        scheduler.step()
        scalar_buffer[buffered].copy_(torch.stack(
            (total.detach(), tracking.detach(), regularization.detach())))
        buffered += 1
        history["learning_rate"].append(learning_rate)
        history["points_per_axis"].append(int(count))
        history["total_training_points"].append(int(count**2))
        # Do not keep the completed derivative graph alive into the next step.
        del x, y, u, total, tracking, regularization
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            scalars = scalar_buffer[:buffered].detach().cpu().numpy()
            finite = np.isfinite(scalars).all(axis=1)
            if not finite.all():
                first_bad = epoch + 2 - buffered + int(np.flatnonzero(~finite)[0])
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {first_bad}; "
                    f"detected at the epoch-{epoch + 1} logging boundary.")
            for column, name in enumerate(("loss", "tracking", "regularization")):
                history[name].extend(scalars[:, column].tolist())
            buffered = 0
            synchronize(device)
            elapsed_training = setup_seconds + max(
                0.0, time.perf_counter() - loop_start - plotting_seconds)
            print(f"Epoch {epoch + 1:5d} | loss {history['loss'][-1]:.9g} | "
                  f"tracking {history['tracking'][-1]:.9g} | "
                  f"regularization {history['regularization'][-1]:.9g} | "
                  f"grid {count} x {count} | lr {learning_rate:.3e} | "
                  f"training time {elapsed_training:.3f} s | "
                  f"mean/epoch {elapsed_training / (epoch + 1):.6f} s", flush=True)
        if plot_every > 0 and epoch % plot_every == 0:
            synchronize(device)
            plot_start = time.perf_counter()
            plot_model = make_evaluation_model(model)
            _, points = plotting_grid()
            plot_y, plot_u = evaluate_points(plot_model, points, batch_size)
            professional_plot(plot_y, plot_u, Path(data_dir) / "Professional_3D_View.png")
            del plot_model
            synchronize(device)
            plotting_seconds += time.perf_counter() - plot_start
    synchronize(device)
    loop_seconds = time.perf_counter() - loop_start
    cache_bytes = sum(t.numel() * t.element_size()
                      for entry in grid_cache.values() for t in entry)
    # Gradients are not part of the returned optimizer state or checkpoint.
    optimizer.zero_grad(set_to_none=True)
    training_seconds = setup_seconds + max(0.0, loop_seconds - plotting_seconds)
    timing = {
        "training_setup_seconds": setup_seconds,
        "training_loop_wall_seconds": loop_seconds,
        "periodic_plotting_seconds": plotting_seconds,
        "training_seconds": training_seconds,
        "training_epochs": int(epochs),
        "mean_training_seconds_per_epoch": training_seconds / epochs,
        "cached_training_grids": len(grid_cache),
        "training_grid_cache_bytes": cache_bytes,
        "diagnostic_buffer_bytes": scalar_buffer.numel() * scalar_buffer.element_size(),
        "training_time_definition": (
            "Model/optimizer/buffer initialization and training loop, including lazy "
            "grid caching and buffered scalar logging; "
            "optional periodic evaluation/plotting excluded. Accelerator synchronized "
            "at timing boundaries. Final evaluation and result writing excluded."),
    }
    return model, history, timing, optimizer.state_dict(), scheduler.state_dict()


def _npz_scalar(data, key, default=None):
    return data[key].item() if key in data.files else default


def load_reference(path):
    path = Path(path)
    if not path.is_file():
        return None
    import json
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(_npz_scalar(data, "metadata_json", "{}")))
        if not bool(_npz_scalar(data, "reference_converged", False)):
            raise ValueError(
                "The reference file is not marked reference_converged=True. "
                "Its last iterate must not be used as a reliable PDAS reference. "
                "Complete the classical solves first, or use a missing reference "
                "path to request standalone CNP diagnostics explicitly.")
        mesh_sizes = [int(x) for x in np.asarray(data["mesh_sizes"]).reshape(-1)]
        reference_mesh = int(_npz_scalar(data, "reference_mesh", max(mesh_sizes)))
        required = (f"mesh_{reference_mesh}_y_native", f"mesh_{reference_mesh}_u_native")
        if any(key not in data.files for key in required):
            raise ValueError("Reference NPZ lacks the saved native reference state/control.")
        result = {
            "path": str(path.resolve()), "metadata": metadata,
            "reference_mesh": reference_mesh, "mesh_sizes": mesh_sizes,
            "native_y": np.asarray(data[required[0]], dtype=np.float64).reshape(-1),
            "native_u": np.asarray(data[required[1]], dtype=np.float64).reshape(-1),
            "native_fields": {},
        }
        for mesh in mesh_sizes:
            y_key, u_key = f"mesh_{mesh}_y_native", f"mesh_{mesh}_u_native"
            if y_key in data.files and u_key in data.files:
                result["native_fields"][mesh] = (
                    np.asarray(data[y_key], dtype=np.float64).reshape(-1),
                    np.asarray(data[u_key], dtype=np.float64).reshape(-1),
                )
        for key in ("x_eval", "weights_eval", "reference_y_eval", "reference_u_eval"):
            if key in data.files:
                result[key] = np.asarray(data[key], dtype=np.float64)
    for key, expected in (("alpha", ALPHA), ("psi", PSI)):
        if key in metadata and not np.isclose(float(metadata[key]), expected,
                                             rtol=0.0, atol=1e-14):
            raise ValueError(f"Reference problem parameter {key} does not match this CNP problem.")
    return result


def reference_values(reference, points, weights):
    """Reuse only a compatible saved rule; otherwise reconstruct native fields."""
    metadata = reference["metadata"]
    protocol = metadata.get("protocol_version")
    recorded_fingerprint = metadata.get("evaluation_fingerprint")
    wanted = evaluation_fingerprint(points, weights)
    keys = ("x_eval", "weights_eval", "reference_y_eval", "reference_u_eval")
    if (protocol == PROTOCOL_VERSION and recorded_fingerprint == wanted
            and all(key in reference for key in keys)
            and evaluation_fingerprint(reference["x_eval"], reference["weights_eval"]) == wanted):
        return (reference["reference_y_eval"].reshape(-1),
                reference["reference_u_eval"].reshape(-1), "saved_matching_rule")
    mesh = reference["reference_mesh"]
    return (reconstruct_fd(mesh, reference["native_y"], points),
            reconstruct_fd(mesh, reference["native_u"], points),
            "reconstructed_native_reference_on_current_rule")


def _fd_coordinates(intervals):
    """Native FD coordinates without constructing an unused sparse operator."""
    h = 1.0 / int(intervals)
    axis = np.arange(1, int(intervals), dtype=np.float64) * h
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((xx.reshape(-1), yy.reshape(-1))), h


def evaluate_model(model, reference=None, eval_cells=256, eval_order=2,
                   check_order=4, skip_check=False, batch_size=2048,
                   diagnostic_meshes=(32, 64, 128, 256),
                   active_tolerances=DEFAULT_ACTIVE_TOLERANCES):
    started = time.perf_counter()
    evaluation_model = make_evaluation_model(model)
    training_parameter = next(model.parameters())
    print("Final evaluation on cpu with torch.float64; "
          f"the trained model remains on {training_parameter.device} "
          f"with {training_parameter.dtype}.", flush=True)
    points, weights = evaluation_rule(eval_cells, eval_order)
    y, u = evaluate_points(evaluation_model, points, batch_size, "Final common quadrature")
    metrics = pair_metrics(points, weights, y, u)
    metrics["network_trainable_parameters"] = int(sum(p.numel() for p in model.parameters()))
    metrics["reference_comparison_available"] = reference is not None
    arrays = dict(x_eval=points, weights_eval=weights, y_eval=y, u_eval=u,
                  target_eval=desired(points))
    reference_information = {"available": False, "reason": "Reference NPZ not found."}
    if reference is not None:
        reference_y, reference_u, reuse = reference_values(reference, points, weights)
        comparison = compare_fields(y, u, reference_y, reference_u, weights, active_tolerances)
        metrics.update(comparison)
        reference_information = {
            "available": True, "path": reference["path"],
            "reference_mesh": reference["reference_mesh"], "field_source": reuse,
            "reference_description": "Finest saved regularized PDAS approximation, not an exact solution.",
            "reference_gamma": reference["metadata"].get("gamma"),
            "quadrature_cells_align_with_reference_cells":
                int(eval_cells) % int(reference["reference_mesh"]) == 0,
            "reference_pair_metrics_on_current_rule":
                pair_metrics(points, weights, reference_y, reference_u),
        }
        arrays.update(reference_y_eval=reference_y, reference_u_eval=reference_u)
        arrays["reference_contact_masks_eval"] = {
            f"{float(tolerance):.12g}": near_contact(reference_y, tolerance)
            for tolerance in active_tolerances}
    arrays["contact_masks_eval"] = {
        f"{float(tolerance):.12g}": near_contact(y, tolerance)
        for tolerance in active_tolerances}

    quadrature_check = {"performed": False}
    if not skip_check:
        check_points, check_weights = evaluation_rule(eval_cells, check_order)
        check_y, check_u = evaluate_points(evaluation_model, check_points, batch_size,
                                           "Higher-order quadrature check")
        check_metrics = pair_metrics(check_points, check_weights, check_y, check_u)
        quadrature_check = {
            "performed": True, "order": int(check_order), "metrics": check_metrics,
            "fingerprint": evaluation_fingerprint(check_points, check_weights),
            "absolute_objective_difference":
                abs(check_metrics["final_objective"] - metrics["final_objective"]),
            "relative_objective_difference":
                abs(check_metrics["final_objective"] - metrics["final_objective"])
                / max(abs(check_metrics["final_objective"]), np.finfo(float).tiny),
        }
        arrays.update(x_check=check_points, weights_check=check_weights,
                      y_check=check_y, u_check=check_u)
        if reference is not None:
            reference_y, reference_u, _ = reference_values(reference, check_points, check_weights)
            quadrature_check["reference_comparison"] = compare_fields(
                check_y, check_u, reference_y, reference_u, check_weights, active_tolerances)

    kkt_by_mesh = {}
    for mesh in diagnostic_meshes:
        native_points, h = _fd_coordinates(mesh)
        native_y, native_u = evaluate_points(evaluation_model, native_points, batch_size,
                                             f"FD diagnostic grid N={mesh}")
        diagnostics, recovered_p, recovered_mu = kkt_diagnostics(mesh, native_y, native_u)
        diagnostics["discrete_pde_residual_includes_fd_truncation_error"] = True
        if reference is not None and mesh in reference["native_fields"]:
            ref_y, ref_u = reference["native_fields"][mesh]
            diagnostics["comparison_with_pdas_on_same_native_grid"] = compare_fields(
                native_y, native_u, ref_y, ref_u,
                np.full(native_y.size, h**2), active_tolerances)
        kkt_by_mesh[str(mesh)] = diagnostics
        arrays.update({
            f"mesh_{mesh}_coordinates": native_points,
            f"mesh_{mesh}_y_sampled": native_y,
            f"mesh_{mesh}_u_sampled": native_u,
            f"mesh_{mesh}_p_recovered": recovered_p,
            f"mesh_{mesh}_mu_recovered": recovered_mu,
            f"mesh_{mesh}_contact_masks": {
                f"{float(tolerance):.12g}": near_contact(native_y, tolerance)
                for tolerance in active_tolerances},
        })

    axis, plot_points = plotting_grid()
    plot_y, plot_u = evaluate_points(evaluation_model, plot_points, batch_size,
                                    "Legacy 256 x 256 plotting arrays")
    plot_y = plot_y.reshape(256, 256)
    plot_u = plot_u.reshape(256, 256)
    boundary = np.concatenate((plot_y[0], plot_y[-1], plot_y[:, 0], plot_y[:, -1]))
    metrics["boundary_state_residual_linf"] = float(np.abs(boundary).max())
    arrays.update(Y_pred=plot_y, U_pred=plot_u, plot_axis=axis)
    metadata = {
        "protocol_version": PROTOCOL_VERSION, "alpha": ALPHA, "psi": PSI,
        "evaluation_cells_per_axis": int(eval_cells),
        "evaluation_gauss_order": int(eval_order),
        "evaluation_point_count": int(points.shape[0]),
        "evaluation_fingerprint": evaluation_fingerprint(points, weights),
        "evaluation_dtype": "float64", "evaluation_device": "cpu",
        "diagnostic_meshes": [int(mesh) for mesh in diagnostic_meshes],
        "active_tolerances": [float(tolerance) for tolerance in active_tolerances],
        "control_definition": "Actual automatic-differentiation control -Delta(y).",
        "kkt_definition": (
            "Common discrete FD diagnostics using p=alpha*u and "
            "mu=max(yd-y-A*p,0). The recovered control-stationarity and "
            "dual-feasibility zeros are identities, not independent optimality tests. "
            "The FD state residual for CNP includes truncation error. "
            "These are not pointwise continuous KKT residuals."),
    }
    return {
        "metrics": metrics, "arrays": arrays, "kkt_by_mesh": kkt_by_mesh,
        "quadrature_check": quadrature_check, "reference": reference_information,
        "evaluation_metadata": metadata,
        "final_evaluation_seconds": time.perf_counter() - started,
    }


def checkpoint_safe(value):
    """Use tensors/basic Python types so this script's file supports weights_only loading."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def rng_state():
    numpy_state = np.random.get_state()
    result = {
        "python_random": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            and hasattr(torch.mps, "get_rng_state")):
        result["torch_mps"] = torch.mps.get_rng_state()
    return result


def save_checkpoint_atomic(payload, path):
    """Keep the previous checkpoint intact if serialization is interrupted."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as holder:
        temporary = Path(holder.name)
    try:
        torch.save(checkpoint_safe(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return time.perf_counter() - started


def load_checkpoint(path):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("This script requires a PyTorch version supporting "
                           "torch.load(..., weights_only=True).") from error
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError(
            "This checkpoint has no model weights. The original file saved only "
            "Y_pred/U_pred arrays, which cannot recover the neural function or its "
            "derivatives on new grids. Train once with this revised script, or provide "
            "a full checkpoint containing model_state_dict.")
    for key, expected in (("alpha", ALPHA), ("psi", PSI)):
        if key in saved and not np.isclose(float(saved[key]), expected, rtol=0.0, atol=1e-14):
            raise ValueError(f"Checkpoint problem parameter {key} does not match.")
    architecture = dict(saved.get("architecture", ARCHITECTURE))
    if architecture.get("input_dim") != 2 or architecture.get("output_dim") != 1:
        raise ValueError("Checkpoint architecture is not a two-dimensional scalar CNP.")
    state = saved["model_state_dict"]
    dtype = next(value.dtype for value in state.values()
                 if isinstance(value, torch.Tensor) and value.is_floating_point())
    model = ResNetFunction(**architecture, dtype=dtype, device="cpu")
    model.load_state_dict(state, strict=True)
    return model, saved


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="Reevaluate saved weights; do not retrain.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Default: DATA_DIR/CNP_state_bound.pth")
    parser.add_argument("--reference", type=Path,
                        help="Default: DATA_DIR/classical_state_constraint.npz")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--n-samples", "--base-points-per-axis", dest="n_samples", type=int, default=50,
                        help="Training axis has this many points plus a random integer 1..50.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numpy-seed", type=int, default=45)
    parser.add_argument("--grid-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread count.")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Print and transfer scalar diagnostics every N completed epochs (default: 1000).")
    parser.add_argument("--plot-every", type=int, default=0,
                        help="0 disables periodic evaluation/PNG writing (default).")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-cells", type=int, default=256)
    parser.add_argument("--eval-order", type=int, default=2)
    parser.add_argument("--check-order", type=int, default=4)
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip the higher-order quadrature check; recorded explicitly.")
    parser.add_argument("--diagnostic-meshes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--active-tolerances", nargs="+", type=float,
                        default=list(DEFAULT_ACTIVE_TOLERANCES))
    args = parser.parse_args()
    if (args.epochs < 1 or args.n_samples < 2
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("epochs >= 1, n-samples >= 2, and a finite lr > 0 are required.")
    if args.batch_size < 1 or args.eval_cells < 1 or args.eval_order < 1:
        parser.error("batch-size, eval-cells, and eval-order must be positive.")
    if not args.skip_check and args.check_order <= args.eval_order:
        parser.error("check-order must exceed eval-order unless --skip-check is used.")
    if args.plot_every < 0 or args.log_every < 1 or any(mesh < 2 for mesh in args.diagnostic_meshes):
        parser.error("plot-every must be nonnegative, log-every positive, and diagnostic meshes at least 2.")
    if any(not math.isfinite(tolerance) or tolerance <= 0
           for tolerance in args.active_tolerances):
        parser.error("Active-set tolerances must be finite and positive.")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive.")
    args.diagnostic_meshes = sorted(set(args.diagnostic_meshes))
    return args


def main():
    args = parse_args()
    overall_start = time.perf_counter()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or args.data_dir / "CNP_state_bound.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = args.reference or args.data_dir / "classical_state_constraint.npz"
    # Reject an unconverged reference before starting an expensive training run.
    reference = load_reference(reference_path)
    if reference is None:
        print(f"Reference unavailable: {reference_path}. Standalone objective/feasibility/KKT "
              "diagnostics will be saved; reference errors and contact comparisons are unavailable.",
              flush=True)
    elif args.eval_cells % int(reference["reference_mesh"]) != 0:
        print("WARNING: evaluation quadrature cells do not align with all reference FD cells. "
              "The reconstructed fields remain evaluable, but this may affect quadrature "
              "accuracy. Use --eval-cells equal to a multiple of "
              f"{reference['reference_mesh']} and inspect the quadrature check.", flush=True)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    saved = {}
    if args.checkpoint:
        model, saved = load_checkpoint(args.checkpoint)
        history = saved.get("history", {})
        training_timing = saved.get("training_timing", {"training_seconds": None})
        training_metadata = saved.get("training_metadata", {})
        optimizer_state = saved.get("optimizer_state_dict")
        scheduler_state = saved.get("scheduler_state_dict")
        training_rng_state = saved.get("rng_state_after_training")
        print(f"Reevaluating weights from {args.checkpoint}; training dtype preserved.", flush=True)
    else:
        device = choose_device(args.device)
        seed_everything(args.seed, args.numpy_seed, args.grid_seed)
        print(f"Training on {device} with torch.float32; "
              f"{args.n_samples + 1}--{args.n_samples + 50} "
              "points per coordinate at each epoch.", flush=True)
        model, history, training_timing, optimizer_state, scheduler_state = train_model(
            args.n_samples, args.epochs, args.lr, device, args.plot_every,
            args.batch_size, args.data_dir, log_every=args.log_every)
        training_metadata = {
            "epochs": args.epochs, "base_points_per_axis": args.n_samples,
            "possible_axis_counts": [args.n_samples + 1, args.n_samples + 50],
            "possible_total_point_counts": [(args.n_samples + 1)**2, (args.n_samples + 50)**2],
            "torch_seed": args.seed, "numpy_seed": args.numpy_seed, "python_grid_seed": args.grid_seed,
            "initial_learning_rate": args.lr, "scheduler_step_size": 500, "scheduler_gamma": 0.8,
            "logging_interval_epochs": args.log_every,
            "finite_loss_check": "All per-epoch loss components checked at each logging boundary and at the final epoch.",
            "training_integration": "Tensor-product composite trapezoidal quadrature on endpoint-inclusive uniform grids.",
            "dtype": str(next(model.parameters()).dtype), "machine": torch_machine_info(device),
        }
        training_rng_state = rng_state()
    trained_epochs = int(training_metadata.get("epochs") or len(history.get("loss", [])))
    training_seconds = training_timing.get("training_seconds")
    mean_training_seconds = (training_seconds / trained_epochs
                             if training_seconds is not None and trained_epochs > 0 else None)
    if not args.checkpoint:
        print(f"Training complete: {training_seconds:.3f} s; "
              f"mean/epoch {mean_training_seconds:.6f} s "
              "(setup included; periodic plotting excluded).", flush=True)
    base_payload = {
        "format_version": 1, "model_state_dict": model.state_dict(),
        "architecture": model.architecture, "activation": "Tanh", "alpha": ALPHA, "psi": PSI,
        "optimizer_state_dict": optimizer_state, "scheduler_state_dict": scheduler_state,
        "rng_state_after_training": training_rng_state,
        "history": history, "loss_history": history.get("loss", []),
        "data_history": history.get("tracking", []), "reg_history": history.get("regularization", []),
        "training_metadata": training_metadata, "training_timing": training_timing,
        "training_seconds": training_timing.get("training_seconds"),
        "run_arguments": vars(args),
    }
    checkpoint_write_seconds = 0.0
    if not args.checkpoint:
        initial_payload = {
            **base_payload, "final_evaluation_complete": False,
            "evaluation_status": "Trained weights saved; final evaluation pending.",
            "checkpoint_write_seconds": None,
        }
        checkpoint_write_seconds = save_checkpoint_atomic(initial_payload, output_path)
        print(f"Trained weights saved to {output_path} before final evaluation "
              f"({checkpoint_write_seconds:.3f} s).", flush=True)
    result = evaluate_model(
        model, reference, args.eval_cells, args.eval_order, args.check_order,
        args.skip_check, args.batch_size, args.diagnostic_meshes, args.active_tolerances)
    timings = {
        "training_seconds": training_timing.get("training_seconds"),
        "training_epochs": trained_epochs,
        "mean_training_seconds_per_epoch": mean_training_seconds,
        "final_evaluation_seconds": result["final_evaluation_seconds"],
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "initial_checkpoint_written_this_run": not bool(args.checkpoint),
        "workflow_seconds_before_final_result_write": time.perf_counter() - overall_start,
        "workflow_mode": "reevaluation" if args.checkpoint else "training_and_evaluation",
        "workflow_time_definition": (
            "Elapsed time for this invocation before the final result-file write, including "
            "the initial checkpoint write (new training only), optional plotting, and final "
            "evaluation. checkpoint_write_seconds refers to that initial write. "
            "In reevaluation mode, the training "
            "time is copied from the original checkpoint, not measured again."),
    }
    payload = {
        **base_payload, "final_evaluation_complete": True,
        "evaluation_status": "Final evaluation complete.",
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "timings": timings,
        "metrics": result["metrics"], "kkt_by_mesh": result["kkt_by_mesh"],
        "quadrature_check": result["quadrature_check"], "reference": result["reference"],
        "evaluation_metadata": result["evaluation_metadata"],
        "evaluation_machine": torch_machine_info(torch.device("cpu")),
        **result["arrays"],
    }
    final_write_seconds = save_checkpoint_atomic(payload, output_path)
    workflow_through_save_seconds = time.perf_counter() - overall_start
    summary = {"output": str(output_path), "metrics": result["metrics"],
               "timings": timings, "reference": result["reference"],
               "final_result_write_seconds_measured_after_save": final_write_seconds,
               "workflow_seconds_through_final_save": workflow_through_save_seconds,
               "quadrature_check": result["quadrature_check"],
               "kkt_by_mesh": result["kkt_by_mesh"]}
    print(dumps_json(summary, indent=2, sort_keys=True), flush=True)
    print("\nCOMPUTATION TIME", flush=True)
    training_label = "Original training (from checkpoint)" if args.checkpoint else "Training"
    if training_seconds is not None:
        print(f"  {training_label}: {training_seconds:.3f} s "
              "(setup included; periodic plotting excluded)", flush=True)
        if mean_training_seconds is not None:
            print(f"  Mean per training epoch: {mean_training_seconds:.6f} s "
                  f"({trained_epochs:,} epochs)", flush=True)
    else:
        print(f"  {training_label}: unavailable in the checkpoint", flush=True)
    if not args.checkpoint and training_timing.get("periodic_plotting_seconds", 0.0) > 0.0:
        print(f"  Periodic plotting: {training_timing['periodic_plotting_seconds']:.3f} s", flush=True)
    print(f"  Final evaluation: {result['final_evaluation_seconds']:.3f} s", flush=True)
    print(f"  Checkpoint writes this run: {checkpoint_write_seconds + final_write_seconds:.3f} s", flush=True)
    print(f"  Total workflow through final save (this run): {workflow_through_save_seconds:.3f} s", flush=True)


if __name__ == "__main__":
    main()
