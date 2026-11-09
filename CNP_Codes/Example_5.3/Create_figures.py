#!/usr/bin/env python3
"""Figures for Example 5.3, using the revised CNP and classical result files.

Run after classical_copy.py and state_bound_contraint.py:
    python Create_figures.py
    python Create_figures.py --diagnostics

The two original surface figures are saved by default. All loading and
plotting run on CPU; no tensor is sent to MPS or CUDA. Quantitative errors
are evaluated with the saved final quadrature, not the surface plotting
grid. The classical reference is a numerical regularized PDAS solution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def as_numpy(value, name):
    """Accept saved CPU tensors and NumPy arrays without selecting a GPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return result


def load_cnp(path):
    if not path.is_file():
        raise FileNotFoundError(f"CNP result not found: {path}. Run state_bound_contraint.py first.")
    import torch
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("Use a PyTorch version supporting torch.load(..., weights_only=True).") from error
    if not isinstance(data, dict):
        raise ValueError("The CNP checkpoint must be a dictionary.")
    if data.get("final_evaluation_complete") is False:
        raise ValueError(
            "The trained weights were saved, but final evaluation is incomplete. "
            "Run state_bound_contraint.py --checkpoint <this .pth file> "
            "--reference <the classical .npz file>, then run Create_figures.py again.")
    if not all(key in data for key in ("Y_pred", "U_pred")):
        raise ValueError("The CNP file has no Y_pred/U_pred plotting arrays. Complete final evaluation first.")
    y = as_numpy(data["Y_pred"], "CNP Y_pred")
    u = as_numpy(data["U_pred"], "CNP U_pred")
    if y.ndim != 2 or y.shape != u.shape or y.shape[0] != y.shape[1] or y.shape[0] < 2:
        raise ValueError("Y_pred and U_pred must be equally sized square two-dimensional arrays.")
    if "plot_axis" in data:
        axis = as_numpy(data["plot_axis"], "plot_axis").reshape(-1)
    else:
        # Original array-only checkpoints used an endpoint-inclusive uniform grid.
        axis = np.linspace(0.0, 1.0, y.shape[0])
        warnings.warn("Legacy checkpoint: assuming the original uniform plotting grid on [0,1].")
    if (axis.size != y.shape[0] or np.any(np.diff(axis) <= 0)
            or not np.isclose(axis[0], 0.0) or not np.isclose(axis[-1], 1.0)
            or np.any((axis < 0.0) | (axis > 1.0))):
        raise ValueError("plot_axis does not describe an increasing grid on [0,1] matching Y_pred.")
    return data, axis, y, u


def load_classical(path):
    if not path.is_file():
        raise FileNotFoundError(f"Classical result not found: {path}. Run classical_copy.py first.")
    with np.load(path, allow_pickle=False) as data:
        if "reference_converged" not in data or not bool(data["reference_converged"].item()):
            raise ValueError("The classical result is not a converged reference. Complete the PDAS solves first.")
        metadata = json.loads(data["metadata_json"].item()) if "metadata_json" in data else {}
        mesh = int(data["reference_mesh"].item())
        sizes = [int(n) for n in data["mesh_sizes"].reshape(-1)]
        native = {}
        for n in sizes:
            keys = (f"mesh_{n}_y_native", f"mesh_{n}_u_native")
            if not all(key in data for key in keys):
                raise ValueError(f"Classical result lacks native state/control for N={n}.")
            values = tuple(as_numpy(data[key], key).reshape(-1) for key in keys)
            if n < 2 or any(value.size != (n - 1)**2 for value in values):
                raise ValueError(f"Native classical arrays do not match N={n}.")
            native[n] = values
        if mesh not in native or mesh != max(sizes):
            raise ValueError("The classical reference mesh is inconsistent with its saved native arrays.")
        result = {"metadata": metadata, "reference_mesh": mesh, "native": native,
                  "rows": json.loads(data["results_json"].item()) if "results_json" in data else []}
    return result


def reconstruct(intervals, values, points):
    """Same separate bilinear state/control reconstruction as the solver."""
    field = np.zeros((intervals + 1, intervals + 1), dtype=np.float64)
    field[1:-1, 1:-1] = values.reshape(intervals - 1, intervals - 1)
    axis = np.linspace(0.0, 1.0, intervals + 1)
    return RegularGridInterpolator((axis, axis), field, method="linear",
                                   bounds_error=True)(points).reshape(-1)


def fingerprint(points, weights):
    digest = hashlib.sha256()
    for array in (points, weights):
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def problem_parameters(cnp, classical):
    parameters = []
    for name, default in (("alpha", 0.1), ("psi", 0.01)):
        cnp_value = float(cnp.get(name, cnp.get("evaluation_metadata", {}).get(name, default)))
        classical_value = float(classical["metadata"].get(name, default))
        if (not np.isfinite(cnp_value) or not np.isfinite(classical_value)
                or not np.isclose(cnp_value, classical_value, rtol=0.0, atol=1e-14)):
            raise ValueError(f"CNP and classical results use different problem parameter {name}.")
        parameters.append(cnp_value)
    return tuple(parameters)


def metrics(points, weights, y, u, alpha, psi):
    target = 10.0 * (np.sin(2.0 * np.pi * points[:, 0]) + points[:, 1])
    violation = np.maximum(y - psi, 0.0)
    return {
        "final_objective": float(0.5 * np.dot(weights, (y - target)**2)
                                 + 0.5 * alpha * np.dot(weights, u**2)),
        "state_violation_linf_at_evaluation_points": float(violation.max()),
        "state_violation_l1_quadrature": float(np.dot(weights, violation)),
        "state_violation_l2_quadrature": float(np.sqrt(np.dot(weights, violation**2))),
    }


def relative_error(candidate, reference, weights):
    denominator = float(np.dot(weights, reference**2))
    return (float(np.sqrt(np.dot(weights, (candidate - reference)**2) / denominator))
            if denominator > 0.0 else None)


def common_comparison(cnp, classical, alpha, psi, tolerance):
    """Recompute against the loaded classical file, never stale saved errors."""
    required = ("x_eval", "weights_eval", "y_eval", "u_eval")
    if not all(key in cnp for key in required):
        print("Legacy CNP file contains plotting arrays only. Surface figures can be made, "
              "but common-quadrature errors/objectives are unavailable. Complete final "
              "evaluation with the revised NN script to obtain them.", flush=True)
        return None
    points = as_numpy(cnp["x_eval"], "x_eval")
    weights, y, u = (as_numpy(cnp[key], key).reshape(-1) for key in required[1:])
    if (points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1
            or any(vector.size != points.shape[0] for vector in (weights, y, u))
            or np.any((points < 0.0) | (points > 1.0)) or np.any(weights <= 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12)):
        raise ValueError("The saved final-evaluation points, weights or values are inconsistent.")
    expected = cnp.get("evaluation_metadata", {}).get("evaluation_fingerprint")
    if expected and expected != fingerprint(points, weights):
        raise ValueError("The saved evaluation grid does not match its metadata fingerprint.")
    mesh = classical["reference_mesh"]
    ry, ru = (reconstruct(mesh, values, points) for values in classical["native"][mesh])
    for key, current in (("reference_y_eval", ry), ("reference_u_eval", ru)):
        if key in cnp:
            previous = as_numpy(cnp[key], key).reshape(-1)
            if previous.shape != current.shape or not np.allclose(previous, current, rtol=1e-10, atol=1e-12):
                print("The loaded classical reference differs from the one previously saved "
                      "with CNP. All errors below are recomputed against the currently loaded file.")
                break
    cells = cnp.get("evaluation_metadata", {}).get("evaluation_cells_per_axis")
    if cells is not None and any(int(cells) % n for n in classical["native"]):
        warnings.warn("Saved evaluation cells do not align with every classical mesh. "
                      "Inspect quadrature accuracy or reevaluate CNP on a common aligned rule.")
    cnp_metrics = metrics(points, weights, y, u, alpha, psi)
    reference_metrics = metrics(points, weights, ry, ru, alpha, psi)
    errors = {"state": relative_error(y, ry, weights), "control": relative_error(u, ru, weights)}
    print(f"\nCommon final quadrature: {weights.size:,} points; weight sum={weights.sum():.15g}")
    print(f"Reference: current regularized PDAS result, N={mesh} (not an exact solution).")
    print(f"{'Metric':48s} {'CNP':>17s} {'PDAS reference':>17s}")
    for key in cnp_metrics:
        print(f"{key:48s} {cnp_metrics[key]:17.10g} {reference_metrics[key]:17.10g}")
    for name, value in errors.items():
        print(f"CNP relative L2 {name} error against reference: "
              + (f"{value:.10g}" if value is not None else "undefined (zero reference norm)"))
    contact, ref_contact = np.abs(y - psi) <= tolerance, np.abs(ry - psi) <= tolerance
    union = float(np.dot(weights, contact | ref_contact))
    intersection = float(np.dot(weights, contact & ref_contact))
    print(f"Near-contact tolerance={tolerance:g}; symmetric-difference area="
          f"{np.dot(weights, contact ^ ref_contact):.8g}; "
          f"IoU={intersection / union if union > 0.0 else 1.0:.8g}")
    print("These integrals use the saved final quadrature. Surface-grid samples are used only for figures.\n")
    return {"points": points, "weights": weights, "reference_y": ry, "reference_u": ru,
            "cnp_metrics": cnp_metrics, "reference_metrics": reference_metrics, "errors": errors}



def print_kkt_residuals(cnp, classical):
    """Print recovered discrete KKT diagnostics for CNP and PDAS on common FD grids.

    CNP diagnostics are postprocessing quantities saved by state_bound_contraint.py.
    PDAS diagnostics are read from results_json saved by classical_copy.py.
    The recovered residual uses the same definition for both methods.
    """
    cnp_kkt = cnp.get("kkt_by_mesh", {})
    if not isinstance(cnp_kkt, dict) or not cnp_kkt:
        print("KKT comparison unavailable: CNP checkpoint has no kkt_by_mesh diagnostics.")
        return

    pdas_by_mesh = {}
    for row in classical.get("rows", []):
        try:
            mesh = int(row.get("intervals_per_coordinate"))
        except (TypeError, ValueError):
            continue
        native = row.get("native_kkt", {})
        if isinstance(native, dict):
            pdas_by_mesh[mesh] = native

    common_meshes = sorted(set(int(k) for k in cnp_kkt) & set(pdas_by_mesh))
    if not common_meshes:
        print("KKT comparison unavailable: no common diagnostic meshes in CNP and PDAS files.")
        return

    print("\nRecovered discrete KKT residuals (same postprocessing definition for CNP and PDAS):")
    print(f"{'N':>5s} {'CNP KKT_max':>16s} {'PDAS KKT_max':>16s} "
          f"{'PDAS native_KKT':>18s}")
    for mesh in common_meshes:
        c = cnp_kkt[str(mesh)]
        p = pdas_by_mesh[mesh]
        cmax = c.get("recovered_kkt_max_residual")
        pmax = p.get("recovered_kkt_max_residual")
        native = p.get("native_kkt_max_residual")
        def fmt(v):
            return f"{float(v):.6e}" if v is not None else "n/a"
        print(f"{mesh:5d} {fmt(cmax):>16s} {fmt(pmax):>16s} {fmt(native):>18s}")

    # Detailed component-wise comparison on the finest common diagnostic mesh.
    mesh = max(common_meshes)
    c = cnp_kkt[str(mesh)]
    p = pdas_by_mesh[mesh]
    fields = (
        ("discrete_pde_residual_weak_l2", "state/PDE weak L2"),
        ("recovered_adjoint_residual_weak_l2", "stationarity weak L2"),
        ("state_violation_linf_at_native_nodes", "state violation Linf"),
        ("recovered_complementarity_l1", "complementarity L1"),
        ("recovered_dual_violation_linf", "dual violation Linf"),
        ("recovered_kkt_max_residual", "KKT max residual"),
    )
    print(f"\nKKT components on finest common diagnostic grid N={mesh}:")
    print(f"{'Component':32s} {'CNP':>16s} {'PDAS':>16s}")
    for key, label in fields:
        cv = c.get(key)
        pv = p.get(key)
        def fmt2(v):
            return f"{float(v):.6e}" if v is not None else "n/a"
        print(f"{label:32s} {fmt2(cv):>16s} {fmt2(pv):>16s}")

    print("CNP KKT quantities are postprocessing diagnostics and are not used in training.")
    print("For CNP, the discrete PDE residual also contains finite-difference truncation error.")


def save_figure(fig, path, dpi, show, plt):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {path}", flush=True)
    if not show:
        plt.close(fig)


def surface_figure(axis, y, u, path, args, plt):
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    fig = plt.figure(figsize=(12, 5))
    for index, field, title in ((1, y, "(a) State"), (2, u, "(b) Control")):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(xx, yy, field, cmap="inferno", rstride=2, cstride=2, alpha=0.9)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)
        ax.view_init(elev=30, azim=250)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    save_figure(fig, path, args.dpi, args.show, plt)


def contact_figure(axis, y, ry, psi, args, plt):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    masks = (np.abs(y - psi) <= args.active_tolerance, np.abs(ry - psi) <= args.active_tolerance)
    comparison = masks[0].astype(int) + 2 * masks[1].astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=False)
    cmap = ListedColormap(["#f3f3f3", "#2a6f97"])
    for ax, field, title in zip(axes[:2], masks, ("CNP near-contact set", "PDAS near-contact set")):
        ax.pcolormesh(axis, axis, field.T.astype(int), shading="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
    colors = ["#f3f3f3", "#d95f02", "#1b9e77", "#343a40"]
    axes[2].pcolormesh(axis, axis, comparison.T, shading="nearest", cmap=ListedColormap(colors),
                       norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], len(colors)))
    axes[2].set_title("Contact classification")
    axes[2].legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(
        colors, ("Neither", "CNP only", "PDAS only", "Both"))], loc="upper right", fontsize=8)
    for ax in axes:
        ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    fig.suptitle(r"Near-contact sets: $|y-\psi|\leq\tau$, " + rf"$\tau={args.active_tolerance:g}$")
    save_figure(fig, args.output_dir / "active_set_comparison.png", args.dpi, args.show, plt)


def training_figure(cnp, args, plt):
    history = cnp.get("history", {})
    loss = as_numpy(history.get("loss", cnp.get("loss_history", [])), "loss_history").reshape(-1)
    if loss.size == 0:
        print("Training-history figure skipped: the checkpoint has no loss history.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, loss.size + 1), loss, linewidth=1.2, label="Training objective")
    for key, label in (("tracking", "Tracking term"), ("regularization", "Control regularization")):
        if key in history:
            values = as_numpy(history[key], key).reshape(-1)
            if values.size == loss.size:
                ax.plot(np.arange(1, loss.size + 1), values, linewidth=1, label=label)
    ax.set(xlabel="Epoch", ylabel="Training quadrature value", title="CNP training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, args.output_dir / "CNP_training_history.png", args.dpi, args.show, plt)


def mesh_figure(classical, comparison, alpha, psi, args, plt):
    if comparison is None or len(classical["native"]) < 2:
        print("Mesh-refinement figure skipped: multiple meshes and common evaluation arrays are required.")
        return
    points, weights = comparison["points"], comparison["weights"]
    ry, ru = comparison["reference_y"], comparison["reference_u"]
    meshes = sorted(classical["native"])
    objectives, state_errors, control_errors = [], [], []
    for mesh in meshes:
        y, u = (reconstruct(mesh, values, points) for values in classical["native"][mesh])
        objectives.append(metrics(points, weights, y, u, alpha, psi)["final_objective"])
        if mesh != classical["reference_mesh"]:
            state_errors.append(relative_error(y, ry, weights))
            control_errors.append(relative_error(u, ru, weights))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    coarse = meshes[:-1]
    for values, label, marker in ((state_errors, "PDAS state", "o"), (control_errors, "PDAS control", "s")):
        valid = [(n, value) for n, value in zip(coarse, values) if value is not None and value > 0.0]
        if valid:
            axes[0].loglog(*zip(*valid), marker + "-", label=label)
    for key, color in (("state", "C0"), ("control", "C1")):
        value = comparison["errors"][key]
        if value is not None and value > 0.0:
            axes[0].axhline(value, color=color, linestyle="--", label=f"CNP {key}")
    axes[0].set(xlabel="Intervals per coordinate N", ylabel="Relative $L^2$ error",
                title=f"Errors against numerical reference (N={meshes[-1]})")
    axes[1].plot(meshes, objectives, "o-", label="PDAS")
    axes[1].axhline(comparison["cnp_metrics"]["final_objective"], color="C1", linestyle="--", label="CNP")
    axes[1].set_xscale("log", base=2)
    axes[1].set(xlabel="Intervals per coordinate N", ylabel="$J(y,u)$", title="Common-quadrature objective")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    save_figure(fig, args.output_dir / "classical_mesh_refinement.png", args.dpi, args.show, plt)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnp", type=Path, default=Path("data/CNP_state_bound.pth"))
    parser.add_argument("--classical", type=Path, default=Path("data/classical_state_constraint.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--diagnostics", action="store_true", help="Also plot near-contact sets, training and mesh refinement.")
    parser.add_argument("--active-tolerance", type=float, default=1e-4)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Show the figures after saving them.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.active_tolerance) or args.active_tolerance <= 0 or args.dpi < 1:
        parser.error("active-tolerance and dpi must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cnp, axis, y, u = load_cnp(args.cnp)
    classical = load_classical(args.classical)
    alpha, psi = problem_parameters(cnp, classical)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    plot_points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    mesh = classical["reference_mesh"]
    ry, ru = (reconstruct(mesh, values, plot_points).reshape(y.shape)
              for values in classical["native"][mesh])
    comparison = common_comparison(cnp, classical, alpha, psi, args.active_tolerance)
    print_kkt_residuals(cnp, classical)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    surface_figure(axis, y, u, args.output_dir / "CNP_state_bound.png", args, plt)
    surface_figure(axis, ry, ru, args.output_dir / "classical_state_bound.png", args, plt)
    if args.diagnostics:
        contact_figure(axis, y, ry, psi, args, plt)
        training_figure(cnp, args, plt)
        mesh_figure(classical, comparison, alpha, psi, args, plt)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
