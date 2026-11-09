#!/usr/bin/env python3
"""Example 5.1: standalone direct constrained P1 finite-element method.

Run:
    python classical.py
Dependencies: numpy, scipy, threadpoolctl. No companion source file is needed.

The default uniform P1 meshes have 64, 128, 256, 512, and 1024 elements.
Run selected meshes: python classical.py --n-elements 128 256 512
Run one mesh:       python classical.py --n-elements 1024
All nodal values
are free (natural Neumann boundary data). The two inequality constraints are
handled by checking four active sets, using one sparse LU factorization and
one block solve with three right-hand sides per solve.

Final errors and objective use the same 10,240 Gauss points and weights as
CNP.py and penalty.py, with a 5,120-point quadrature check. Timing reports the
median of 11 fresh assembly/solve repetitions after a warm-up, with one BLAS
thread. Evaluation, validation, startup and saving are excluded from timing.

All results are saved in ONE file: data/classical_example2.npz.
Read it with np.load(path, allow_pickle=False). All mesh results are obtained
from json.loads(data["results_json"].item()). Arrays named mesh_64_*, mesh_128_*,
etc. retain each mesh's solution and timings. Unprefixed solution arrays and
metrics_json describe the finest mesh for compatibility with the prior script.
The 1000-point x_test/u_pred/u_deriv arrays are plotting data; use the stored
metrics for final comparisons. Every mesh uses the same final integration rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits


# The following problem and evaluation definitions are identical in all three scripts.
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


BOUNDS = np.array([C1, C2], dtype=np.float64)


def assemble(n, quadrature_order=16):
    """Form 0.5*v.T*H*v-r.T*v, with exact mass/stiffness/moment matrices."""
    h = 1.0/n
    nodes = np.linspace(0.0, 1.0, n+1)
    mass_diagonal = np.full(n+1, 2*h/3)
    mass_diagonal[[0, -1]] = h/3
    stiffness_diagonal = np.full(n+1, 2/h)
    stiffness_diagonal[[0, -1]] = 1/h
    mass = diags([np.full(n, h/6), mass_diagonal, np.full(n, h/6)],
                 [-1, 0, 1], format="csc")
    stiffness = diags([np.full(n, -1/h), stiffness_diagonal, np.full(n, -1/h)],
                      [-1, 0, 1], format="csc")
    H = 2*(mass+ALPHA*stiffness)
    C = np.zeros((2, n+1))
    C[0, :-1] += h/2
    C[0, 1:] += h/2
    C[1, :-1] += h*(nodes[:-1]/2+h/6)
    C[1, 1:] += h*(nodes[:-1]/2+h/3)
    z, weights = leggauss(quadrature_order)
    t, weights = (z+1)/2, weights/2
    target = desired_solution(nodes[:-1, None]+h*t[None, :])
    load = np.zeros(n+1)
    load[:-1] += h*np.sum(target*((1-t)*weights)[None, :], axis=1)
    load[1:] += h*np.sum(target*(t*weights)[None, :], axis=1)
    return nodes, H, 2*load, C


def solve_four_active_sets(H, r, C, tolerance=1e-9):
    """One sparse LU, one block solve with three RHS, four active-set checks."""
    factor = splu(H)
    solved = factor.solve(np.column_stack((r, C.T)))
    free, basis = solved[:, 0], solved[:, 1:]
    schur = C@basis
    candidates = []
    for active in ((), (0,), (1,), (0, 1)):
        multiplier = np.zeros(2)
        if active:
            indices = list(active)
            multiplier[indices] = np.linalg.solve(
                schur[np.ix_(indices, indices)], C[indices]@free-BOUNDS[indices])
        v = free-basis@multiplier
        if np.min(BOUNDS-C@v) >= -tolerance and np.min(multiplier) >= -tolerance:
            candidates.append((v, multiplier, active))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one admissible KKT candidate; found {len(candidates)}.")
    return candidates[0]


def evaluate_p1(nodes, coefficients, x):
    """Evaluate the function and its exact element slope on the common grid."""
    element = np.clip(np.searchsorted(nodes, x, side="right")-1, 0, len(nodes)-2)
    u = np.interp(x, nodes, coefficients)
    du = (np.diff(coefficients)/np.diff(nodes))[element]
    return u, du


def timing_summary(records):
    summary = {}
    for phase in ("assembly", "solve", "total"):
        values = 1000*np.array([record[phase+"_seconds"] for record in records])
        for label, value in (("median", np.median(values)),
                             ("q25", np.quantile(values, .25)),
                             ("q75", np.quantile(values, .75))):
            summary[f"{phase}_{label}_ms"] = float(value)
    return summary


def diagnostics(nodes, coefficients, multipliers, H, r, C, x, weights):
    """Evaluate common metrics and independently verify the discrete KKT system."""
    u, du = evaluate_p1(nodes, coefficients, x)
    metrics = evaluate_metrics(x, weights, u, du)
    discrete_moments = C@coefficients
    slacks = BOUNDS-discrete_moments
    stationarity = H@coefficients-r+C.T@multipliers
    metrics.update({
        "discrete_moment0": float(discrete_moments[0]),
        "discrete_moment1": float(discrete_moments[1]),
        "constraint_1_slack": float(slacks[0]),
        "constraint_2_slack": float(slacks[1]),
        "max_constraint_violation": max(metrics["positive_violation0"],
                                        metrics["positive_violation1"]),
        "discrete_max_constraint_violation": max(0.0, -float(np.min(slacks))),
        "multiplier_1": float(multipliers[0]),
        "multiplier_2": float(multipliers[1]),
        "stationarity_linf": float(np.linalg.norm(stationarity, np.inf)),
        "dual_violation": max(0.0, -float(np.min(multipliers))),
        "complementarity_linf": float(np.max(np.abs(multipliers*slacks))),
        "moment_verification_linf": float(np.max(np.abs(
            discrete_moments-np.array([metrics["moment0"], metrics["moment1"]])))),
    })
    return metrics, u, du


def validate(metrics):
    """Reject a failed KKT solve or inconsistent final integration."""
    checks = {
        "max_constraint_violation": 1e-9,
        "discrete_max_constraint_violation": 1e-9,
        "dual_violation": 1e-9,
        "stationarity_linf": 1e-9,
        "complementarity_linf": 1e-9,
        "moment_verification_linf": 1e-11,
        "continuum_objective_identity_error": 1e-9,
    }
    for name, tolerance in checks.items():
        if not np.isfinite(metrics[name]) or metrics[name] > tolerance:
            raise RuntimeError(
                f"Direct FEM: {name}={metrics[name]:.3e} exceeds {tolerance:.3e}.")


def solve_mesh(n_elements, args, x_eval, weights_eval, x_check, weights_check, x_test):
    """Run one mesh using the fixed common evaluation rules."""
    timings = []
    # One complete warm-up. It is not included in the timings.
    nodes, H, r, C = assemble(n_elements, args.load_order)
    coefficients, multipliers, active = solve_four_active_sets(H, r, C)
    warm_metrics, _, _ = diagnostics(nodes, coefficients, multipliers, H, r, C,
                                     x_eval, weights_eval)
    validate(warm_metrics)

    for repeat in range(args.repeats):
        begin = time.perf_counter()
        nodes, H, r, C = assemble(n_elements, args.load_order)
        middle = time.perf_counter()
        coefficients, multipliers, active = solve_four_active_sets(H, r, C)
        end = time.perf_counter()
        timings.append({
            "repeat": repeat,
            "assembly_seconds": middle-begin,
            "solve_seconds": end-middle,
            "total_seconds": end-begin,
        })

    metrics, u_eval, du_eval = diagnostics(
        nodes, coefficients, multipliers, H, r, C, x_eval, weights_eval)
    validate(metrics)
    u_check, du_check = evaluate_p1(nodes, coefficients, x_check)
    check = evaluate_metrics(x_check, weights_check, u_check, du_check)
    checked_keys = ("relative_l2_u", "relative_l2_du", "final_objective", "moment0", "moment1")
    metrics.update({
        "method": "direct",
        "alpha": ALPHA, "c1": C1, "c2": C2,
        "elements": n_elements,
        "degrees_of_freedom": n_elements+1,
        "repetitions": args.repeats,
        "active_constraints": [index+1 for index in active],
        "active_sets_checked": 4,
        "sparse_factorizations": 1,
        "rhs_columns": 3,
        "solver_success": True,
        "solver_message": "Direct KKT verification passed.",
        "evaluation_rule": "composite Gauss-Legendre",
        "evaluation_intervals": args.eval_intervals,
        "evaluation_order": args.eval_order,
        "evaluation_points": len(x_eval),
        "evaluation_device": "cpu", "evaluation_dtype": "float64",
        "rule_sha256": rule_fingerprint(x_eval, weights_eval),
        "quadrature_check": {
            "order": args.check_order,
            "points": len(x_check),
            "rule_sha256": rule_fingerprint(x_check, weights_check),
            "metrics": {key: check[key] for key in checked_keys},
            "absolute_differences": {key: abs(metrics[key]-check[key]) for key in checked_keys},
        },
        **timing_summary(timings),
    })

    u_test, du_test = evaluate_p1(nodes, coefficients, x_test)
    data = {
        "nodes": nodes,
        "coefficients": coefficients,
        "multipliers": multipliers,
        "active_constraints": np.asarray([index+1 for index in active], dtype=np.int64),
        "u_pred": u_test.reshape(-1, 1),
        "u_deriv": du_test.reshape(-1, 1),
        "u_eval": u_eval,
        "du_eval": du_eval,
        "u_check": u_check,
        "du_check": du_check,
        "timing_assembly_seconds": np.array([item["assembly_seconds"] for item in timings]),
        "timing_solve_seconds": np.array([item["solve_seconds"] for item in timings]),
        "timing_total_seconds": np.array([item["total_seconds"] for item in timings]),
    }
    return metrics, data


def print_summary(rows):
    print("\nDirect P1 FEM: common final evaluation for every mesh")
    print(f"{'Elements':>8} {'DOFs':>6} {'RelL2(u)':>13} {'RelL2(du)':>13} "
          f"{'Objective':>16} {'Time(ms)':>10} {'Violation':>11}")
    for row in rows:
        print(f"{row['elements']:8d} {row['degrees_of_freedom']:6d} "
              f"{row['relative_l2_u']:13.6e} {row['relative_l2_du']:13.6e} "
              f"{row['final_objective']:16.12f} {row['total_median_ms']:10.4f} "
              f"{row['max_constraint_violation']:11.3e}")
    print("Time = median assembly + direct solve after warm-up; evaluation and saving excluded.")


def run(args):
    requested = [args.n_elements] if isinstance(args.n_elements, int) else args.n_elements
    meshes = sorted(set(requested))
    if not meshes or min(meshes) < 2 or args.repeats < 1 or args.load_order < 2:
        raise ValueError("Use at least two elements, positive repeats, and load order at least two.")
    x_eval, weights_eval = evaluation_rule(args.eval_intervals, args.eval_order)
    x_check, weights_check = evaluation_rule(args.eval_intervals, args.check_order)
    for n_elements in meshes:
        if args.eval_intervals % n_elements:
            raise ValueError(
                f"Mesh {n_elements}: each --n-elements value must divide --eval-intervals "
                "so every derivative jump lies on an integration-interval boundary.")

    x_test = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
    rows, mesh_data = [], {}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for n_elements in meshes:
            metrics, data = solve_mesh(
                n_elements, args, x_eval, weights_eval, x_check, weights_check, x_test)
            previous = rows[-1] if rows else None
            for key, name in (
                ("relative_l2_u", "observed_order_u"),
                ("relative_l2_du", "observed_order_du"),
                ("objective_gap_to_continuous_optimum", "observed_order_objective_gap"),
            ):
                metrics[name] = (
                    float(np.log(previous[key]/metrics[key])
                          / np.log(n_elements/previous["elements"]))
                    if previous is not None and previous[key] > 0 and metrics[key] > 0
                    else None
                )
            rows.append(metrics)
            mesh_data[n_elements] = data

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": str(np.__version__), "scipy": str(scipy.__version__),
        "platform": platform.platform(),
        "settings": vars(args),
        "mesh_sizes": meshes,
        "threadpool_information_during_timing": pools,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_protocol": (
            "One warm-up per mesh, then fresh assembly and direct solve in each repetition. "
            "Reported total is the median of assembly+solve in each repetition. "
            "Evaluation, validation, imports, startup and saving are excluded. One BLAS thread."),
        "evaluation_note": (
            "Every mesh and both neural methods use the same float64 final evaluation nodes, "
            "weights and relative-error denominators. FE derivatives are exact element slopes."),
        "data_layout": (
            "results_json contains all mesh results in ascending mesh order. "
            "mesh_N_* stores each mesh's numerical arrays. Unprefixed numerical solution "
            "arrays and metrics_json refer to the finest mesh, preserving the previous layout."),
        "plotting_note": "The 1000 equispaced plot points do not define the final integration rule.",
    }
    # Keep every mesh in one NPZ. The common points/reference values are stored once.
    arrays = {
        "mesh_sizes": np.asarray(meshes, dtype=np.int64),
        "finest_mesh": np.int64(meshes[-1]),
        "x_test": x_test.reshape(-1, 1),
        "u_exact": exact_solution(x_test).reshape(-1, 1),
        "u_exa_deri": exact_derivative(x_test).reshape(-1, 1),
        "bounds": BOUNDS,
        "x_eval": x_eval,
        "weights_eval": weights_eval,
        "u_exact_eval": exact_solution(x_eval),
        "du_exact_eval": exact_derivative(x_eval),
        "u_d_eval": desired_solution(x_eval),
        "x_check": x_check,
        "weights_check": weights_check,
        "results_json": json.dumps(rows, indent=2),
        "metrics_json": json.dumps(rows[-1], indent=2),
        "metadata_json": json.dumps(metadata, indent=2),
        **mesh_data[meshes[-1]],
    }
    for n_elements, data in mesh_data.items():
        arrays.update({f"mesh_{n_elements}_{key}": value for key, value in data.items()})
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir/"classical_example2.npz"
    np.savez_compressed(data_path, **arrays)
    print_summary(rows)
    print(f"Saved all mesh results: {data_path.resolve()}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-elements", "--meshes", nargs="+", type=int,
                        default=[64, 128, 256, 512, 1024],
                        help="Element counts; each must divide --eval-intervals.")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--load-order", type=int, default=16)
    parser.add_argument("--eval-intervals", type=int, default=1024)
    parser.add_argument("--eval-order", type=int, default=10)
    parser.add_argument("--check-order", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
