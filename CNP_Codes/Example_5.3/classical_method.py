#!/usr/bin/env python3
"""Example 5.3: single-file benchmark based on the supplied classical copy.py.

This preserves the algorithm in the supplied ``classical copy.py``:
gamma=1e9, the discrete multiplier shift, and the cold start y=psi.
It solves a shifted Moreau--Yosida regularization, not the exact
inequality-constrained problem at a finite value of gamma. Original-
constraint KKT residuals and regularized equation residuals are both
reported. Every final integral uses the common quadrature protocol.
The optimized implementation reuses the Jacobian's sparse structure and
updates only its diagonal. On small SuperLU systems it also reuses the first
COLAMD column ordering; each changed matrix is still factorized anew. Larger
systems and UMFPACK installations retain the original spsolve path. Float64
arithmetic, cold starts, stopping rules and all parameter defaults are retained.
The final residual/control and the finest-grid reconstruction are reused
instead of being evaluated a second time.

Run:
    python classical_copy_2.py
    python classical_copy_2.py --meshes 16 32 --eval-cells 32

Then train the companion CNP and create the figures:
    python state_bound_contraint_2.py --device mps
    python Create_figures.py

This classical solver runs entirely on the CPU using NumPy/SciPy float64;
it does not create Torch tensors or use MPS.

All definitions are included in this script. Exactly one NPZ result file
contains every mesh, iteration history, metric and a human-readable report.
The consolidated report is also printed. No figures are saved by default.
The default meshes count intervals per coordinate, not interior nodes.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


def _diagonal_positions(matrix):
    """Data offsets of the diagonal in a square CSR or CSC matrix.

    Do not sort the CSR indices: preserving the original row ordering also
    preserves the summation order in the reported linear-system residual.
    """
    major_indices = np.repeat(
        np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
        np.diff(matrix.indptr))
    positions = np.flatnonzero(matrix.indices == major_indices)
    if positions.size != matrix.shape[0]:
        raise ValueError("Expected exactly one stored diagonal entry per row/column")
    return positions


class _FixedPatternSolver:
    """Reuse sparse storage and, on small SuperLU systems, column ordering.

    Reordering the columns only changes the internal unknown numbering.
    The solution is restored to the original numbering before any PDAS
    calculation. Numeric factors are NEVER reused after the diagonal changes.
    The conservative size limit avoids the slowdown observed when reusing
    the ordering on larger systems. If UMFPACK is installed, preserve the
    original backend selection by delegating directly to spsolve.
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.diagonal = _diagonal_positions(matrix)
        self.order = None
        self.reuse_ordering = matrix.shape[0] <= 4096
        if self.reuse_ordering:
            from importlib.util import find_spec
            try:
                self.reuse_ordering = find_spec("scikits.umfpack") is None
            except ModuleNotFoundError:
                pass  # No scikits namespace: spsolve uses SuperLU.
            except (ImportError, ValueError):
                self.reuse_ordering = False  # Preserve an uncertain backend.

    def solve(self, diagonal, rhs):
        if self.order is None:
            self.matrix.data[self.diagonal] = diagonal
            if not self.reuse_ordering:
                return spla.spsolve(self.matrix, rhs)
            factor = spla.splu(self.matrix, permc_spec="COLAMD")
            solution = factor.solve(rhs)
            self.order = np.argsort(factor.perm_c)
            del factor
            self.matrix = self.matrix[:, self.order].tocsc()
            original_columns = np.repeat(self.order, np.diff(self.matrix.indptr))
            self.diagonal = np.flatnonzero(self.matrix.indices == original_columns)
            if self.diagonal.size != diagonal.size:
                raise ValueError("Invalid diagonal after sparse column permutation")
            return solution
        self.matrix.data[self.diagonal] = diagonal[self.order]
        permuted_solution = spla.spsolve(
            self.matrix, rhs, permc_spec="NATURAL", use_umfpack=False)
        solution = np.empty_like(permuted_solution)
        solution[self.order] = permuted_solution
        return solution


def pdas_solve(intervals, gamma=1e9, max_iterations=1080,
               relative_tolerance=1e-8, progress_every=25):
    """One cold solve, including assembly in the reported solve time.

    The density multiplier is used: the common h**2 factor in the
    discrete objective and constraints has been divided out of the
    stationarity equations. Active-set stability alone is insufficient
    for the returned ``converged`` flag; the nonlinear regularized
    stationarity residual must also meet the stated relative tolerance.
    """
    start = time.perf_counter()
    operator, points, h = fd_system(intervals)
    yd = desired(points)
    psi = np.full(yd.size, PSI, dtype=np.float64)
    hessian = (sp.eye(yd.size, format="csr")
               + ALPHA * (operator @ operator)).tocsr()
    shift = np.maximum(yd - hessian @ psi, 0.0)
    # This one addition reproduces the exact CSR ordering of the original
    # per-iteration assembly, including when hessian is not sorted.
    matrix = hessian + sp.diags(np.zeros(yd.size, dtype=np.float64), format="csr")
    linear_solver = _FixedPatternSolver(matrix.tocsc())
    diagonal_csr = _diagonal_positions(matrix)
    base_diagonal = matrix.data[diagonal_csr].copy()
    del hessian
    assembly_seconds = time.perf_counter() - start

    y = psi.copy()
    active = shift + gamma * (y - psi) > 0.0
    data_scale_l2 = max(1.0, float(h * np.linalg.norm(yd)))
    history = []
    converged = False
    stop_reason = "maximum_iterations_reached"
    solve_start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        active_float = active.astype(np.float64)
        active_penalty = gamma * active_float
        diagonal = base_diagonal + active_penalty
        matrix.data[diagonal_csr] = diagonal
        # Keep the arithmetic order of the original right-hand side.
        rhs = yd - active_float * shift + active_penalty * psi
        linear_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            y_new = linear_solver.solve(diagonal, rhs)
        linear_seconds = time.perf_counter() - linear_start
        if not np.all(np.isfinite(y_new)):
            raise FloatingPointError(
                f"PDAS produced non-finite state values on mesh {intervals}.")

        u_new = operator @ y_new
        state_gap = y_new - psi
        shifted_gap = shift + gamma * state_gap
        mu_new = np.maximum(shifted_gap, 0.0)
        active_new = shifted_gap > 0.0
        changed = int(np.count_nonzero(active_new ^ active))
        tracking_difference = y_new - yd
        regularized_residual = tracking_difference + ALPHA * (operator @ u_new) + mu_new
        residual_l2 = float(h * np.linalg.norm(regularized_residual))
        residual_linf = float(np.max(np.abs(regularized_residual)))
        relative_residual = residual_l2 / data_scale_l2
        discrete_objective = 0.5 * h**2 * float(
            np.dot(tracking_difference, tracking_difference) + ALPHA * np.dot(u_new, u_new))
        history.append({
            "iteration": iteration,
            "active_count_before_solve": int(np.count_nonzero(active)),
            "active_count_after_solve": int(np.count_nonzero(active_new)),
            "active_set_changes": changed,
            "linear_solve_seconds": linear_seconds,
            "linear_residual_linf": float(np.max(np.abs(matrix @ y_new - rhs))),
            "regularized_stationarity_l2": residual_l2,
            "regularized_stationarity_linf": residual_linf,
            "regularized_stationarity_relative_l2": relative_residual,
            "native_interior_objective": discrete_objective,
            "native_state_violation_linf": float(np.maximum(state_gap, 0.0).max()),
        })
        y = y_new
        active = active_new

        if progress_every and (iteration % progress_every == 0):
            print(
                f"  N={intervals}, iteration={iteration}, "
                f"active changes={changed}, "
                f"relative regularized residual={relative_residual:.3e}",
                flush=True)
        if changed == 0:
            if relative_residual <= relative_tolerance:
                converged = True
                stop_reason = "stable_active_set_and_regularized_residual"
            else:
                # Solving the identical linear system again will not improve
                # the attainable floating-point accuracy of this fixed point.
                stop_reason = "stable_active_set_but_residual_above_tolerance"
            break

    iteration_seconds = time.perf_counter() - solve_start
    total_seconds = time.perf_counter() - start
    # These arrays already correspond to the final accepted iterate.
    u, mu, final_residual = u_new, mu_new, regularized_residual
    summary = {
        "intervals_per_coordinate": int(intervals),
        "interior_points_per_coordinate": int(intervals - 1),
        "grid_spacing": float(h),
        "state_degrees_of_freedom": int(y.size),
        "control_degrees_of_freedom": int(y.size),
        "independent_reduced_solver_unknowns": int(y.size),
        "gamma": float(gamma),
        "initialization": "cold start: every interior state value equals psi",
        "shift_definition": "max(yd - (I + alpha*A_h*A_h)*psi_vector, 0)",
        "shift_maximum": float(shift.max()),
        "iterations": int(len(history)),
        "maximum_iterations": int(max_iterations),
        "converged_regularized_problem": bool(converged),
        "stopping_reason": stop_reason,
        "active_set_stable": bool(history[-1]["active_set_changes"] == 0),
        "active_set_changes_last_iteration": history[-1]["active_set_changes"],
        "algorithmic_active_count": int(np.count_nonzero(active)),
        "algorithmic_active_area_node_weights": float(h**2 * np.count_nonzero(active)),
        "regularized_residual_relative_tolerance": float(relative_tolerance),
        "regularized_stationarity_l2": float(h * np.linalg.norm(final_residual)),
        "regularized_stationarity_linf": float(np.max(np.abs(final_residual))),
        "regularized_stationarity_relative_l2": float(h * np.linalg.norm(final_residual) / data_scale_l2),
        "assembly_seconds": float(assembly_seconds),
        "iteration_seconds": float(iteration_seconds),
        "assembly_and_solution_seconds": float(total_seconds),
        "timing_excludes": "final common evaluation, KKT diagnostics, plots and file writing",
        "native_interior_objective_note":
            "Used by the native FD problem; not the common-quadrature final objective.",
    }
    return {
        "intervals": int(intervals), "y": y, "u": u, "mu": mu,
        "shift": shift, "active": active, "points": points,
        "summary": summary, "history": history,
    }


def repeated_solve(intervals, args):
    runs = []
    repeat_histories = []
    result = None
    for repetition in range(args.repeats):
        print(f"Solving N={intervals}: cold run {repetition + 1}/{args.repeats}", flush=True)
        result = pdas_solve(
            intervals, gamma=args.gamma, max_iterations=args.max_iterations,
            relative_tolerance=args.relative_tolerance,
            progress_every=args.progress_every if repetition == 0 else 0)
        runs.append(dict(result["summary"]))
        repeat_histories.append(list(result["history"]))
        if not result["summary"]["converged_regularized_problem"]:
            break
    result["repeat_summaries"] = runs
    result["repeat_histories"] = repeat_histories
    result["timing"] = {
        "requested_repetitions": int(args.repeats),
        "completed_repetitions": len(runs),
        "warmup_runs": 0,
        "initialization_each_run": "independent cold start",
        "assembly_seconds_median": float(np.median([r["assembly_seconds"] for r in runs])),
        "iteration_seconds_median": float(np.median([r["iteration_seconds"] for r in runs])),
        "assembly_and_solution_seconds_median": float(np.median([r["assembly_and_solution_seconds"] for r in runs])),
        "assembly_and_solution_seconds_minimum": float(min(r["assembly_and_solution_seconds"] for r in runs)),
        "assembly_and_solution_seconds_maximum": float(max(r["assembly_and_solution_seconds"] for r in runs)),
        "iteration_counts": [int(r["iterations"]) for r in runs],
        "all_runs_converged": bool(all(r["converged_regularized_problem"] for r in runs)),
    }
    return result


def build_report(result_rows, metadata):
    """Human-readable tables stored inside the same NPZ and printed once."""
    lines = [
        "Example 5.3: five-point finite-difference Moreau--Yosida PDAS",
        "Based on the supplied classical copy.py; each mesh uses an independent cold start.",
        f"alpha={metadata['alpha']:g}, psi={metadata['psi']:g}, gamma={metadata['gamma']:.6g}",
        f"Common evaluation points: {metadata['evaluation_point_count']:,}",
        f"Reference mesh: {metadata['reference_mesh']}; converged={metadata['reference_converged']}",
        "The reference is a regularized numerical solution, not an exact continuous solution.",
        "Time includes assembly and PDAS solves; postprocessing and file writing are separate.",
        "Default is one timing run per mesh; with --repeats, the displayed time is the median.",
        "",
        "N       DOFs   iterations converged    time(s)          J(Q)       max_violation(nodes)",
    ]
    for row in result_rows:
        solver = row["solver"]
        kkt = row["native_kkt"]
        lines.append(
            f"{row['intervals_per_coordinate']:4d} {solver['independent_reduced_solver_unknowns']:8d} "
            f"{solver['iterations']:10d} {str(solver['converged_regularized_problem']):>9s} "
            f"{row['timing']['assembly_and_solution_seconds_median']:10.4f} "
            f"{row['metrics']['final_objective']:15.10f} "
            f"{kkt['state_violation_linf_at_native_nodes']:21.6e}")
    lines.extend([
        "", "Recovered discrete optimality diagnostics (same definitions as CNP):",
        "N       state_weak_L2   stationarity_weak_L2   complementarity_L1   KKT_max     native_KKT_max",
    ])
    for row in result_rows:
        k = row["native_kkt"]
        lines.append(
            f"{row['intervals_per_coordinate']:4d} "
            f"{k['discrete_pde_residual_weak_l2']:17.6e} "
            f"{k['recovered_adjoint_residual_weak_l2']:22.6e} "
            f"{k['recovered_complementarity_l1']:20.6e} "
            f"{k['recovered_kkt_max_residual']:11.4e} "
            f"{k['native_kkt_max_residual']:17.6e}")
    lines.extend([
        "", "Comparison against the finest saved regularized PDAS reference:",
        "N        relative_L2_state    relative_L2_control     abs(Qcheck-Q)      regularized_relative_residual",
    ])
    for row in result_rows:
        comparison = row["reference_comparison"]
        if row["is_finest_reference"]:
            state_error = control_error = "reference"
        elif comparison is None:
            state_error = control_error = "unavailable"
        else:
            state_error = f"{comparison['relative_l2_state_error_against_reference']:.6e}"
            control_error = f"{comparison['relative_l2_control_error_against_reference']:.6e}"
        difference = row["quadrature_check"]["absolute_metric_differences"]["final_objective"]
        lines.append(
            f"{row['intervals_per_coordinate']:4d} {state_error:>24s} {control_error:>22s} "
            f"{difference:18.6e} "
            f"{row['solver']['regularized_stationarity_relative_l2']:30.6e}")
    lines.extend([
        "", "Near-contact sets use abs(y-psi)<=tau on the common evaluation points.",
        "N         tau        contact_area       reference_area      symmetric_difference       IoU",
    ])
    for row in result_rows:
        comparison = row["reference_comparison"]
        if comparison is None:
            lines.append(f"{row['intervals_per_coordinate']:4d}: reference comparison unavailable.")
            continue
        for value in comparison["near_contact_comparison"].values():
            lines.append(
                f"{row['intervals_per_coordinate']:4d} {value['state_tolerance']:11.3e} "
                f"{value['candidate_contact_area']:19.8f} "
                f"{value['reference_contact_area']:19.8f} "
                f"{value['symmetric_difference_area']:25.8f} "
                f"{value['intersection_over_union']:10.7f}")
    lines.extend([
        "", "All raw scalar diagnostics and every timing-run history are in results_json.",
        "Native arrays, common evaluation arrays and active masks are saved by mesh in this same file.",
        "Regularized convergence does not assert zero original constraint or complementarity residual.",
    ])
    return "\n".join(lines)


def evaluate_and_save(solutions, args):
    """Common integrals and native-grid KKT checks, outside solver timings."""
    evaluation_start = time.perf_counter()
    points, weights = evaluation_rule(args.eval_cells, args.eval_order)
    check_points, check_weights = evaluation_rule(args.eval_cells, args.check_order)
    finest = solutions[-1]
    reference_ok = (
        finest["intervals"] == max(args.meshes)
        and all(s["summary"]["converged_regularized_problem"] for s in solutions))
    reference_y = reconstruct_fd(finest["intervals"], finest["y"], points)
    reference_u = reconstruct_fd(finest["intervals"], finest["u"], points)
    root_arrays = {
        "mesh_sizes": np.array([s["intervals"] for s in solutions], dtype=np.int64),
        "x_eval": points,
        "weights_eval": weights,
        "reference_y_eval": reference_y,
        "reference_u_eval": reference_u,
        "reference_intervals": np.array(finest["intervals"], dtype=np.int64),
        "reference_mesh": np.array(finest["intervals"], dtype=np.int64),
        "reference_converged": np.array(reference_ok),
        "active_tolerances": np.array(args.active_tolerances, dtype=np.float64),
    }
    result_rows = []
    for solution in solutions:
        intervals = solution["intervals"]
        prefix = f"mesh_{intervals}_"
        if solution is finest:
            y_eval, u_eval = reference_y, reference_u
        else:
            y_eval = reconstruct_fd(intervals, solution["y"], points)
            u_eval = reconstruct_fd(intervals, solution["u"], points)
        metrics = pair_metrics(points, weights, y_eval, u_eval)
        y_check = reconstruct_fd(intervals, solution["y"], check_points)
        u_check = reconstruct_fd(intervals, solution["u"], check_points)
        check_metrics = pair_metrics(check_points, check_weights, y_check, u_check)
        kkt, p, recovered_mu = kkt_diagnostics(
            intervals, solution["y"], solution["u"],
            native_multiplier=solution["mu"])
        differences = {
            key: abs(float(check_metrics[key]) - float(value))
            for key, value in metrics.items()
            if isinstance(value, (float, np.floating)) and key in check_metrics}
        comparison = (compare_fields(
            y_eval, u_eval, reference_y, reference_u, weights,
            active_tolerances=args.active_tolerances)
            if reference_ok else None)
        row = {
            "intervals_per_coordinate": int(intervals),
            "solver": solution["summary"],
            "timing": solution["timing"],
            "repeat_summaries": solution["repeat_summaries"],
            "repeat_histories": solution["repeat_histories"],
            "solution_saved_from_repetition": len(solution["repeat_summaries"]),
            "metrics": metrics,
            "native_kkt": kkt,
            "quadrature_check": {
                "primary_order": int(args.eval_order),
                "check_order": int(args.check_order),
                "cells_per_coordinate": int(args.eval_cells),
                "check_metrics": check_metrics,
                "absolute_metric_differences": differences,
            },
            "reference_comparison": comparison,
            "is_finest_reference": bool(reference_ok and intervals == finest["intervals"]),
            "iteration_history": solution["history"],
        }
        result_rows.append(row)
        root_arrays.update({
            prefix + "points_native": solution["points"],
            prefix + "y_native": solution["y"],
            prefix + "u_native": solution["u"],
            prefix + "p_native": p,
            prefix + "mu_native": solution["mu"],
            prefix + "mu_recovered": recovered_mu,
            prefix + "shift_native": solution["shift"],
            prefix + "active_mask_native": solution["active"],
            prefix + "y_eval": y_eval,
            prefix + "u_eval": u_eval,
            prefix + "history_json": np.array(dumps_json(solution["history"])),
        })
        for tolerance in args.active_tolerances:
            key = f"{float(tolerance):.12g}"
            root_arrays[prefix + "near_contact_eval_" + key] = near_contact(y_eval, tolerance)
        print(
            f"N={intervals:4d}, unknowns={(intervals - 1)**2:6d}, "
            f"PDAS iterations={solution['summary']['iterations']:4d}, "
            f"converged={solution['summary']['converged_regularized_problem']}, "
            f"J(Q)={metrics['final_objective']:.10f}, "
            f"max violation(Q)={metrics['state_violation_linf_at_evaluation_points']:.3e}, "
            f"time={solution['timing']['assembly_and_solution_seconds_median']:.3f}s",
            flush=True)

    axis = np.linspace(0.0, 1.0, args.plot_points)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    plot_points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    plot_y = reconstruct_fd(finest["intervals"], finest["y"], plot_points).reshape(xx.shape)
    plot_u = reconstruct_fd(finest["intervals"], finest["u"], plot_points).reshape(xx.shape)
    root_arrays.update({
        "x_plot": plot_points, "Y_pred": plot_y, "U_pred": plot_u,
        "classical_solu": np.stack((plot_y, plot_u)),
    })
    metadata = {
        "problem": "Example 5.3: -Delta y=u, y=0 on boundary, y<=0.01",
        "alpha": ALPHA, "psi": PSI,
        "method": "five-point FD shifted Moreau--Yosida PDAS",
        "protocol_version": PROTOCOL_VERSION,
        "requested_meshes": list(args.meshes),
        "evaluated_meshes": [s["intervals"] for s in solutions],
        "gamma": float(args.gamma),
        "machine": machine_info(),
        "requested_blas_threads": int(args.threads),
        "thread_limit_applied": bool(getattr(args, "thread_limit_applied", False)),
        "evaluation_cells_per_coordinate": int(args.eval_cells),
        "evaluation_order_per_coordinate": int(args.eval_order),
        "evaluation_point_count": int(points.shape[0]),
        "evaluation_fingerprint": evaluation_fingerprint(points, weights),
        "check_order_per_coordinate": int(args.check_order),
        "check_point_count": int(check_points.shape[0]),
        "active_tolerances": list(args.active_tolerances),
        "reference_intervals": int(finest["intervals"]),
        "reference_mesh": int(finest["intervals"]),
        "reference_converged": bool(reference_ok),
        "reference_note": "Finest numerical regularized PDAS solution, not an exact solution.",
        "reconstruction": "Bilinear state and control with zero boundary nodal values.",
        "pde_feasibility_note":
            "The native FD pair satisfies A_h*y=u; separately interpolated fields need not satisfy the continuous PDE.",
        "kkt_note":
            "Native-grid diagnostics of the unregularized inequalities; actual PDAS and recovered multipliers are both checked.",
        "active_set_note":
            "Algorithmic PDAS activity uses shift+gamma*(y-psi)>0; comparison masks use abs(y-psi)<=tolerance.",
        "postprocessing_seconds_before_file_write": float(time.perf_counter() - evaluation_start),
    }
    root_arrays["metadata_json"] = np.array(dumps_json(metadata, indent=2))
    root_arrays["results_json"] = np.array(dumps_json(result_rows, indent=2))
    root_arrays["metrics_json"] = np.array(dumps_json(result_rows[-1]["metrics"], indent=2))
    report = build_report(result_rows, metadata)
    root_arrays["report_text"] = np.array(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **root_arrays)
    print(report, flush=True)
    print(f"Saved all meshes, solver histories and comparison data to {output}", flush=True)
    return reference_ok, (xx, yy, plot_y, plot_u)


def show_plots(plot_data):
    import matplotlib.pyplot as plt
    xx, yy, y, u = plot_data
    fig = plt.figure(figsize=(12, 5))
    for index, field, title in ((1, y, "State y"), (2, u, "Control u")):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(xx, yy, field, cmap="inferno", rstride=2, cstride=2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        ax.view_init(elev=30, azim=250)
    fig.tight_layout()
    plt.show()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Uniform intervals per coordinate; native unknowns are (N-1)^2.")
    parser.add_argument("--gamma", type=float, default=1e9)
    parser.add_argument("--max-iterations", type=int, default=1080)
    parser.add_argument("--relative-tolerance", type=float, default=1e-8,
                        help="Regularized stationarity L2 residual / max(1,||yd||_h).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent cold solves per mesh; reports their median time.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval-cells", type=int, default=256)
    parser.add_argument("--eval-order", type=int, default=2)
    parser.add_argument("--check-order", type=int, default=4)
    parser.add_argument("--active-tolerances", type=float, nargs="+",
                        default=list(DEFAULT_ACTIVE_TOLERANCES))
    parser.add_argument("--plot-points", type=int, default=256)
    parser.add_argument("--output", default="data/classical_state_constraint.npz")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    args.meshes = sorted(set(args.meshes))
    if (not args.meshes or min(args.meshes) < 2 or args.repeats < 1
            or args.max_iterations < 1 or args.threads < 1 or args.plot_points < 2):
        parser.error("Mesh sizes, repetitions, iteration limit, threads and plot size must be valid positive integers.")
    if (not np.isfinite(args.gamma) or args.gamma <= 0
            or not np.isfinite(args.relative_tolerance) or args.relative_tolerance <= 0):
        parser.error("gamma and relative-tolerance must be finite and positive.")
    if (args.eval_cells < 1 or args.eval_order < 1
            or args.check_order <= args.eval_order or args.progress_every < 0):
        parser.error("Require positive quadrature settings, check-order > eval-order and progress-every >= 0.")
    if any(args.eval_cells % n for n in args.meshes):
        parser.error("eval-cells must be divisible by every mesh size so quadrature cells respect all reconstruction cells.")
    if any(not np.isfinite(t) or t <= 0 for t in args.active_tolerances):
        parser.error("All active-set comparison tolerances must be finite and positive.")
    args.active_tolerances = sorted(set(args.active_tolerances))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        from threadpoolctl import threadpool_limits
        thread_context = threadpool_limits(limits=args.threads)
        args.thread_limit_applied = True
    except ImportError:
        warnings.warn("threadpoolctl is unavailable; the BLAS thread limit could not be enforced.")
        thread_context = nullcontext()
        args.thread_limit_applied = False
    solutions = []
    with thread_context:
        for intervals in args.meshes:
            solution = repeated_solve(intervals, args)
            solutions.append(solution)
            if not solution["summary"]["converged_regularized_problem"]:
                print(
                    f"ERROR: N={intervals} did not meet the regularized stopping criterion: "
                    f"{solution['summary']['stopping_reason']}. "
                    "Saving diagnostic results; no valid finest-reference comparison will be reported.",
                    flush=True)
                break
        reference_ok, plot_data = evaluate_and_save(solutions, args)
    if args.show:
        show_plots(plot_data)
    if not reference_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
