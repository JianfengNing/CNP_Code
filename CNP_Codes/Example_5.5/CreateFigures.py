"""Create the paper figures and cost table from the two completed Adam runs.

Keep this file beside reduced_method.py and KKT_method.py. Run the two training
scripts sequentially, then run this file without arguments. Default inputs:
  results/reduced_adam_seed42/{report.json,evaluation_fields.pth}
  results/kkt_adam_seed42/{report.json,evaluation_fields.pth}
All default paths are relative to THIS script, not the terminal working folder.

Optional: python CreateFigures.py --reduced /path/to/reduced_run --kkt /path/to/kkt_run
Outputs: data/NS_high_fre.png/.pdf, error_history_NS.png/.pdf,
error_history_NS_time.png/.pdf, comparison.csv, comparison.tex (blue text),
and comparison_details.json. Only saved CPU tensors are loaded; no network
inference, automatic differentiation or GPU allocation is needed for plotting.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent


def load_run(directory, expected_method):
    directory = Path(directory)
    report_path = directory/"report.json"
    field_path = directory/"evaluation_fields.pth"
    if not report_path.is_file() or not field_path.is_file():
        raise FileNotFoundError(
            f"Missing report.json or evaluation_fields.pth in {directory}. "
            f"Run the revised {expected_method} training script first, or supply its output directory."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data = torch.load(field_path, map_location="cpu", weights_only=True)
    if report["method"] != expected_method or data.get("method") != expected_method:
        raise ValueError(f"{directory}: wrong method for this input")
    for source in (report["configuration"], data):
        if source.get("optimizer") != "adam" or source.get("weight_decay") != 0.0:
            raise ValueError(f"{directory}: expected Adam with zero weight decay; use the new Adam results")
    if data["epochs"] != report["configuration"]["epochs"] or data["seed"] != report["configuration"]["seed"]:
        raise ValueError(f"{directory}: fields and report describe different runs")
    if not math.isclose(data["training_seconds"], report["timing"]["training_seconds"], rel_tol=1e-12):
        raise ValueError(f"{directory}: timing metadata differs between fields and report")
    n = report["evaluation"]["nodes_per_axis"]
    if data["nodes_per_axis"] != n:
        raise ValueError(f"{directory}: evaluation-grid metadata mismatch")
    arrays = {}
    for key in ("x", "y", "u", "y_exact", "u_exact"):
        tensor = data[key]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (n*n, 2):
            raise ValueError(f"{directory}: expected {key} with shape {(n*n, 2)}")
        arrays[key] = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{directory}: nonfinite {key}")
    axis = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    expected_grid = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.allclose(arrays["x"], expected_grid, rtol=0, atol=2e-7):
        raise ValueError(f"{directory}: fields are not ordered on the reported uniform grid")
    history = {
        "epoch": np.asarray(data["error_epochs"], dtype=np.int64),
        "training_seconds": np.asarray(data["error_training_seconds"], dtype=np.float64),
        "control_relative_l2": np.asarray(data["error_history"], dtype=np.float64),
    }
    if not len(history["epoch"]) or len({len(v) for v in history.values()}) != 1:
        raise ValueError(f"{directory}: missing or inconsistent error history")
    if any(not np.isfinite(v).all() for v in history.values()):
        raise ValueError(f"{directory}: nonfinite error history")
    if np.any(np.diff(history["epoch"]) <= 0) or np.any(np.diff(history["training_seconds"]) < 0):
        raise ValueError(f"{directory}: error checkpoints are not ordered")
    if history["epoch"][-1] != report["configuration"]["epochs"]:
        raise ValueError(f"{directory}: final error checkpoint is missing")
    # Verify displayed numbers against the saved fields, rather than retaining
    # the hard-coded error labels in the original plotting file.
    one_dimensional_weights = np.ones(n)/(n-1)
    one_dimensional_weights[[0, -1]] *= 0.5
    q = np.outer(one_dimensional_weights, one_dimensional_weights).ravel()
    for key, metric in (("y", "state_relative_l2"), ("u", "control_relative_l2")):
        reference = arrays[key+"_exact"]
        denominator = np.sum(q*np.sum(reference**2, axis=1))
        if denominator <= 0:
            raise ValueError(f"{directory}: zero reference norm")
        relative_error = np.sqrt(np.sum(q*np.sum((arrays[key]-reference)**2, axis=1))/denominator)
        if not np.isclose(relative_error, report["final_metrics"][metric], rtol=1e-5, atol=1e-10):
            raise ValueError(f"{directory}: saved fields and reported {metric} disagree")
    if not math.isclose(history["control_relative_l2"][-1],
                        report["final_metrics"]["control_relative_l2"], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{directory}: final history error disagrees with report")
    return {"directory": str(directory.resolve()), "report": report, "fields": arrays,
            "history": history, "n": n, "X": xx, "Y": yy}


def check_comparison(reduced, kkt):
    a, b = reduced["report"], kkt["report"]
    if a["evaluation"] != b["evaluation"]:
        raise ValueError("Both methods must use the same error-evaluation grid and quadrature")
    for key in ("mu", "alpha", "adjoint_frequency"):
        if a["configuration"][key] != b["configuration"][key]:
            raise ValueError(f"The benchmark problems differ in {key}")
    for key in ("x", "y_exact", "u_exact"):
        if not np.allclose(reduced["fields"][key], kkt["fields"][key], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Reference field or grid differs: {key}")
    notes = []
    for key in ("epochs", "width", "blocks", "lr", "lr_step", "lr_gamma", "seed",
                "data_seed", "train_batch_size", "eval_batch_size", "eval_every", "cache"):
        if a["configuration"][key] != b["configuration"][key]:
            notes.append(f"Different {key}: {a['configuration'][key]} versus {b['configuration'][key]}")
    for key in ("device", "device_name", "dtype", "pytorch", "cpu_threads"):
        if a["environment"][key] != b["environment"][key]:
            notes.append(f"Different environment {key}: {a['environment'][key]} versus {b['environment'][key]}")
    if a["training_grid"]["schedule_sha256"] != b["training_grid"]["schedule_sha256"]:
        notes.append("The training-grid schedules differ")
    for note in notes:
        warnings.warn(note, stacklevel=2)
    return notes


def scientific_tex(value, digits=3):
    coefficient, exponent = f"{value:.{digits-1}e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def write_comparison(reduced, kkt, output, notes):
    rows = []
    for run in (reduced, kkt):
        r = run["report"]
        rows.append({"method": r["method"], "optimizer": "Adam", "weight_decay": 0.0,
                     "trainable_parameters": r["parameters"]["total"],
                     "epochs": r["configuration"]["epochs"],
                     "setup_seconds": r["timing"]["setup_seconds"],
                     "training_seconds": r["timing"]["training_seconds"],
                     "mean_epoch_seconds": r["timing"]["mean_epoch_seconds"],
                     "monitoring_seconds": r["timing"]["monitoring_seconds"],
                     "final_evaluation_seconds": r["timing"]["final_evaluation_seconds"],
                     **r["final_metrics"]})
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"% Requires \usepackage{xcolor}.", r"\begin{table}[htbp]", r"\centering",
             r"{\color{blue}\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Method & Parameters & Epochs & Training (s) & s/epoch & State error & Control error \\",
             r"\hline"]
    for row in rows:
        name = "Reduced" if row["method"] == "reduced" else "KKT"
        cells = [name, str(row["trainable_parameters"]), str(row["epochs"]),
                 f"{row['training_seconds']:.3f}", f"{row['mean_epoch_seconds']:.6f}",
                 "$"+scientific_tex(row["state_relative_l2"], 4)+"$",
                 "$"+scientific_tex(row["control_relative_l2"], 4)+"$"]
        lines.append(" & ".join(cells)+r" \\")
    n = reduced["n"]
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Computational cost and accuracy for the Navier--Stokes example. "
              r"Both methods use Adam with zero weight decay. Each row reports one run. "
              r"Training times are synchronized wall-clock measurements, excluding setup, "
              r"error monitoring, plotting and file writing. Errors are relative $L^2$ errors "
              rf"computed using tensor-product trapezoidal quadrature on a common ${n}\times{n}$ grid.}}",
              r"\label{tab:navier_stokes_cost}", r"}", r"\end{table}", ""]
    (output/"comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    details = {"comparison_notes": notes,
               "reduced": {"directory": reduced["directory"], "report": reduced["report"]},
               "kkt": {"directory": kkt["directory"], "report": kkt["report"]}}
    (output/"comparison_details.json").write_text(json.dumps(details, indent=2)+"\n", encoding="utf-8")
    for row in rows:
        print(f"{row['method']}: parameters={row['trainable_parameters']:,}; "
              f"training={row['training_seconds']:.3f} s; per epoch={row['mean_epoch_seconds']:.6f} s; "
              f"state error={row['state_relative_l2']:.6e}; control error={row['control_relative_l2']:.6e}")


def create_figures(reduced, kkt, output, component=1, dpi=200, surface_points=100, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    index = component-1
    n = reduced["n"]
    fields = [reduced["fields"]["u_exact"][:, index].reshape(n, n),
              reduced["fields"]["u"][:, index].reshape(n, n),
              kkt["fields"]["u"][:, index].reshape(n, n)]
    low, high = min(float(v.min()) for v in fields), max(float(v.max()) for v in fields)
    span = max(high-low, 1e-12)
    norm = Normalize(vmin=low, vmax=high if high>low else low+1e-12)
    error_reduced = reduced["report"]["final_metrics"]["control_relative_l2"]
    error_kkt = kkt["report"]["final_metrics"]["control_relative_l2"]
    titles = [rf"Exact control $u_{component}$",
              rf"Reduced method: $u_{component}$"+"\n"+rf"Vector relative $L^2$ error: ${scientific_tex(error_reduced)}$",
              rf"KKT method: $u_{component}$"+"\n"+rf"Vector relative $L^2$ error: ${scientific_tex(error_kkt)}$"]
    stride = max(1, math.ceil((n-1)/(surface_points-1)))
    fig = plt.figure(figsize=(15, 5.2), layout="constrained")
    for i, (values, title) in enumerate(zip(fields, titles), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(reduced["X"], reduced["Y"], values, cmap="inferno", norm=norm,
                        rstride=stride, cstride=stride, linewidth=0, antialiased=True,
                        rasterized=True)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(rf"$u_{component}$", labelpad=6)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(low-0.05*span, high+0.05*span)
        ax.view_init(elev=30, azim=250)
        ax.grid(True, alpha=0.3)
    suffix = "" if component == 1 else f"_u{component}"
    for extension in ("png", "pdf"):
        fig.savefig(output/f"NS_high_fre{suffix}.{extension}", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    labels = ("Exact reduced neural method", "KKT-based neural method")
    for coordinate, xlabel, filename in (
        ("epoch", "Epoch", "error_history_NS"),
        ("training_seconds", "Training time (s; excludes error evaluation)", "error_history_NS_time"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        for run, label in zip((reduced, kkt), labels):
            history = run["history"]
            error = history["control_relative_l2"]
            if np.any(error <= 0):
                warnings.warn("Zero errors cannot be displayed on a log axis; those points are omitted")
            positive = error > 0
            ax.semilogy(history[coordinate][positive], error[positive], marker="o",
                        markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Relative $L^2$ error of the vector control")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        for extension in ("png", "pdf"):
            fig.savefig(output/f"{filename}.{extension}", dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced", type=Path, default=SCRIPT_DIR/"results"/"reduced_adam_seed42")
    parser.add_argument("--kkt", type=Path, default=SCRIPT_DIR/"results"/"kkt_adam_seed42")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR/"data")
    parser.add_argument("--component", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--surface-points", type=int, default=100,
                        help="Approximate nodes per coordinate for rendering only; errors use the full grid")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1 or args.surface_points < 2:
        parser.error("dpi must be positive and surface-points must be at least two")
    reduced = load_run(args.reduced, "reduced")
    kkt = load_run(args.kkt, "kkt")
    notes = check_comparison(reduced, kkt)
    args.output.mkdir(parents=True, exist_ok=True)
    create_figures(reduced, kkt, args.output, args.component, args.dpi, args.surface_points, args.show)
    write_comparison(reduced, kkt, args.output, notes)
    print(f"Saved figures and comparison table to {args.output.resolve()}")


if __name__ == "__main__":
    main()
