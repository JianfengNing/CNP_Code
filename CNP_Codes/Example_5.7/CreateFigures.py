"""Plot the saved reconstructions for Example 5.7 and report their errors."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent / "data"


def relative_l2_error(predicted, exact):
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    reference_norm = np.linalg.norm(exact.ravel())
    if reference_norm == 0:
        raise ValueError("Relative L2 error is undefined for a zero reference field.")
    return np.linalg.norm((predicted - exact).ravel()) / reference_norm


def create_figure(input_path, output_path):
    data = torch.load(input_path, map_location="cpu", weights_only=True)
    x = data["x_1"].numpy()
    y = data["x_2"].numpy()
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("Expected x_1 and x_2 to be matching two-dimensional grids.")

    sigma_exact = data["C_exa"].numpy().reshape(x.shape)
    sigma_pred = data["C_pred"].numpy().reshape(x.shape)
    f_exact = data["F_exa"].numpy().reshape(x.shape)
    f_pred = data["F_pred"].numpy().reshape(x.shape)

    sigma_error = relative_l2_error(sigma_pred, sigma_exact)
    f_error = relative_l2_error(f_pred, f_exact)
    print(f"Relative L2 error (sigma) = {sigma_error:.6e}")
    print(f"Relative L2 error (f)     = {f_error:.6e}")

    panels = (
        (r"(a) Exact $\sigma$", sigma_exact, None),
        (r"(b) Numerical $\sigma$", sigma_pred, sigma_error),
        (r"(c) Exact $f$", f_exact, None),
        (r"(d) Numerical $f$", f_pred, f_error),
    )
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), subplot_kw={"projection": "3d"})
    for ax, (title, values, error) in zip(axes, panels):
        ax.plot_surface(x, y, values, cmap="inferno", rstride=2, cstride=2, alpha=0.9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if error is not None:
            mantissa, exponent = f"{error:.2e}".split("e")
            title += "\n" + rf"Relative $L^2$ error: ${mantissa}\times 10^{{{int(exponent)}}}$"
        ax.set_title(title, pad=0)
        ax.view_init(elev=30, azim=250)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "hybrid_inverse_problem.pth")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "hybrid_inv_prob.png")
    args = parser.parse_args()
    create_figure(args.input, args.output)


if __name__ == "__main__":
    main()
