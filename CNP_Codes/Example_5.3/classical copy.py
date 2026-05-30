import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def laplacian_5pt_dirichlet(n: int, h: float) -> sp.csr_matrix:
    e = np.ones(n)
    T = sp.diags([-e, 4*e, -e], [-1,0,1], shape=(n,n))
    S = sp.diags([-e, -e], [-1,1], shape=(n,n))
    I = sp.eye(n, format="csr")
    A = (sp.kron(I, T) + sp.kron(S, I)) / (h*h)  # approximates -Δ
    return A.tocsr()

def build_grid_from_h(h: float):
    n_total = int(round(1/h)) + 1
    n = n_total - 2
    xs = np.linspace(h, 1-h, n)
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    return n, xs, X1, X2

def yd_problem1(X1, X2):
    return 10.0*(np.sin(2*np.pi*X1) + X2)

def psi_problem1(X1, X2):
    return 0.01*np.ones_like(X1)

def pdas_gamma_moreau_yosida(A, yd, psi, beta, gamma, lam_bar, y0, maxit=1080):
    """
    Correct scaling:
      (beta*A^2 + I + gamma*D) y = yd - D*lam_bar + gamma*D*psi
    with D = diag(1_{active}).
    """
    y = y0.copy()
    n2 = y.size
    I = sp.eye(n2, format="csr")
    A2 = (A @ A).tocsr()

    for _ in range(maxit):
        Aset = (lam_bar + gamma*(y-psi) > 0.0)
        D = sp.diags(Aset.astype(float), 0, format="csr")

        K = (beta*A2 + I + gamma*D).tocsr()
        rhs = yd - (D @ lam_bar) + gamma*(D @ psi)
        y_new = spla.spsolve(K, rhs)

        Aset_new = (lam_bar + gamma*(y_new-psi) > 0.0)
        y = y_new
        if np.array_equal(Aset, Aset_new):
            print('break')
            break

    u = A @ y
    lam = np.maximum(lam_bar + gamma*(y-psi), 0.0)  # multiplier
    return y, u, lam

def plot_surface(ax, X1, X2, Z, title, cmap='inferno'):
    surf = ax.plot_surface(
        X1, X2, Z,
        cmap=cmap,
        rstride=2, cstride=2, alpha=0.9
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return surf

def solve_problem1_and_plot_3d(h=1/255, gamma=1e9, beta=0.1):
    n, xs, X1, X2 = build_grid_from_h(h)
    A = laplacian_5pt_dirichlet(n, h)
    A2 = (A @ A).tocsr()

    yd = yd_problem1(X1, X2).reshape(-1)
    psi = psi_problem1(X1, X2).reshape(-1)

    # λ̄-shift: λ̄ = max( yd - (βA^2 + I)ψ, 0 )
    lam_bar = np.maximum(yd - (beta*(A2 @ psi) + psi), 0.0)

    y0 = psi.copy()
    y, u, lam = pdas_gamma_moreau_yosida(A, yd, psi, beta, gamma, lam_bar, y0)

    Y = y.reshape(n, n)
    U = u.reshape(n, n)
    Lam = lam.reshape(n, n)

    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    ax1.view_init(elev=30, azim=250)
    ax2.view_init(elev=30, azim=250)
    ax3.view_init(elev=30, azim=250)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plot_surface(ax1, X1, X2, Y, "State y_h", cmap='inferno')
    plot_surface(ax2, X1, X2, U, "Control u_h", cmap='inferno')
    plot_surface(ax3, X1, X2, Lam, "Multiplier λ_h",cmap='inferno')

    Y2 = np.zeros((256,256))
    Y2[1:-1, 1:-1] = Y

    U2 = np.zeros((256, 256))
    U2[1:-1, 1:-1] = U

    np.save('data/classical_solu.npy', [Y2, U2])




    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_problem1_and_plot_3d()
