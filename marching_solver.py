import numpy as np
import matplotlib.pyplot as plt


# ---------- core 4-bar map (paper eq. (1)) ----------
def R(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def compute_linkage(x0: np.ndarray, x3: np.ndarray, phi: float, eps: float) -> np.ndarray:
    I = np.eye(2)
    Q = (1.0 + eps) * R(-phi)
    x1 = (I - Q) @ x0 + Q @ x3
    x2 = -Q @ x0 + (I + Q) @ x3
    return np.vstack([x0, x1, x2, x3])  # [x0,x1,x2,x3]


# ---------- helpers ----------
def checkerboard_phi(m, n, phi):
    ij = np.add.outer(np.arange(m), np.arange(n))
    return np.where(ij % 2 == 0, phi, np.pi - phi)


def linspace2d(a, b, k):  # sample k points between 2D points a -> b (inclusive)
    t = np.linspace(0.0, 1.0, k)
    return (1 - t)[:, None] * a + t[:, None] * b


# March the whole m×n array, *one negative space at a time* (forward DP)
def march_array(m, n, phi_field, eps_field, top_seeds, left_seeds):
    """
    top_seeds[j]  = x_{0,j,3} for j=0..n-1  (top boundary seed nodes)
    left_seeds[i] = x_{i,0,0} for i=0..m-1  (left boundary seed nodes)
    Returns nodes[i,j,k,:] with k in {0,1,2,3}.
    """
    nodes = np.zeros((m, n, 4, 2), dtype=float)
    for i in range(m):  # row by row
        for j in range(n):  # left to right
            x0 = left_seeds[i] if j == 0 else nodes[i, j - 1, 2]  # share x2 from the left
            x3 = top_seeds[j] if i == 0 else nodes[i - 1, j, 1]  # share x1 from above
            nodes[i, j] = compute_linkage(x0, x3, phi_field[i, j], eps_field[i, j])
    return nodes


def draw_linkages(nodes):
    m, n = nodes.shape[:2]
    plt.figure(figsize=(7, 7))
    for i in range(m):
        for j in range(n):
            P = nodes[i, j]
            cyc = [0, 1, 2, 3, 0]
            plt.plot(P[cyc, 0], P[cyc, 1], lw=1)
            # optionally mark the four vertices:
            plt.plot(P[:, 0], P[:, 1], "o", ms=2)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


# ---------- example usage ----------
m = n = 3
phi_global = np.pi / 2

# interior offsets (your example) — tune these to sculpt the reconfigured shape
eps = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)

# choose two boundary *seed* polylines instead of "+2" hacks
# (edit these three points to set your shape)
x00 = np.array([0.0, 0.0])  # left seed for (0,0): x_{0,0,0}
x03 = np.array([1.0, 1.0])  # top  seed for (0,0): x_{0,0,3}
top_right = x03 + np.array([4.0, 0.0])  # end of the top seed line
bottom_left = x00 + np.array([0.0, -4.0])  # end of the left seed line

# sample the seeds along those lines (replace with any polyline sampler you like)
top_seeds = linspace2d(x03, top_right, n)  # x_{0,j,3}, j=0..n-1
left_seeds = linspace2d(x00, bottom_left, m)  # x_{i,0,0}, i=0..m-1

phi_field = checkerboard_phi(m, n, phi_global)  # compact-reconfigurable field

nodes = march_array(m, n, phi_field, eps, top_seeds, left_seeds)
draw_linkages(nodes)
