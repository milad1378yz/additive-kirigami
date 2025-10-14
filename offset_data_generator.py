import numpy as np
import pickle
from matplotlib.path import Path
from tqdm import trange

from tqdm import tqdm

from Structure import MatrixStructure

from Utils import plot_structure
from copy import deepcopy


from concurrent.futures import ThreadPoolExecutor

try:
    from scipy.stats import qmc as _qmc

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------
# Config (edit to taste)
# ----------------------------
GRID_ROWS = 18  # number of linkage rows  (height)
GRID_COLS = 18  # number of linkage cols  (width)
IMG_H, IMG_W = 128, 128  # output image resolution
N_TRAIN = 5000  # how many training samples to generate
N_VALID = 1000  # how many validation samples to generate
N_TRAIN = 10000  # how many training samples to generate
N_VALID = 500  # how many validation samples to generate
OUTPUT_PKL = "kirigami_dataset.pkl"
RANDOM_SEED = 42  # set to None for non-deterministic

# sampling / epsilon-range hyperparameters
USE_SOBOL = True  # do sobol sampling to cover better the space
EPS_MIN = -0.9  # range of the epsilons should be in my hand (hyperparameters)
EPS_MAX = 9.0  # (matches your original ~(-0.9, 9))
EPS_SCALE = "log"  # "log" to preserve your 10^x - 1 shape; use "linear" for uniform range
NUM_WORKERS = 20  # do multi threading for the loop of generation (None -> let Python decide)


# ----------------------------
# Helper functions
# ----------------------------
def _compute_boundary_points_and_corners(structure: MatrixStructure):
    """Match your original logic to get boundary points and corners for inverse design."""
    bound_linkage_inds = [structure.get_boundary_linkages(i) for i in range(4)]
    bound_directions = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    boundary_points = []
    corners = []
    for i, bound in enumerate(bound_linkage_inds):
        local_boundary_points = []
        for j, linkage_ind in enumerate(bound):
            p = structure.is_linkage_parallel_to_boundary(linkage_ind[0], linkage_ind[1], i)
            if j == 0:
                corner = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
                if not p:
                    corner += bound_directions[(i - 1) % 4]
                corners.append(corner)
            if not p:
                point = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
                local_boundary_points.append(point)
        boundary_points.append(np.vstack(local_boundary_points))
    corners = np.vstack(corners)
    return boundary_points, corners


# Map uniform [0,1] -> epsilon in desired range (linear or log10 like your original)
def _map_u_to_eps(
    u01: np.ndarray, eps_min: float, eps_max: float, scale: str = "log"
) -> np.ndarray:
    """
    u01: ndarray in [0,1] (any shape)
    returns eps with same shape in [eps_min, eps_max] using the chosen scaling.
    """
    if scale == "log":
        # preserves your shape: eps = 10^(a + (b-a)*u) - 1
        lo = eps_min + 1.0
        hi = eps_max + 1.0
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("For EPS_SCALE='log', require EPS_MIN > -1 so that (eps+1) > 0.")
        a = np.log10(lo)
        b = np.log10(hi)
        return np.power(10.0, a + (b - a) * u01) - 1.0
    else:
        # linear
        return eps_min + (eps_max - eps_min) * u01


# Sobol (quasi-random) block generator; falls back to uniform if SciPy missing or disabled
def _make_u_batches(
    n_samples: int, grid_rows: int, grid_cols: int, seed=None, use_sobol=True
) -> np.ndarray:
    """
    Returns an array of shape (n_samples, grid_rows, grid_cols) with values in [0,1].
    """
    dim = grid_rows * grid_cols
    if use_sobol and _HAVE_SCIPY:
        # Scrambled Sobol for better space-filling; reproducible via seed
        sampler = _qmc.Sobol(d=dim, scramble=True, seed=seed)
        # We don't enforce power-of-two; .random(n) continues the sequence fine
        u = sampler.random(n_samples)  # shape: (n_samples, dim)
        u = u.reshape(n_samples, grid_rows, grid_cols)
        return u
    else:
        if use_sobol and not _HAVE_SCIPY:
            print("[WARN] SciPy not found; falling back to standard uniform sampling.")
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, grid_rows, grid_cols))


# NEW: silhouette rasterizer (φ = 0), fills the union of all quads (no cuts)
def _rasterize_quads_filled(points, quads, out_h=256, out_w=256):
    """
    Rasterize the *shape silhouette* by filling the union of all quads.
    Inside shape = 1, outside = 0. Uses same normalization as above.

    Args:
        points (ndarray): (N, 2) vertex coordinates in 2D.
        quads   (ndarray): (M, 4) indices for each quadrilateral.
        out_h, out_w (int): output image size.

    Returns:
        mask (ndarray): (out_h, out_w) float32 values in {0.0, 1.0}
    """
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Match the same aspect-safe scale as the edge rasterizer
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)

    def to_pixels_float(p):
        # Floating pixel coords; origin top-left, y inverted
        return (p[0] - xmin) * s, (ymax - p[1]) * s

    # Build a compound Path containing all quads
    verts = []
    codes = []
    for quad in quads:
        q = [int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3])]
        p0 = to_pixels_float(pts[q[0]])
        p1 = to_pixels_float(pts[q[1]])
        p2 = to_pixels_float(pts[q[2]])
        p3 = to_pixels_float(pts[q[3]])
        verts.extend([p0, p1, p2, p3, (0.0, 0.0)])  # CLOSEPOLY ignores the last vertex value
        codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    compound_path = Path(verts, codes)

    # Test all pixel centers for containment
    xs = np.arange(out_w) + 0.5
    ys = np.arange(out_h) + 0.5
    xv, yv = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xv.ravel(), yv.ravel()])

    inside = compound_path.contains_points(grid_pts)
    mask = np.zeros((out_h, out_w), dtype=np.float32)
    mask.flat[:] = inside.astype(np.float32)
    return mask


def _make_one_sample(
    grid_rows,
    grid_cols,
    img_h,
    img_w,
    rng,
    # optional quasi-random block & eps config
    u_interior=None,
    eps_min=None,
    eps_max=None,
    eps_scale="log",
):
    """
    Build one kirigami sample and return:
        {
          "image": (1, H, W) float32 in [0,1], paper=1, cuts=0   @ φ = π
          "mask":  (1, H, W) float32 in [0,1], silhouette (no cuts) @ φ = 0
          "metadata": {...}
        }
    """
    # 1) structure for this sample
    structure = MatrixStructure(num_linkage_rows=grid_rows, num_linkage_cols=grid_cols)

    # 2) boundary points & corners (as in your original code)
    boundary_points, corners = _compute_boundary_points_and_corners(structure)

    # 3) random interior offsets ~ (-0.9, 9) using your original distribution
    if u_interior is None:
        # fallback to your original random (kept)
        interior_u = rng.random((grid_rows, grid_cols))
        interior_offsets = np.power(10.0, interior_u * 2.0 - 1.0) - 1.0
    else:
        emn = EPS_MIN if eps_min is None else eps_min
        emx = EPS_MAX if eps_max is None else eps_max
        esc = EPS_SCALE if eps_scale is None else eps_scale
        interior_offsets = _map_u_to_eps(u_interior, emn, emx, esc)

    # make a deep copy of this
    # interior_offsets_copy = deepcopy(interior_offsets)

    # 4) zero boundary offsets
    boundary_offsets = [[0.0] * grid_rows, [0.0] * grid_cols, [0.0] * grid_rows, [0.0] * grid_cols]

    # 5) inverse design + bookkeeping (exactly as your code does)
    structure.linear_inverse_design(
        np.vstack(boundary_points), corners, interior_offsets, boundary_offsets
    )
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()

    # 6) two layouts:
    #    - φ = π: "flat" for the original cuts image (paper=1, cuts=0)
    #    - φ = 0: deformed endpoint for the silhouette mask (no cuts)
    points_0, _ = structure.layout(0.0)

    # recentre both (kept from your original; rasterizer still normalizes to bbox)
    points_0[:, 0] = points_0[:, 0] - (np.max(points_0[:, 0]) + np.min(points_0[:, 0])) / 2
    points_0[:, 1] = points_0[:, 1] - (np.max(points_0[:, 1]) + np.min(points_0[:, 1])) / 2

    # 7) rasterize (unchanged image) + NEW silhouette mask
    silhouette_mask = _rasterize_quads_filled(
        points_0, structure.quads, out_h=img_h, out_w=img_w
    )  # φ = 0

    # 8) add channel dimension, ensure float32
    image = interior_offsets.astype(np.float32)[None, :, :]
    mask = silhouette_mask.astype(np.float32)[None, :, :]

    # # save fig for both mask and image
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image[0, :, :], cmap="gray")
    # axs[0].set_title("Cuts Mask (φ = π)")
    # axs[1].imshow(mask[0, :, :], cmap="gray")
    # axs[1].set_title("Silhouette Mask (φ = 0)")
    # plt.savefig(f"sample.png")
    # plt.close(fig)

    # # also save the ful image
    # _, axs = plt.subplots(1, 1, figsize=(5, 5))
    # plot_structure(points_0, structure.quads, ax=axs, linkages=None)
    # plt.savefig(f"sample_full.png")
    # plt.close()

    # 9) metadata
    metadata = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "phi_image": float(np.pi),  # φ used to make "image"
        "phi_mask": 0.0,  # φ used to make "mask" (silhouette)
        "interior_offsets": interior_offsets.astype(np.float32),  # reproducibility / analysis
        "num_quads": int(len(structure.quads)),
    }

    return {"image": image, "metadata": metadata, "mask": mask}


def build_dataset(
    n_train,
    n_valid,
    grid_rows,
    grid_cols,
    img_h,
    img_w,
    seed=None,
    # Pass-through knobs for eps mapping and sobol
    eps_min=EPS_MIN,
    eps_max=EPS_MAX,
    eps_scale=EPS_SCALE,
    use_sobol=USE_SOBOL,
    num_workers=NUM_WORKERS,
):
    rng = np.random.default_rng(seed)
    ds = {"train": [], "valid": []}

    # Precompute quasi-random blocks for reproducibility & threading
    u_train = _make_u_batches(n_train, grid_rows, grid_cols, seed=seed, use_sobol=use_sobol)
    u_valid = _make_u_batches(
        n_valid, grid_rows, grid_cols, seed=None if seed is None else seed + 1, use_sobol=use_sobol
    )

    # Threaded generation for train
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        # use tqdm for progress (keeps your original feel)
        for sample in tqdm(
            ex.map(
                lambda u: _make_one_sample(
                    grid_rows,
                    grid_cols,
                    img_h,
                    img_w,
                    None,
                    u_interior=u,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    eps_scale=eps_scale,
                ),
                u_train,
            ),
            total=n_train,
            desc="Generating train samples (threads)",
        ):
            ds["train"].append(sample)

    # Threaded generation for valid
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for sample in tqdm(
            ex.map(
                lambda u: _make_one_sample(
                    grid_rows,
                    grid_cols,
                    img_h,
                    img_w,
                    None,
                    u_interior=u,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    eps_scale=eps_scale,
                ),
                u_valid,
            ),
            total=n_valid,
            desc="Generating valid samples (threads)",
        ):
            ds["valid"].append(sample)

    return ds


# ----------------------------
# Run & save
# ----------------------------
if __name__ == "__main__":
    dataset = build_dataset(
        n_train=N_TRAIN,
        n_valid=N_VALID,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        img_h=IMG_H,
        img_w=IMG_W,
        seed=RANDOM_SEED,
        # Expose epsilon-range and Sobol/threading knobs here as well
        eps_min=EPS_MIN,
        eps_max=EPS_MAX,
        eps_scale=EPS_SCALE,
        use_sobol=USE_SOBOL,
        num_workers=NUM_WORKERS,
    )

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved {len(dataset['train'])} train and {len(dataset['valid'])} valid samples to {OUTPUT_PKL}"
    )
    # Each sample:
    #   sample["image"] -> np.ndarray of shape (1, IMG_H, IMG_W), dtype=float32  (paper=1, cuts=0)   @ φ = π
    #   sample["mask"]  -> np.ndarray of shape (1, IMG_H, IMG_W), dtype=float32  (sheet silhouette) @ φ = 0
    #   sample["metadata"] -> dict with phi_image / phi_mask and other info
