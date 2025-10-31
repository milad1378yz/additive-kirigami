import numpy as np
import matplotlib.pyplot as plt
from Structure import MatrixStructure
from Utils import rotate_points, plot_structure


# %%
def structure_heart_3_3(interior_offsets):
    height = 3
    width = 3

    boundary_offset = 0.0
    boundary_offsets = np.array(
        [
            [boundary_offset] * height,
            [boundary_offset] * width,
            [boundary_offset] * height,
            [boundary_offset] * width,
        ]
    )

    points = (
        np.array(
            [
                [0, 0],
                [-80, 50],
                [-160, 35],
                [-210, -45],
                [-150, -170],
                [0, -295],
                [150, -170],
                [210, -45],
                [160, 35],
                [80, 50],
            ]
        )
        / 100.0
    )

    corners = np.array([points[2], points[4], points[7], points[9]])

    structure = MatrixStructure(num_linkage_rows=height, num_linkage_cols=width)

    midpoints = np.array(
        [
            np.mean(points[[2, 3], :], axis=0),
            np.mean(points[[4, 5], :], axis=0),
            np.mean(points[[5, 6], :], axis=0),
            np.mean(points[[7, 8], :], axis=0),
            np.mean(points[[9, 0], :], axis=0),
            np.mean(points[[0, 1], :], axis=0),
        ]
    )

    structure.linear_inverse_design(
        boundary_points=midpoints,
        corners=corners,
        interior_offsets=interior_offsets,
        boundary_offsets=boundary_offsets,
    )

    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()

    # Return structure and the target boundary point set for plotting
    return (structure, midpoints, corners, points)


def _seed_node_indices(structure: MatrixStructure):
    """Return node indices corresponding to seed points (top, left, corners)."""
    num_rows = structure.num_linkage_rows
    num_cols = structure.num_linkage_cols

    seed_inds = []
    # top edge seed points (k=3 on row i=0)
    for j in range(num_cols):
        seed_inds.append(structure.linkage2matrix(i=0, j=j, k=3))
    # left edge seed points (k=0 on col j=0)
    for i in range(num_rows):
        seed_inds.append(structure.linkage2matrix(i=i, j=0, k=0))
    # corners as used in calculate_design_matrix
    seed_inds.append(structure.linkage2matrix(i=0, j=-1, k=3))  # top-left
    seed_inds.append(structure.linkage2matrix(i=num_rows, j=0, k=0))  # bottom-left
    seed_inds.append(structure.linkage2matrix(i=num_rows - 1, j=num_cols, k=1))  # bottom-right
    seed_inds.append(structure.linkage2matrix(i=-1, j=num_cols - 1, k=2))  # top-right
    return seed_inds


# ## Example 1

offset = 0.0
interior_offsets = np.array([[offset] * 3] * 3)

# plot the offset field
plt.matshow(interior_offsets, cmap="coolwarm")
plt.clim(-1, 10)
plt.axis("off")
cbar = plt.colorbar()

# Create a compact reconfigurable heart pattern with the corresponding offset field
structure, boundary_targets, corners, points = structure_heart_3_3(interior_offsets)
seed_inds = _seed_node_indices(structure)

# plot deployment snapshots of the structure created
num_frames = 1
phi = np.pi
panel_size = 10
fig, axs = plt.subplots(1, num_frames, figsize=(10, 10), sharey=True)


deployed_points, deployed_hinge_contact_points = structure.layout(phi)
rot_angle = -(np.pi - phi) / 2.0
deployed_points = rotate_points(deployed_points, np.array([0, 0]), rot_angle)
plot_structure(deployed_points, structure.quads, structure.linkages, axs)
axs.set_title(f"Deployment angle: {np.degrees(phi):.1f}°")
axs.set_aspect("equal")
axs.axis("off")

# plot boundary target points
axs.plot(
    boundary_targets[:, 0],
    boundary_targets[:, 1],
    "ro",
    markersize=8,
    label="Boundary targets",
)

# plot seed points
seed_points = deployed_points[seed_inds, :]
axs.plot(
    seed_points[:, 0],
    seed_points[:, 1],
    "gs",
    markersize=6,
    label="Seed points",
)

# plot corners
axs.plot(
    corners[:, 0],
    corners[:, 1],
    "b^",
    markersize=8,
    label="Corners",
)

# plot all the target points for reference
axs.plot(
    points[:, 0],
    points[:, 1],
    "cD",
    markersize=4,
    label="All target points",
)


# add legend to the first subplot only
axs.legend(loc="upper right")

plt.savefig("heart_3_3_deployment.png", dpi=300)

# --- Subplots per legend category: draw shape and bold that category ---
categories = [
    ("Boundary targets", boundary_targets, dict(marker="x", s=160, c="red", linewidths=3.0)),
    (
        "Seed points",
        seed_points,
        dict(marker="o", s=140, c="green", edgecolors="k", linewidths=1.5),
    ),
    ("Corners", corners, dict(marker="^", s=160, c="blue", edgecolors="k", linewidths=1.0)),
    (
        "All target points",
        points,
        dict(marker="D", s=120, c="cyan", edgecolors="k", linewidths=1.0),
    ),
    (
        "Deployed points",
        deployed_points,
        dict(marker="o", s=140, c="orange", edgecolors="k", linewidths=1.5),
    ),
]

panel_size = 6
fig2, axs2 = plt.subplots(
    1,
    len(categories),
    figsize=(1.2 * panel_size * len(categories), panel_size),
    sharey=True,
)

for idx, (label, pts, style) in enumerate(categories):
    ax = axs2[idx] if len(categories) > 1 else axs2
    # Draw main structure
    plot_structure(deployed_points, structure.quads, structure.linkages, ax)
    # Overlay this category's points in bold
    ax.scatter(pts[:, 0], pts[:, 1], zorder=7, **style)
    ax.set_title(label)

# fig2.tight_layout()
plt.savefig("heart_3_3_categories.png", dpi=300)
