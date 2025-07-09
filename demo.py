import numpy as np
import matplotlib.pyplot as plt
from Structure import MatrixStructure, DeployedMatrixStructure
from Utils import plot_structure
import scipy.optimize as scopt


def demo_linear_offset():
    """Linear design with uniform offsets."""
    height = width = 3
    offsets = np.zeros((height, width))
    boundary_offsets = np.zeros((4, max(height, width)))

    boundary = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    corners = boundary.copy()

    structure = MatrixStructure(num_linkage_rows=height, num_linkage_cols=width)
    structure.linear_inverse_design(boundary, corners, offsets, boundary_offsets)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pts_pi, _ = structure.layout(np.pi)
    pts_0, _ = structure.layout(0.0)
    plot_structure(pts_pi, structure.quads, structure.linkages, ax[0])
    ax[0].set_title("undeployed (phi=pi)")
    plot_structure(pts_0, structure.quads, structure.linkages, ax[1])
    ax[1].set_title("contracted (phi=0)")
    plt.suptitle("Linear inverse design: offsets")
    plt.show()


def demo_linear_angle():
    """Linear design specifying deployment angles."""
    height = width = 3
    offsets = np.zeros((height, width))
    phis = np.full((height, width), 0.75 * np.pi)
    boundary_offsets = np.zeros((4, max(height, width)))
    boundary_phis = [[0.0] * height, [0.0] * width, [0.0] * height, [0.0] * width]

    boundary = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    corners = boundary.copy()

    structure = DeployedMatrixStructure(num_linkage_rows=height, num_linkage_cols=width)
    structure.linear_inverse_design(
        boundary, corners, offsets, boundary_offsets, phis, boundary_phis
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pts_pi, _ = structure.layout(np.pi)
    pts_0, _ = structure.layout(0.0)
    plot_structure(pts_pi, structure.quads, structure.linkages, ax[0])
    ax[0].set_title("undeployed (phi=pi)")
    plot_structure(pts_0, structure.quads, structure.linkages, ax[1])
    ax[1].set_title("contracted (phi=0)")
    plt.suptitle("Linear inverse design: angles")
    plt.show()


def demo_nonlinear():
    """Optimize offsets so the contracted state is circular."""
    height = width = 4
    structure = MatrixStructure(num_linkage_rows=height, num_linkage_cols=width)

    bound_links = [structure.get_boundary_linkages(i) for i in range(4)]
    directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    boundary = []
    corners = []
    for i, bound in enumerate(bound_links):
        pts = []
        for j, (r, c) in enumerate(bound):
            parallel = structure.is_linkage_parallel_to_boundary(r, c, i)
            if j == 0:
                corner = np.array([c, -r]) + directions[i]
                if not parallel:
                    corner += directions[(i - 1) % 4]
                corners.append(corner)
            if not parallel:
                pts.append(np.array([c, -r]) + directions[i])
        boundary.append(np.vstack(pts))
    boundary_vec = np.vstack(boundary)
    corners = np.vstack(corners)
    boundary_offsets = [[0.0] * height, [0.0] * width, [0.0] * height, [0.0] * width]

    structure.linear_inverse_design(
        boundary_vec, corners, np.zeros((height, width)), boundary_offsets
    )

    dual_inds = []
    for b in range(4):
        dual_inds.extend(structure.get_dual_boundary_node_inds(b))

    def residual(off_vec):
        offsets = off_vec.reshape(height, width)
        structure.linear_inverse_design(boundary_vec, corners, offsets, boundary_offsets)
        pts, _ = structure.layout(0.0)
        pts = pts[dual_inds]
        r = np.sqrt(((pts - pts.mean(0)) ** 2).sum(axis=1))
        return r - r.mean()

    res = scopt.least_squares(
        residual, np.zeros(height * width), bounds=(-np.ones(height * width), np.inf)
    )
    opt_offsets = res.x.reshape(height, width)
    structure.linear_inverse_design(boundary_vec, corners, opt_offsets, boundary_offsets)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(np.cos(theta), np.sin(theta))
    ax[0].set_aspect("equal")
    ax[0].set_title("Target circle")

    start_pts, _ = structure.layout(np.pi)
    plot_structure(start_pts, structure.quads, structure.linkages, ax[1])
    ax[1].set_title("undeployed")

    end_pts, _ = structure.layout(0.0)
    plot_structure(end_pts, structure.quads, structure.linkages, ax[2])
    ax[2].set_title("optimized contracted")
    plt.suptitle("Nonlinear optimization")
    plt.show()


if __name__ == "__main__":
    demo_linear_offset()
    demo_linear_angle()
    demo_nonlinear()
