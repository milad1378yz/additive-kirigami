import numpy as np


def norm(x):
    """
    Calculate the Euclidean norm (magnitude) of a vector.

    Args:
        x (array-like): Input vector of any dimension

    Returns:
        float: The Euclidean norm of the vector

    Shape transition: (n,) -> scalar
    """
    return np.linalg.norm(x)


def normalize(x):
    """
    Normalize a vector to unit length.

    Args:
        x (array-like): Input vector to normalize

    Returns:
        ndarray: Unit vector in the same direction as input

    Shape transition: (n,) -> (n,) with norm = 1
    """
    return x / norm(x)


def append_row(mat, row):
    """
    Append a row to a matrix.

    Args:
        mat (ndarray): Input matrix of shape (m, n)
        row (array-like): Row to append of shape (n,)

    Returns:
        ndarray: Matrix with appended row of shape (m+1, n)

    Shape transition: (m, n) + (n,) -> (m+1, n)
    """
    return np.append(mat, [row], axis=0)


def get_num_rows(mat):
    """
    Get the number of rows in a matrix.

    Args:
        mat (ndarray): Input matrix of shape (m, n)

    Returns:
        int: Number of rows in the matrix

    Shape query: (m, n) -> m
    """
    return mat.shape[0]


def is_even(x):
    """
    Check if a number is even using modular arithmetic.

    Args:
        x (int): Integer to check

    Returns:
        bool: True if x is even, False otherwise

    Note: Uses (x + 1) % 2 which returns True for even numbers
    """
    return bool((x + 1) % 2)


def is_odd(x):
    """
    Check if a number is odd using modular arithmetic.

    Args:
        x (int): Integer to check

    Returns:
        bool: True if x is odd, False otherwise

    Note: Uses x % 2 which returns True for odd numbers
    """
    return bool(x % 2)


def rotate_90(v, ccw):
    """
    Rotate a 2D vector by 90 degrees.

    Args:
        v (array-like): 2D vector of shape (2,)
        ccw (int): Direction multiplier (1 for CCW, -1 for CW)

    Returns:
        ndarray: Rotated vector of shape (2,)

    Shape transition: (2,) -> (2,)
    Mathematical operation: [x, y] -> ccw * [-y, x]
    """
    rotated_v = np.array((0.0, 0.0))
    rotated_v[0], rotated_v[1] = ccw * -v[1], ccw * v[0]
    return rotated_v


def cyclic(x, a):
    """
    Cyclically shift array elements along the first axis.

    Args:
        x (ndarray): Input array to shift
        a (int): Number of positions to shift (positive = right shift)

    Returns:
        ndarray: Array with cyclically shifted elements

    Shape transition: (n, ...) -> (n, ...) with elements reordered
    """
    return np.roll(x, a, axis=0)


def parse_vertex(v):
    """
    Convert a vertex coordinate to OBJ file format string.

    Args:
        v (array-like): Vertex coordinates of shape (2,) or (3,)

    Returns:
        str: OBJ format vertex line with z-coordinate set to 0.0

    Format: "v x y 0.0\n"
    """
    return "v " + " ".join([str(coord) for coord in v]) + " 0.0\n"


def parse_face(f):
    """
    Convert face indices to OBJ file format string.

    Args:
        f (array-like): Face vertex indices (0-based)

    Returns:
        str: OBJ format face line with 1-based indices

    Format: "f i1// i2// i3// i4//\n"
    Note: Converts from 0-based to 1-based indexing
    """
    return "f " + "// ".join([str(ind + 1) for ind in f]) + "//\n"


def empty_list_of_lists(n):
    """
    Create a list of n empty lists.

    Args:
        n (int): Number of empty lists to create

    Returns:
        list: List containing n empty lists

    Shape creation: Creates structure for n independent sublists
    """
    return [[] for _ in range(n)]


def rotation_matrix(angle):
    """
    Create a 2D rotation matrix for given angle.

    Args:
        angle (float): Rotation angle in radians

    Returns:
        ndarray: 2x2 rotation matrix

    Shape: (2, 2)
    Matrix form: [[cos(θ), -sin(θ)],
                  [sin(θ),  cos(θ)]]
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def identity_matrix(n):
    """
    Create an n×n identity matrix.

    Args:
        n (int): Size of the square identity matrix

    Returns:
        ndarray: n×n identity matrix

    Shape: (n, n) with 1s on diagonal, 0s elsewhere
    """
    return np.eye(n)


def rotation_matrix_3d(angle):
    """
    Create a 3D rotation matrix for rotation around z-axis.

    Args:
        angle (float): Rotation angle in radians around z-axis

    Returns:
        ndarray: 3x3 rotation matrix

    Shape: (3, 3)
    Matrix form: [[cos(θ), -sin(θ), 0],
                  [sin(θ),  cos(θ), 0],
                  [0,       0,      1]]
    """
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_matrix_homog(angle):
    """
    Create a 3*3 homogeneous rotation matrix for 2D rotation.

    Args:
        angle (float): Rotation angle in radians

    Returns:
        ndarray: 3*3 homogeneous transformation matrix

    Shape: (3, 3)
    Matrix form: [[cos(θ), -sin(θ), 0],
                  [sin(θ),  cos(θ), 0],
                  [0,       0,      1]]
    """
    zero_row = np.array([0.0, 0.0])
    homog_col = np.array([[0.0], [0.0], [1.0]])
    return np.hstack([np.vstack([rotation_matrix(angle), zero_row]), homog_col])


def translation_matrix_homog(tx, ty):
    """
    Create a 3*3 homogeneous translation matrix for 2D translation.

    Args:
        tx (float): Translation in x-direction
        ty (float): Translation in y-direction

    Returns:
        ndarray: 3*3 homogeneous transformation matrix

    Shape: (3, 3)
    Matrix form: [[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]]
    """
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def multiply_matrices(mats):
    """
    Multiply a list of matrices from left to right using recursion.

    Args:
        mats (list): List of matrices to multiply

    Returns:
        ndarray: Product of all matrices

    Shape transition: Depends on input matrices, follows matrix multiplication rules
    Note: Modifies input list by consuming elements during recursion
    """
    if len(mats) == 2:
        return np.dot(mats[0], mats[1])
    else:
        mats[-2] = np.dot(mats[-2], mats[-1])
        mats.pop()
        return multiply_matrices(mats)


def rotate_points(points, origin, angle):
    """
    Rotate points around a given origin by specified angle.

    Args:
        points (ndarray): Points to rotate, shape (n, 2) or (n, 3) or (2,) or (3,)
        origin (array-like): Center of rotation, shape (2,) or (3,)
        angle (float): Rotation angle in radians

    Returns:
        ndarray: Rotated points with same shape as input

    Shape transition:
        - Input: (n, d) where d=2 or 3, or (d,)
        - Output: Same shape as input

    Process:
        1. Translate points to origin
        2. Apply rotation matrix
        3. Translate back to original position
    """

    if len(points.shape) == 1:
        onedim = True
        points = np.array([points])
    else:
        onedim = False

    num_points = len(points)

    if points.shape[1] == 3:
        rot_mat = rotation_matrix_3d(angle)
    else:
        rot_mat = rotation_matrix(angle)

    rotated_points = points - np.tile(origin, (num_points, 1))

    for i, point in enumerate(rotated_points):
        rotated_points[i] = np.array((np.matrix(rot_mat) * np.matrix(point).T).T)
    rotated_points += np.tile(origin, (num_points, 1))

    if onedim:
        rotated_points = rotated_points[0]

    return rotated_points


def planar_cross(a, b):
    """
    Calculate the 2D cross product (determinant) of two 2D vectors.

    Args:
        a (array-like): First 2D vector, shape (2,)
        b (array-like): Second 2D vector, shape (2,)

    Returns:
        float: Cross product a[0]*b[1] - a[1]*b[0]

    Shape transition: (2,) * (2,) -> scalar
    Geometric interpretation: Magnitude of z-component of 3D cross product
    """

    return a[0] * b[1] - a[1] * b[0]


def calculate_angle(a, b, c):
    """
    Calculate the angle ABC (angle at point B) in the range [0, 2π).

    Args:
        a (array-like): First point, shape (2,)
        b (array-like): Vertex point (angle measured here), shape (2,)
        c (array-like): Third point, shape (2,)

    Returns:
        float: Angle ABC in radians, range [0, 2π)

    Process:
        1. Create unit vectors from B to A and B to C
        2. Use atan2 of cross and dot products for full angular range
        3. Normalize to [0, 2π) range
    """

    a = np.array(a, copy=True)
    b = np.array(b, copy=True)
    c = np.array(c, copy=True)

    ab_hat = normalize(b - a)
    ac_hat = normalize(c - a)

    x = np.dot(ab_hat, ac_hat)
    y = planar_cross(ab_hat, ac_hat)

    atan2_angle = np.arctan2(y, x)

    return atan2_angle % (2.0 * np.pi)


def shift_points(points, shift):
    """
    Translate all points by a constant shift vector.

    Args:
        points (ndarray): Points to translate, shape (n, d)
        shift (array-like): Translation vector, shape (d,)

    Returns:
        ndarray: Translated points, shape (n, d)

    Shape transition: (n, d) + (d,) -> (n, d)
    Operation: Each point[i] += shift
    """

    return points + np.tile(shift, (len(points), 1))


def plot_structure(points, quads, linkages, ax):
    """
    Plot a kirigami structure showing quadrilateral faces.

    Args:
        points (ndarray): Vertex coordinates, shape (n, 2)
        quads (ndarray): Quad face indices, shape (m, 4)
        linkages (ndarray): Linkage connectivity (unused in current implementation)
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting

    Shape requirements:
        - points: (n_vertices, 2) for 2D coordinates
        - quads: (n_quads, 4) for quad vertex indices

    Visual styling:
        - Fill color: Light peach (1, 229/255, 204/255)
        - Edge color: Black
        - Edge width: 2
        - Alpha: 0.8
    """

    for i, quad in enumerate(quads):
        x = points[quad, 0]
        y = points[quad, 1]
        ax.fill(x, y, color=(1, 229 / 255, 204 / 255), edgecolor="k", linewidth=2, alpha=0.8)

    ax.axis("off")
    ax.set_aspect("equal")


def deployment_linkage2matrix(i, j, num_linkage_rows, num_linkage_cols):
    """
    Convert linkage grid coordinates to matrix row index for deployment analysis.

    Args:
        i (int): Row index in linkage grid (can be negative for boundary)
        j (int): Column index in linkage grid (can be negative for boundary)
        num_linkage_rows (int): Number of rows in the linkage grid
        num_linkage_cols (int): Number of columns in the linkage grid

    Returns:
        tuple or None: (matrix_row_index, label) or None if invalid

    Coordinate system:
        - Bulk linkages: 0 ≤ i < num_rows, 0 ≤ j < num_cols
        - Boundary linkages: Negative indices indicate boundary regions

    Labels: 'bulk', 'left', 'bottom', 'right', 'top'
    """

    if 0 <= i < num_linkage_rows and 0 <= j < num_linkage_cols:  # bulk

        matrix_row_ind = i * num_linkage_cols + j
        label = "bulk"

    else:

        num_bulk_linkages = num_linkage_rows * num_linkage_cols

        num_boundary_linkages = [
            num_linkage_rows,
            num_linkage_cols,
            num_linkage_rows,
            num_linkage_cols,
        ]

        if j == -1 and 0 <= i < num_linkage_rows:  # left
            side_ind = 0
            bound_ind = i
            label = "left"

        elif i == num_linkage_rows and 0 <= j < num_linkage_cols:  # bottom
            side_ind = 1
            bound_ind = j
            label = "bottom"

        elif j == num_linkage_cols and 0 <= i < num_linkage_rows:  # right
            side_ind = 2
            bound_ind = num_linkage_rows - 1 - i
            label = "right"

        elif i == -1 and 0 <= j < num_linkage_cols:  # top
            side_ind = 3
            bound_ind = num_linkage_cols - 1 - j
            label = "top"

        else:  # linkage DNE
            return None

        num_other_boundary_linkages = sum(num_boundary_linkages[:side_ind])

        matrix_row_ind = num_bulk_linkages + num_other_boundary_linkages + bound_ind

    return matrix_row_ind, label


def write_obj(filename, points, quads):
    """
    Write geometry data to Wavefront OBJ file format.

    Args:
        filename (str): Output filename
        points (ndarray): Vertex coordinates, shape (n, 2) or (n, 3)
        quads (ndarray): Quad face indices, shape (m, 4)

    File format:
        - Header: "# n vertices, m faces"
        - Vertices: "v x y 0.0" (z forced to 0.0 for 2D data)
        - Faces: "f i1// i2// i3// i4//" (1-based indexing)
    """

    obj = open(filename, "w")
    obj.write("# {} vertices, {} faces\n".format(get_num_rows(points), get_num_rows(quads)))
    str_points = [parse_vertex(point) for point in points]
    obj.writelines(str_points)
    str_quads = [parse_face(quad) for quad in quads]
    obj.writelines(str_quads)
    obj.close()


def read_obj(filename):
    """
    Read geometry data from Wavefront OBJ file format.

    Args:
        filename (str): Input filename

    Returns:
        tuple: (points, faces) where
            - points: ndarray of shape (n, 2) with vertex coordinates
            - faces: ndarray of shape (m, 4) with face indices (0-based)

    Parsing:
        - Lines starting with 'v': Vertex coordinates
        - Lines starting with 'f': Face indices (converted from 1-based to 0-based)
        - Other lines: Ignored
    """

    obj = open(filename, "r")

    points = []
    faces = []

    for line in obj:

        first_char = line[0]

        if first_char == "v":
            point = [float(_) for _ in line.split(" ")[1:-1]]
            points.append(point)

        elif first_char == "f":
            face = [int(_) for _ in line.replace("//", "").split(" ")[1:]]
            faces.append(face)

        else:
            continue

    obj.close()

    return np.array(points), np.array(faces)


def main():
    """
    Main function for module reloading during development.

    Returns:
        None

    Purpose: Provides feedback when module is reloaded in interactive sessions
    """
    print("reloading Utils")
    return
