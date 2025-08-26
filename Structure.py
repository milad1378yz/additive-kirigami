from Utils import *


class GenericStructure:
    """
    A generic kirigami/origami structure represented as a grid of linkages and quadrilaterals.

    This class provides the foundation for creating and manipulating deployable structures
    based on a regular grid of linkages. Each linkage connects four nodes in a specific pattern,
    and quadrilaterals are formed between linkages to create the deployable surface.

    Attributes:
        num_linkage_rows (int): Number of linkage rows in the grid
        num_linkage_cols (int): Number of linkage columns in the grid
        points (ndarray): Vertex coordinates, shape (n_vertices, 2)
        linkages (ndarray): Linkage connectivity, shape (n_linkages, 4)
        quads (ndarray): Quadrilateral faces, shape (n_quads, 4)
        node2quad_map (list): Maps each node to incident quads
        linkage2quad_map (list): Maps each linkage to incident quads
        node_layers (ndarray): Layer assignment for each node (for collision detection)
        quad_genders (ndarray): Gender assignment for quads (deployment direction)
        hinge_contact_points (list): Contact points for hinges between quads
        hinge_parent_nodes (list): Parent nodes for hinge contact points

    Grid topology:
        - Linkages are arranged in an (m x n) grid
        - Each linkage has 4 nodes arranged in a specific pattern
        - Quadrilaterals are formed between adjacent linkages
        - Total quads: (m+1) * n (horizontal strips of quads)
    """

    def __init__(self, num_linkage_rows, num_linkage_cols):
        """
        Initialize a generic structure with specified grid dimensions.

        Args:
            num_linkage_rows (int): Number of rows in the linkage grid (m)
            num_linkage_cols (int): Number of columns in the linkage grid (n)

        Creates:
            - Linkage topology: (m * n) linkages, each with 4 nodes
            - Quad topology: ((m+1) * n) quads connecting linkages
            - Mapping structures for efficient neighbor queries

        Note: points array is initialized as None and must be set by subclasses
        """
        self.num_linkage_rows = num_linkage_rows
        self.num_linkage_cols = num_linkage_cols
        self.points = None  # Will be set by subclasses after geometry computation
        self.linkages = self.build_linkages()  # Shape: (m*n, 4)
        self.quads = self.build_quads()  # Shape: ((m+1)*n, 4)
        self.node2quad_map = self.build_node2quad_map(self.quads)
        self.linkage2quad_map = self.build_linkage2quad_map(
            self.linkages, self.quads, self.node2quad_map
        )

        # Additional structure properties (initialized by specific methods)
        self.node_layers = None  # Shape: (n_vertices,)
        self.quad_genders = None  # Shape: (n_quads,)
        self.hinge_contact_points = None  # List of arrays for each quad
        self.hinge_parent_nodes = None  # List of arrays for each quad

    def build_linkage(self, i, j):
        """
        Build a single linkage at grid position (i, j).

        Args:
            i (int): Row index in linkage grid
            j (int): Column index in linkage grid

        Returns:
            list: Four node indices forming the linkage [node0, node1, node2, node3]

        Node arrangement in linkage:
            3 ---- 2
            |      |
            |      |
            0 ---- 1

        Shape: Returns 4 node indices for the linkage corners
        """
        return [self.linkage2matrix(i=i, j=j, k=k) for k in range(4)]

    def build_linkages(self):
        """
        Build all linkages in the grid structure.

        Returns:
            ndarray: All linkages, shape (num_linkage_rows * num_linkage_cols, 4)

        Process:
            - Iterates through (i, j) grid positions
            - Creates a linkage at each position using build_linkage()
            - Returns stacked array of all linkage node indices

        Shape transition: Grid (m, n) -> Array (m*n, 4)
        """
        linkages = []
        for i in range(self.num_linkage_rows):
            for j in range(self.num_linkage_cols):
                linkages.append(self.build_linkage(i, j))
        return np.array(linkages)

    def linkage2matrix(self, i, j, k):
        """
        Convert linkage grid coordinates to global node index.

        Args:
            i (int): Row index in linkage grid (can be negative for boundary)
            j (int): Column index in linkage grid (can be negative for boundary)
            k (int): Corner index within linkage (0-3)

        Returns:
            int or None: Global node index, or None if invalid coordinates

        Coordinate system:
            - Bulk linkages: 0 ≤ i < num_rows, 0 ≤ j < num_cols
            - Boundary linkages: Use negative indices
            - Corner k follows: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left

        Shape mapping: (i, j, k) -> global_node_index
        Note: Complex logic handles node sharing between adjacent linkages
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        if 0 <= i < num_linkage_rows and 0 <= j < num_linkage_cols:  # bulk
            if j > 0 and k == 0:
                j -= 1
                k = 2
            if i > 0 and k == 3:
                i -= 1
                k = 1

            inc = 3 - (i > 0)
            num_bulk_rows = i * (2 * num_linkage_cols + 1) + (i > 0) * num_linkage_cols
            num_new_rows = j * inc + (j > 0)
            num_linkage_rows = k - (j > 0)
            matrix_row_ind = num_bulk_rows + num_new_rows + num_linkage_rows

        else:

            num_points = (
                self.linkage2matrix(i=num_linkage_rows - 1, j=num_linkage_cols - 1, k=2) + 1
            )
            num_boundary_points = [
                num_linkage_rows + 1,
                num_linkage_cols + 1,
                num_linkage_rows + 1,
                num_linkage_cols + 1,
            ]

            if j == -1:  # left
                side_ind = 0
                bound_ind = i
                inner_linkage_ind = (i, 0)
                valid_k = [3, 2, 1]

            elif i == num_linkage_rows:  # bottom
                side_ind = 1
                bound_ind = j
                inner_linkage_ind = (num_linkage_rows - 1, j)
                valid_k = [0, 3, 2]

            elif j == num_linkage_cols:  # right
                side_ind = 2
                bound_ind = num_linkage_rows - 1 - i
                inner_linkage_ind = (i, num_linkage_cols - 1)
                valid_k = [1, 0, 3]

            elif i == -1:  # top
                side_ind = 3
                bound_ind = num_linkage_cols - 1 - j
                inner_linkage_ind = (0, j)
                valid_k = [2, 1, 0]

            else:  # linkage DNE
                return None

            num_other_boundary_points = sum(num_boundary_points[:side_ind])

            if k not in valid_k:  # point DNE
                return None
            elif k == valid_k[1]:  # point exists in bulk
                return self.linkage2matrix(
                    i=inner_linkage_ind[0], j=inner_linkage_ind[1], k=(k + 2) % 4
                )
            elif k == valid_k[0]:  # point exists in this boundary
                bound_ind -= 1

            matrix_row_ind = num_points + num_other_boundary_points + bound_ind + 1

        return matrix_row_ind

    @staticmethod
    def is_horizontal_linkage(i, j):
        """
        Determine if a linkage at position (i, j) is horizontally oriented.

        Args:
            i (int): Row index in linkage grid
            j (int): Column index in linkage grid

        Returns:
            bool: True if linkage is horizontal, False if vertical

        Logic:
            - Uses checkerboard pattern: is_even(i) + is_even(j) == 1
            - Alternates orientation to create deployable pattern
            - Horizontal linkages have their long axis along x-direction

        Pattern visualization (H=horizontal, V=vertical):
            H V H V
            V H V H
            H V H V
        """
        return is_even(i) + is_even(j) == 1

    def build_quads(self):
        """
        Build quadrilateral faces connecting adjacent linkages.

        Returns:
            ndarray: Quad connectivity, shape ((num_linkage_rows+1) * num_linkage_cols, 4)

        Process:
            - Creates quads between adjacent linkages in a grid pattern
            - Each quad connects parts of 4 neighboring linkages
            - Uses cyclic shifts to handle horizontal/vertical linkage orientations

        Topology:
            - For each linkage position (i,j), creates quads above it
            - Handles boundary conditions and linkage orientations
            - Total quads: (m+1) * n where m=rows, n=cols

        Shape transition: Linkage grid (m, n) -> Quad array ((m+1)*n, 4)
        """

        quads = []
        for i in range(self.num_linkage_rows + 1):
            for j in range(self.num_linkage_cols):

                linkage = self.build_linkage(i, j)

                top_linkage = self.build_linkage(i - 1, j)
                right_linkage = self.build_linkage(i, j + 1)
                left_linkage = self.build_linkage(i, j - 1)

                top_left_quad = [left_linkage[3], linkage[0], linkage[3], top_linkage[0]]
                top_right_quad = [linkage[3], linkage[2], right_linkage[3], top_linkage[2]]

                if self.is_horizontal_linkage(i, j):
                    top_right_quad = cyclic(top_right_quad, 1)
                else:
                    top_left_quad = cyclic(top_left_quad, 1)

                quads.append(top_left_quad)
                if j == self.num_linkage_cols - 1:
                    quads.append(top_right_quad)

        return np.array(quads)

    @staticmethod
    def build_node2quad_map(quads):
        """
        Build mapping from nodes to incident quadrilaterals.

        Args:
            quads (ndarray): Quad connectivity, shape (n_quads, 4)

        Returns:
            list: For each node, list of incident quad indices
                  Shape: [n_nodes][variable_length]

        Process:
            - Determines total number of nodes from max quad index
            - For each quad, adds quad index to all constituent nodes
            - Creates inverse topology for efficient neighbor queries

        Shape transition: Quads (n_quads, 4) -> Node map [n_nodes][var]
        """

        num_nodes = max(quads.flatten()) + 1
        node2quad_map = empty_list_of_lists(num_nodes)

        for quad_index, quad in enumerate(quads):
            for node_index in quad:
                node2quad_map[node_index].append(quad_index)

        return node2quad_map

    @staticmethod
    def build_linkage2quad_map(linkages, quads, node2quad_map):
        """
        Build mapping from linkages to incident quadrilaterals.

        Args:
            linkages (ndarray): Linkage connectivity, shape (n_linkages, 4)
            quads (ndarray): Quad connectivity, shape (n_quads, 4)
            node2quad_map (list): Node to quad mapping from build_node2quad_map

        Returns:
            list: For each linkage, list of incident quad indices
                  Shape: [n_linkages][4] (each linkage touches 4 quads)

        Process:
            - For each linkage edge (consecutive node pair)
            - Finds quads containing both nodes of the edge
            - Each linkage should connect to exactly 4 quads

        Shape transition: Linkages (n_linkages, 4) -> Linkage map [n_linkages][4]
        """

        linkage2quad_map = empty_list_of_lists(len(linkages))

        for linkage_index, linkage in enumerate(linkages):
            for position, node_index in enumerate(linkage):
                incident_quad_indices = node2quad_map[node_index]
                previous_node_index = linkage[(position - 1) % 4]
                for quad_index in incident_quad_indices:
                    quad = quads[quad_index]
                    if node_index in quad and previous_node_index in quad:
                        linkage2quad_map[linkage_index].append(quad_index)

        return linkage2quad_map

    def layout_linkage(self, linkage_index, phi, phi_index):
        """
        Layout a single linkage with specified deployment angle.

        Args:
            linkage_index (int): Index of linkage to layout
            phi (float): Target deployment angle in radians
            phi_index (int): Starting corner index for angle measurement (0-3)

        Returns:
            tuple: (linkage_quads, quad_points) where:
                - linkage_quads: Quad indices connected to this linkage, shape (4, 4)
                - quad_points: Transformed quad vertex coordinates, shape (16, 2)

        Process:
            1. Extract linkage and connected quads
            2. Measure current deployment angle
            3. Apply rotation to achieve target angle phi
            4. Alternate rotation direction for folding motion

        Shape transitions:
            - Input: linkage_index -> 4 connected quads
            - Output: Transformed coordinates for all 16 quad vertices

        Deployment mechanics:
            - phi controls the "opening" angle of the linkage
            - Rotation sequence creates proper folding kinematics
        """

        linkage2quad_map = self.linkage2quad_map
        linkages = self.linkages
        quads = self.quads
        points = self.points

        quad_inds = linkage2quad_map[linkage_index]

        linkage = linkages[linkage_index]
        linkage_quads = quads[quad_inds]

        next_phi_index = (phi_index + 1) % 4
        last_phi_index = (phi_index - 1) % 4
        opp_phi_index = (phi_index + 2) % 4
        phi_inds = [phi_index, next_phi_index, opp_phi_index, last_phi_index]

        linkage_points = empty_list_of_lists(4)
        quad_points = empty_list_of_lists(4)

        for i, phi_ind in enumerate(phi_inds):
            linkage_points[i] = np.array(points[linkage[phi_ind], :], copy=True)
            quad_points[i] = np.array(points[linkage_quads[phi_ind, :], :], copy=True)

        linkage_points = np.vstack(linkage_points)
        quad_points = np.vstack(quad_points)

        reference_phi = calculate_angle(linkage_points[0], linkage_points[1], linkage_points[-1])
        rotation_angle = reference_phi - phi
        for i in range(1, 4):
            rotation_origin = linkage_points[i - 1]
            quad_points[i * 4 :, :] = rotate_points(
                quad_points[i * 4 :, :], rotation_origin, rotation_angle
            )
            linkage_points[i:, :] = rotate_points(
                linkage_points[i:, :], rotation_origin, rotation_angle
            )

            rotation_angle = -rotation_angle

        quad_points = np.roll(quad_points, 4 * phi_index, axis=0)

        return linkage_quads, quad_points

    def layout(self, phi):
        """
        Deploy the entire structure to a specified linkage angle.

        Args:
            phi (float): Target deployment angle in radians for all linkages

        Returns:
            tuple: (layout_points, mapped_hinge_contact_points) where:
                - layout_points: Deployed vertex coordinates, shape (n_vertices, 2)
                - mapped_hinge_contact_points: Transformed hinge contact points per quad

        Process:
            1. Layout first linkage as anchor
            2. Sequentially layout remaining linkages by matching shared edges
            3. Transform hinge contact points to deployed configuration

        Deployment strategy:
            - Row-by-row layout starting from top-left linkage
            - Each new linkage positioned to maintain edge connectivity
            - Rotation and translation applied to align shared edges

        Shape transitions:
            - Input: phi (scalar angle)
            - Output: Full deployed geometry for visualization/analysis
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        quads = self.quads
        linkages = self.linkages
        linkage2quad_map = self.linkage2quad_map
        points = self.points

        layout_points = np.empty_like(points)

        # layout top left linkage
        linkage_index = 0
        linkage_quads, linkage_points = self.layout_linkage(linkage_index, phi, 0)

        self.store_linkage_layout(
            layout_points, linkage2quad_map, linkage_index, linkage_points, quads
        )
        linkage_index += 1

        # layout remaining linkages
        for i in range(num_linkage_rows):
            for j in range(num_linkage_cols):
                if i == 0 and j == 0:
                    continue
                linkage = linkages[linkage_index]
                if i == 0:
                    phi = calculate_angle(
                        layout_points[linkage[0]],
                        layout_points[linkage[1]],
                        layout_points[linkage[3]],
                    )
                    linkage_quads, linkage_points = self.layout_linkage(linkage_index, phi, 0)
                else:
                    phi = calculate_angle(
                        layout_points[linkage[3]],
                        layout_points[linkage[0]],
                        layout_points[linkage[2]],
                    )
                    linkage_quads, linkage_points = self.layout_linkage(linkage_index, phi, 3)

                linkage_points, shift_origin = self.translate_linkage(
                    layout_points, linkage_points, linkage_quads
                )
                linkage_points = self.rotate_linkage(
                    layout_points, linkage_points, linkage_quads, shift_origin
                )
                self.store_linkage_layout(
                    layout_points, linkage2quad_map, linkage_index, linkage_points, quads
                )

                linkage_index += 1

        # recover hinge contact point layout
        quad_mappings = self.extract_quad_mappings(points, layout_points)
        hinge_contact_points = self.hinge_contact_points
        mapped_hinge_contact_points = empty_list_of_lists(len(quads))
        for i, quad in enumerate(quads):
            local_hcps = hinge_contact_points[i]
            if len(local_hcps) > 0:
                local_hcps_homog = np.hstack([local_hcps, np.ones((len(local_hcps), 1))])
                local_hcps_homog = multiply_matrices([quad_mappings[i], local_hcps_homog.T]).T
                mapped_hinge_contact_points[i] = local_hcps_homog[:, :2]
            else:
                mapped_hinge_contact_points[i] = np.array([])

        return layout_points, mapped_hinge_contact_points

    @staticmethod
    def rotate_linkage(layout_points, linkage_points, linkage_quads, shift_origin):
        rotation_origin = layout_points[linkage_quads[0, 1]]
        rotation_angle = -calculate_angle(shift_origin, rotation_origin, linkage_points[1])
        linkage_points = rotate_points(linkage_points, shift_origin, rotation_angle)
        return linkage_points

    @staticmethod
    def translate_linkage(layout_points, linkage_points, linkage_quads):
        shift_origin = layout_points[linkage_quads[0, 0]]
        shift = shift_origin - linkage_points[0]
        linkage_points = shift_points(linkage_points, shift)
        return linkage_points, shift_origin

    @staticmethod
    def store_linkage_layout(layout_points, linkage2quad_map, linkage_index, linkage_points, quads):
        linkage_node_inds = quads[linkage2quad_map[linkage_index]].flatten()
        for k, node_index in enumerate(linkage_node_inds):
            layout_points[node_index] = linkage_points[k]

    # computes patterns for generic (non necessarily rigid-deployable) structures
    # if quad_gender -> CCW about hinge
    # else -> CW about hinge
    def generic_layout(self, toggle):
        """
        Deploy structure using quad-by-quad rotation method (non-rigid deployment).

        Args:
            toggle (bool): Deployment direction toggle
                          True: Use original quad genders
                          False: Use inverted quad genders

        Returns:
            ndarray: Deployed vertex coordinates, shape (n_vertices, 2)

        Process:
            1. Start with first quad in reference position
            2. For each subsequent quad, rotate around shared hinge
            3. Rotation direction determined by quad gender and toggle

        Quad gender meanings:
            - True (CCW): Counter-clockwise rotation about hinge
            - False (CW): Clockwise rotation about hinge

        Note: This method allows non-rigid deployment where structure
              may not maintain exact edge lengths during motion

        Shape transition: Reference geometry -> Deployed geometry
        """

        points = self.points
        quads = self.quads
        quad_genders = self.quad_genders
        height = self.num_linkage_rows
        width = self.num_linkage_cols

        rolled_quads = np.zeros_like(quads)
        for i, (quad, gender) in enumerate(zip(quads, quad_genders)):
            if not gender:
                rolled_quads[i, :] = np.roll(quad, -1)
            else:
                rolled_quads[i, :] = quad

        layout_points = np.zeros_like(points)
        layout_points[quads[0], :] = points[quads[0], :]

        for i in range(height + 1):
            for j in range(width + 1):
                if i == 0 and j == 0:
                    continue

                quad_ind = i * (width + 1) + j
                quad = rolled_quads[quad_ind]
                quad_gender = quad_genders[quad_ind] if toggle else not quad_genders[quad_ind]
                parent_quad_ind = (i - (j == 0)) * (width + 1) + j - 1 + (j == 0)
                parent_quad = rolled_quads[parent_quad_ind]
                hinge_node_ind = quad[(0 - (j == 0)) % 4]
                if quad_gender:
                    arm_node_ind = quad[3 - (j == 0)]
                    target_node_ind = parent_quad[3 - (j == 0)]
                else:
                    arm_node_ind = quad[1 - (j == 0)]
                    target_node_ind = parent_quad[1 - (j == 0)]

                arm = points[arm_node_ind, :] - points[hinge_node_ind, :]
                home = layout_points[target_node_ind, :] - layout_points[hinge_node_ind, :]
                origin = layout_points[hinge_node_ind, :]
                position = points[hinge_node_ind, :] - origin
                angle = calculate_angle([0, 0], arm, home)

                quad_points = points[quad, :]
                quad_points = shift_points(quad_points, -position)
                quad_points = rotate_points(quad_points, origin, angle)

                layout_points[quad, :] = quad_points

        return layout_points

    def assign_node_layers(self):
        """
        Assign layer indices to nodes for collision detection and rendering order.

        Updates:
            self.node_layers: ndarray of shape (n_vertices,) with layer assignments

        Layer assignment logic:
            - Based on linkage position and corner index within linkage
            - Alternating pattern ensures proper stacking order
            - Values: -1, 0, 1 indicating relative depth

        Purpose:
            - Prevents visual/physical collisions during deployment
            - Defines which quad appears "on top" at intersections
            - Critical for realistic kirigami folding simulation

        Shape created: (n_vertices,) with integer layer indices
        """

        num_rows = self.num_linkage_rows
        num_cols = self.num_linkage_cols
        points = self.points

        num_points = len(points)
        node_layers = np.array([0] * num_points, copy=True)

        for i in range(num_rows):
            for j in range(num_cols):
                linkage = self.build_linkage(i, j)
                if j == 0:
                    node_layers[linkage[0]] = -1
                node_layers[linkage[2]] = -(2 * is_odd(j) - 1)

                if i == 0:
                    node_layers[linkage[3]] = 1
                node_layers[linkage[1]] = 2 * is_odd(i) - 1

        self.node_layers = node_layers

    def get_node_layer(self, node_index):
        return self.node_layers[node_index]

    def assign_quad_genders(self):
        """
        Assign deployment direction (gender) to each quadrilateral.

        Updates:
            self.quad_genders: ndarray of shape (n_quads,) with gender assignments

        Gender meanings:
            - 0: Clockwise deployment direction
            - 1: Counter-clockwise deployment direction

        Assignment pattern:
            - Based on linkage orientation (horizontal vs vertical)
            - Creates alternating pattern for proper kirigami motion
            - Ensures neighboring quads have compatible deployment

        Purpose:
            - Defines local rotation direction for each quad
            - Critical for correct folding kinematics
            - Prevents self-intersection during deployment

        Shape created: (n_quads,) with binary gender values
        """

        quads = self.quads
        linkage2quad_map = self.linkage2quad_map
        num_rows = self.num_linkage_rows
        num_cols = self.num_linkage_cols

        quad_genders = np.array([-1] * len(quads), copy=True)
        linkage_index = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if is_odd(i) + is_odd(j) == 1:
                    for k in range(4):
                        quad_genders[linkage2quad_map[linkage_index][k]] = int(is_even(k))
                else:
                    for k in range(4):
                        quad_genders[linkage2quad_map[linkage_index][k]] = int(is_odd(k))
                linkage_index += 1

        self.quad_genders = quad_genders

    def get_quad_gender(self, quad_index):
        return self.quad_genders[quad_index]

    def make_hinge_contact_points(self):
        """
        Generate contact points along hinge edges between quadrilaterals.

        Updates:
            self.hinge_contact_points: List of arrays, one per quad
            self.hinge_parent_nodes: List of arrays, parent nodes for each contact point

        Process:
            1. For each linkage, identify longer and shorter edge pairs
            2. Create contact points offset from linkage corners
            3. Assign contact points to adjacent quads
            4. Store parent node information for tracking

        Contact point geometry:
            - Positioned along quad edges adjacent to linkages
            - Offset by linkage edge length in perpendicular direction
            - Two contact points per quad per adjacent linkage

        Purpose:
            - Models physical contact during deployment
            - Critical for collision detection
            - Enables realistic simulation of folding constraints

        Shape created:
            - hinge_contact_points: [n_quads][variable] of 2D points
            - hinge_parent_nodes: [n_quads][variable] of node indices
        """

        points = self.points
        linkages = self.linkages
        quads = self.quads
        linkage2quad_map = self.linkage2quad_map

        num_quads = len(quads)

        hinge_contact_points = empty_list_of_lists(num_quads)
        hinge_parent_nodes = empty_list_of_lists(num_quads)

        for i, linkage in enumerate(linkages):
            es = [points[linkage[(j + 1) % 4]] - points[linkage[j]] for j in range(4)]
            s = norm(es[0]) > norm(es[1])

            quad_a_ind = linkage2quad_map[i][0 + s]
            quad_b_ind = linkage2quad_map[i][2 + s]

            quad_a_point1 = points[linkage[(3 + s) % 4]] - norm(es[0 + s]) * normalize(es[1 + s])
            quad_a_point2 = points[linkage[0 + s]] + norm(es[0 + s]) * normalize(es[1 + s])

            hinge_contact_points[quad_a_ind].append(quad_a_point1)
            hinge_contact_points[quad_a_ind].append(quad_a_point2)

            hinge_parent_nodes[quad_a_ind].append(linkage[2 + s])
            hinge_parent_nodes[quad_a_ind].append(linkage[1 + s])

            quad_b_point1 = points[linkage[1 + s]] + norm(es[0 + s]) * normalize(es[1 + s])
            quad_b_point2 = points[linkage[2 + s]] - norm(es[0 + s]) * normalize(es[1 + s])

            hinge_contact_points[quad_b_ind].append(quad_b_point1)
            hinge_contact_points[quad_b_ind].append(quad_b_point2)

            hinge_parent_nodes[quad_b_ind].append(linkage[0 + s])
            hinge_parent_nodes[quad_b_ind].append(linkage[(3 + s) % 4])

        for i in range(num_quads):
            if len(hinge_contact_points[i]) > 0:
                hinge_contact_points[i] = np.vstack(hinge_contact_points[i])
            else:
                hinge_contact_points[i] = np.array([])

        self.hinge_contact_points = hinge_contact_points
        self.hinge_parent_nodes = hinge_parent_nodes

    def get_hinge_contact_points(self, quad_inds):
        if type(quad_inds) is list:
            hinge_contact_points = empty_list_of_lists(len(quad_inds))
            for i, ind in enumerate(quad_inds):
                if len(self.hinge_contact_points[ind]) > 0:
                    hinge_contact_points[i] = np.array(self.hinge_contact_points[ind], copy=True)
            return hinge_contact_points
        return self.hinge_contact_points[quad_inds]

    def get_hinge_parent_nodes(self, quad_inds):
        if type(quad_inds) is list:
            hinge_parent_nodes = empty_list_of_lists(len(quad_inds))
            for i, ind in enumerate(quad_inds):
                if len(self.hinge_parent_nodes[ind]) > 0:
                    hinge_parent_nodes[i] = np.array(self.hinge_parent_nodes[ind], copy=True)
            return hinge_parent_nodes
        return self.hinge_parent_nodes[quad_inds]

    def get_boundary_linkages(self, bound_ind):
        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        if bound_ind == 0:  # left
            return [(i, 0) for i in range(num_linkage_rows)]
        if bound_ind == 1:  # bottom
            return [(num_linkage_rows - 1, j) for j in range(num_linkage_cols)]
        if bound_ind == 2:  # right
            return [(i, num_linkage_cols - 1) for i in range(num_linkage_rows - 1, -1, -1)]
        if bound_ind == 3:  # top
            return [(0, j) for j in range(num_linkage_cols - 1, -1, -1)]

    def get_outer_boundary_linkages(self, bound_ind):
        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        if bound_ind == 0:  # left
            return [(i, -1) for i in range(num_linkage_rows)]
        if bound_ind == 1:  # bottom
            return [(num_linkage_rows, j) for j in range(num_linkage_cols)]
        if bound_ind == 2:  # right
            return [(i, num_linkage_cols) for i in range(num_linkage_rows - 1, -1, -1)]
        if bound_ind == 3:  # top
            return [(-1, j) for j in range(num_linkage_cols - 1, -1, -1)]

    def get_boundary_node_inds(self, bound_ind):
        linkage_inds = self.get_outer_boundary_linkages(bound_ind)
        boundary_node_inds = []
        for i, j in linkage_inds:
            linkage = self.build_linkage(i, j)
            local_inds = [_ % 4 for _ in range(bound_ind + 1, bound_ind + 4)[::-1]]
            if (self.is_horizontal_linkage(i, j) and is_even(bound_ind)) or (
                not self.is_horizontal_linkage(i, j) and is_odd(bound_ind)
            ):
                local_inds.pop(1)
            boundary_node_inds.extend([linkage[_] for _ in local_inds])

        return boundary_node_inds

    def get_outer_boundary_node_inds(self, bound_ind):

        linkage_inds = self.get_outer_boundary_linkages(bound_ind)
        boundary_node_inds = []
        for i, j in linkage_inds:

            linkage = self.build_linkage(i, j)

            if (self.is_horizontal_linkage(i, j) and is_odd(bound_ind)) or (
                (not self.is_horizontal_linkage(i, j)) and is_even(bound_ind)
            ):

                boundary_node_inds.append(linkage[(bound_ind + 2) % 4])

        return boundary_node_inds

    def linkage2quad(self, i, j, k):

        quad_ind = i * (self.num_linkage_cols + 1) + j
        if k == 1 or k == 2:
            quad_ind += self.num_linkage_cols + k
        if k == 3:
            quad_ind += 1

        return quad_ind

    def is_linkage_parallel_to_boundary(self, i, j, bound_ind):
        h = self.is_horizontal_linkage(i, j)
        return (h and is_odd(bound_ind)) or ((not h) and is_even(bound_ind))

    def get_dual_corner_angles(self):

        num_linkage_rows = self.num_linkage_rows
        num_linkage_cols = self.num_linkage_cols
        quads = self.quads

        dual_corner_angles = []

        corner_linkage_inds = [
            [0, 0],
            [num_linkage_rows - 1, 0],
            [num_linkage_rows - 1, num_linkage_cols - 1],
            [0, num_linkage_cols - 1],
        ]
        for corner_ind, (i, j) in enumerate(corner_linkage_inds):
            corner_quad = quads[self.linkage2quad(i, j, corner_ind)]

            dual_corner_ind = corner_ind + 2 * is_even(corner_ind) - 1
            corner_angle = [
                corner_quad[(dual_corner_ind - 1) % 4],
                corner_quad[dual_corner_ind],
                corner_quad[(dual_corner_ind + 1) % 4],
            ]
            dual_corner_angles.append(corner_angle)

        return dual_corner_angles

    def get_dual_boundary_angles(self):

        quads = self.quads

        dual_boundary_angles = []
        for bound_ind in range(4):
            boundary_linkage_inds = self.get_boundary_linkages(bound_ind)
            local_boundary_angles = []
            for i, j in boundary_linkage_inds:

                boundary_quad1 = np.array(quads[self.linkage2quad(i, j, bound_ind)], copy=True)
                boundary_quad2 = np.array(
                    quads[self.linkage2quad(i, j, (bound_ind + 1) % 4)], copy=True
                )

                if self.is_linkage_parallel_to_boundary(i, j, bound_ind):
                    boundary_quad1 = cyclic(boundary_quad1, -1)
                    boundary_quad2 = cyclic(boundary_quad2, 1)
                else:
                    boundary_quad1 = cyclic(boundary_quad1, 1)
                    boundary_quad2 = cyclic(boundary_quad2, -1)

                boundary_angle1 = boundary_quad1[[_ % 4 for _ in range(bound_ind, bound_ind + 3)]]
                boundary_angle2 = boundary_quad2[
                    [_ % 4 for _ in range(bound_ind - 1, bound_ind + 2)]
                ]

                local_boundary_angles.append(boundary_angle1)
                local_boundary_angles.append(boundary_angle2)
            dual_boundary_angles.append(np.vstack(local_boundary_angles))

        return dual_boundary_angles

    def get_dual_boundary_node_inds(self, bound_ind):
        """
        Get node indices along the outer boundary of the structure.
        Args:
            bound_ind (int): Boundary index (0: left, 1: bottom, 2: right, 3: top)
            Returns:
            list: Node indices along the outer boundary in counter-clockwise order
        """

        linkage_inds = self.get_outer_boundary_linkages(bound_ind)
        boundary_node_inds = []
        for i, j in linkage_inds:
            linkage = self.build_linkage(i, j)

            if not self.is_linkage_parallel_to_boundary(i, j, bound_ind):

                local_inds = [_ % 4 for _ in range(bound_ind + 1, bound_ind + 4)[::-1]]

                boundary_node_inds.extend([linkage[_] for _ in local_inds])

        return boundary_node_inds

    def extract_quad_mappings(self, points_domain, points_image):

        quads = self.quads
        affine_matrices = []

        for i, quad in enumerate(quads):

            origin_domain = points_domain[quad[0]]
            origin_image = points_image[quad[0]]

            arm_domain = points_domain[quad[1]] - origin_domain
            arm_image = points_image[quad[1]] - origin_image

            angle = calculate_angle(np.array([0.0, 0.0]), arm_domain, arm_image)

            trans_domain_2_home = translation_matrix_homog(-origin_domain[0], -origin_domain[1])
            rot_home = rotation_matrix_homog(angle)
            trans_home_2_image = translation_matrix_homog(origin_image[0], origin_image[1])

            affine_matrices.append(
                multiply_matrices([trans_home_2_image, rot_home, trans_domain_2_home])
            )

        return affine_matrices


# ------------------------------------------------------- MATRIX -------------------------------------------------------


class MatrixStructure(GenericStructure):
    """
    A kirigami structure using linear matrix-based design approach.

    This class extends GenericStructure to provide a linear algebraic method
    for generating structure geometry. The approach uses a design matrix that
    linearly combines seed points to generate all vertex positions.

    Key features:
        - Linear design matrix approach for efficient geometry generation
        - Support for interior and boundary offset parameters
        - Inverse design capability for target boundary shapes
        - Exact linear relationships between seed points and final geometry

    Mathematical approach:
        - Design matrix D maps seed points to all vertex coordinates
        - points = D @ seed_points where seed_points = [top_row, left_col, corners]
        - Boundary and interior offsets control local geometry variations

    Applications:
        - Rapid prototyping of kirigami patterns
        - Inverse design for specific boundary shapes
        - Linear optimization of structural parameters
    """

    def __init__(self, num_linkage_rows, num_linkage_cols):
        """
        Initialize a matrix-based structure.

        Args:
            num_linkage_rows (int): Number of linkage rows in grid
            num_linkage_cols (int): Number of linkage columns in grid

        Inherits all topology from GenericStructure and adds matrix-based
        geometry generation capabilities.
        """
        GenericStructure.__init__(self, num_linkage_rows, num_linkage_cols)

    def calculate_design_matrix(self, offsets, boundary_offsets):
        """
        Calculate the linear design matrix for structure geometry generation.

        Args:
            offsets (ndarray): Interior offset parameters, shape (num_rows, num_cols)
            boundary_offsets (list): Boundary offset parameters for each edge
                                   [left_offsets, bottom_offsets, right_offsets, top_offsets]

        Returns:
            ndarray: Design matrix of shape (m, n) where:
                    m = total number of vertices (bulk + boundary)
                    n = number of seed parameters (top + left + corners)

        Matrix structure:
            - Maps seed points to all vertex coordinates via linear combination
            - Seed points: top row nodes + left column nodes + 4 corner nodes
            - Each vertex position computed as weighted sum of seed points
            - Offsets control local interpolation weights

        Shape transitions:
            - offsets: (num_rows, num_cols) -> influences bulk vertex generation
            - boundary_offsets: 4 lists -> influences boundary vertex generation
            - Output: (total_vertices, seed_points) matrix
        """

        num_linkage_rows = self.num_linkage_rows
        num_linkage_cols = self.num_linkage_cols
        assert offsets.shape[0] == num_linkage_rows
        assert offsets.shape[1] == num_linkage_cols

        # design_matrix = np.zeros(self.design_matrix_dims(), dtype=np.float128)
        design_matrix = np.zeros(self.design_matrix_dims(), dtype=np.float64)

        # SEED
        for j in range(num_linkage_cols):
            self.set_identity(design_matrix, self.linkage2matrix(i=0, j=j, k=3), j)
        for i in range(num_linkage_rows):
            self.set_identity(
                design_matrix, self.linkage2matrix(i=i, j=0, k=0), num_linkage_cols + i
            )

        shift = num_linkage_cols + num_linkage_rows
        self.set_identity(design_matrix, self.linkage2matrix(i=0, j=-1, k=3), shift)
        self.set_identity(
            design_matrix, self.linkage2matrix(i=num_linkage_rows, j=0, k=0), shift + 1
        )
        self.set_identity(
            design_matrix,
            self.linkage2matrix(i=num_linkage_rows - 1, j=num_linkage_cols, k=1),
            shift + 2,
        )
        self.set_identity(
            design_matrix, self.linkage2matrix(i=-1, j=num_linkage_cols - 1, k=2), shift + 3
        )

        # BULK
        for i in range(num_linkage_rows):
            for j in range(num_linkage_cols):
                for k in range(1, 3):
                    checkerboard = self.is_horizontal_linkage(i=i, j=j)
                    row_ind = self.linkage2matrix(i=i, j=j, k=k)
                    end_row_ind = self.linkage2matrix(i=i, j=j, k=3 * (not checkerboard))
                    middle_row_ind = self.linkage2matrix(i=i, j=j, k=3 * checkerboard)
                    coeff = (
                        3.0 * (not checkerboard)
                        + (-2.0 * (not checkerboard) + 1.0) * float(k)
                        + offsets[i, j]
                    )
                    self.set_row(design_matrix, coeff, row_ind, end_row_ind, middle_row_ind)

        # BOUNDARY
        for bound_ind in range(4):
            for z, (i, j) in enumerate(self.get_outer_boundary_linkages(bound_ind)):
                h = self.is_horizontal_linkage(i, j)
                offset = boundary_offsets[bound_ind][z]
                growth_k = (bound_ind + 1) % 4
                if is_odd(bound_ind):
                    coeff = 1.0 + float(h) + offset
                    other_k = [(growth_k + 2) % 4, (growth_k + 1) % 4]
                else:
                    coeff = 2.0 - float(h) + offset
                    other_k = [(growth_k + 1) % 4, (growth_k + 2) % 4]
                row_ind = self.linkage2matrix(i=i, j=j, k=growth_k)
                end_row_ind = self.linkage2matrix(i=i, j=j, k=other_k[1 - h])
                middle_row_ind = self.linkage2matrix(i=i, j=j, k=other_k[h])
                self.set_row(design_matrix, coeff, row_ind, end_row_ind, middle_row_ind)

        return design_matrix

    @staticmethod
    def set_row(design_matrix, coeff, row_ind, end_row_ind, middle_row_ind):
        """
        Set a row in the design matrix using linear interpolation.

        Args:
            design_matrix (ndarray): Design matrix to modify
            coeff (float): Interpolation coefficient [0, 1]
            row_ind (int): Target row index to set
            end_row_ind (int): Index of first interpolation row
            middle_row_ind (int): Index of second interpolation row

        Operation:
            design_matrix[row_ind] = (1 - coeff) * end_row + coeff * middle_row

        Purpose:
            - Creates linear combination rows in design matrix
            - Coefficient controls blend between two reference rows
            - Enables smooth transitions in generated geometry
        """
        end_row = design_matrix[end_row_ind, :]
        middle_row = design_matrix[middle_row_ind, :]
        design_matrix[row_ind, :] = (1.0 - coeff) * end_row + coeff * middle_row

    @staticmethod
    def set_identity(design_matrix, row_ind, col_ind):
        """
        Set an identity entry in the design matrix.

        Args:
            design_matrix (ndarray): Design matrix to modify
            row_ind (int): Row index for identity entry
            col_ind (int): Column index for identity entry

        Operation:
            design_matrix[row_ind, col_ind] = 1.0

        Purpose:
            - Creates direct mapping from seed point to vertex
            - Establishes anchor points for matrix interpolation
            - Ensures seed points appear unchanged in final geometry
        """
        design_matrix[row_ind, col_ind] = 1.0

    def design_matrix_dims(self):
        """
        Calculate dimensions of the design matrix.

        Returns:
            tuple: (m, n) where:
                  m = total number of vertices (rows)
                  n = number of seed parameters (columns)

        Matrix size calculation:
            - Seed points: num_cols + num_rows (top + left edges) + 4 (corners)
            - Bulk points: 2 * num_rows * num_cols (interior linkage nodes)
            - Boundary points: 2 * (num_rows + num_cols + 2) (boundary linkage nodes)

        Shape: (total_vertices, seed_parameters)
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        num_seed_points = num_linkage_cols + num_linkage_rows
        num_bulk_points = 2 * num_linkage_rows * num_linkage_cols
        num_boundary_points = 2 * (num_linkage_rows + num_linkage_cols + 2)

        m = num_seed_points + num_bulk_points + num_boundary_points
        n = num_linkage_rows + num_linkage_cols + 4

        return m, n

    def build_structure(self, top_points, left_points, corners, offsets, boundary_offsets):
        """
        Build structure geometry from seed points using design matrix.

        Args:
            top_points (ndarray): Top edge seed points, shape (num_cols, 2)
            left_points (ndarray): Left edge seed points, shape (num_rows, 2)
            corners (ndarray): Corner seed points, shape (4, 2)
            offsets (ndarray): Interior offset parameters, shape (num_rows, num_cols)
            boundary_offsets (list): Boundary offset parameters for each edge

        Updates:
            self.points: All vertex coordinates computed via matrix multiplication

        Process:
            1. Calculate design matrix from offsets
            2. Stack seed points into column vector
            3. Multiply design matrix by seed points to get all coordinates

        Shape transitions:
            - Seed points: Various shapes -> Stacked column vector
            - Design matrix @ seed vector -> All vertex coordinates
        """
        design_matrix = self.calculate_design_matrix(offsets, boundary_offsets)
        self.points = np.dot(design_matrix, np.vstack((top_points, left_points, corners)))

    def linear_inverse_design(self, boundary_points, corners, interior_offsets, boundary_offsets):
        """
        Inverse design: find seed points that produce target boundary shape.

        Args:
            boundary_points (ndarray): Target boundary vertex coordinates
            corners (ndarray): Fixed corner points, shape (4, 2)
            interior_offsets (ndarray): Interior offset parameters, shape (num_rows, num_cols)
            boundary_offsets (list): Boundary offset parameters for each edge

        Updates:
            self.points: Structure geometry achieving target boundary

        Process:
            1. Extract boundary rows from design matrix
            2. Form linear system: boundary_matrix @ seed_points = boundary_points
            3. Solve for top and left seed points via matrix inverse
            4. Build full structure with computed seed points

        Mathematical approach:
            - Reduces to linear system solving
            - Guarantees exact boundary matching (if feasible)
            - Enables target-driven design workflow

        Shape transitions:
            - Target boundary -> Computed seed points -> Full structure
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        design_matrix = self.calculate_design_matrix(interior_offsets, boundary_offsets)

        bound_inds = []
        for bound_ind in range(4):
            bound_nodes = self.get_outer_boundary_node_inds(bound_ind)
            bound_inds.extend(bound_nodes)

        sub_design_matrix = design_matrix[bound_inds, : (num_linkage_cols + num_linkage_rows)]

        if np.abs(np.linalg.det(sub_design_matrix)) <= 1e-6:
            print("This Matrix is not invertible")
        # print("Det:", np.linalg.det(sub_design_matrix))
        sub_design_matrix_inverse = np.linalg.inv(sub_design_matrix)
        inverted_seed_points = np.dot(sub_design_matrix_inverse, boundary_points)

        # inverted_seed_points = np.linalg.solve(sub_design_matrix, boundary_points)

        # inverted_seed_points, *_ = np.linalg.lstsq(sub_design_matrix, boundary_points, rcond=None)
        top_points = inverted_seed_points[:num_linkage_cols, :]
        left_points = inverted_seed_points[
            num_linkage_cols : num_linkage_cols + num_linkage_rows, :
        ]

        self.build_structure(top_points, left_points, corners, interior_offsets, boundary_offsets)


# ------------------------------------------------------- DEPLOYED MATRIX ----------------------------------------------


class DeployedMatrixStructure(GenericStructure):
    """
    A deployable kirigami structure using matrix-based design with deployment angles.

    This class extends GenericStructure to provide matrix-based geometry generation
    that incorporates deployment/folding angles. Unlike MatrixStructure which only
    handles flat configurations, this class can generate deployed 3D configurations.

    Key features:
        - Matrix-based approach with deployment angle parameters
        - 2D design matrix for full coordinate generation
        - Support for uniform or spatially-varying deployment angles
        - Inverse design capability for deployed configurations

    Mathematical approach:
        - Extended design matrix maps seed points to deployed coordinates
        - Deployment angles (phi) control local folding at each linkage
        - Each coordinate (x,y) handled separately in matrix formulation
        - Rotation matrices embedded in design matrix construction

    Applications:
        - Simulating deployed kirigami structures
        - Inverse design for target deployed shapes
        - Analysis of deployment motion and constraints
    """

    def __init__(self, num_linkage_rows, num_linkage_cols):
        """
        Initialize a deployed matrix-based structure.

        Args:
            num_linkage_rows (int): Number of linkage rows in grid
            num_linkage_cols (int): Number of linkage columns in grid

        Inherits topology from GenericStructure and adds deployed geometry
        generation capabilities with angle-dependent design matrices.
        """
        GenericStructure.__init__(self, num_linkage_rows, num_linkage_cols)

    def calculate_design_matrix(self, offsets, boundary_offsets, phis, boundary_phis=None):
        """
        Calculate design matrix for deployed structure geometry generation.

        Args:
            offsets (ndarray): Interior offset parameters, shape (num_rows, num_cols)
            boundary_offsets (list): Boundary offset parameters for each edge
            phis (ndarray or float): Deployment angles in radians
                                   If float: uniform angle for all linkages
                                   If array: shape (num_rows, num_cols) for per-linkage angles
            boundary_phis (ndarray, optional): Boundary deployment angles
                                             If None, computed automatically from phis

        Returns:
            ndarray: Design matrix of shape (2*m, 2*n) where:
                    2*m = total coordinates (x,y for each vertex)
                    2*n = seed coordinate parameters (x,y for each seed point)

        Design matrix structure:
            - Handles x,y coordinates separately in 2D block structure
            - Rotation matrices embedded based on deployment angles
            - Each linkage contributes rotation-based relationships
            - Boundary conditions handled with deployment constraints

        Shape transitions:
            - Input angles: scalar or (num_rows, num_cols)
            - Output matrix: (2*total_vertices, 2*seed_points)
            - Enables 2D coordinate generation from 2D seed points
        """

        num_linkage_rows = self.num_linkage_rows
        num_linkage_cols = self.num_linkage_cols
        assert offsets.shape[0] == num_linkage_rows
        assert offsets.shape[1] == num_linkage_cols

        if type(phis) is float:
            print(
                "calculate_design_matrix:\n    phis is float, building deployment arrays using that"
            )
            phi = phis
            phis = np.zeros(offsets.shape)
            for i in range(num_linkage_rows):
                for j in range(num_linkage_cols):
                    if self.is_horizontal_linkage(i=i, j=j):
                        phis[i, j] = phi  # left angle
                    else:
                        phis[i, j] = np.pi - phi  # left angle

            boundary_phis = np.array(
                [
                    [0.0] * num_linkage_rows,
                    [0.0] * num_linkage_cols,
                    [0.0] * num_linkage_rows,
                    [0.0] * num_linkage_cols,
                ],
                dtype=object,
            )
            for bound_ind in range(4):
                for z, (i, j) in enumerate(self.get_outer_boundary_linkages(bound_ind)):
                    if self.is_horizontal_linkage(i, j):
                        boundary_phis[bound_ind][z] = np.pi - phi
                    else:
                        boundary_phis[bound_ind][z] = phi

        design_matrix = np.zeros(self.design_matrix_dims(), dtype=np.float64)

        # SEED
        for j in range(num_linkage_cols):
            self.set_rows_identity(design_matrix, 2 * self.linkage2matrix(i=0, j=j, k=3), 2 * j)
        for i in range(num_linkage_rows):
            self.set_rows_identity(
                design_matrix, 2 * self.linkage2matrix(i=i, j=0, k=0), 2 * (num_linkage_cols + i)
            )

        shift = num_linkage_cols + num_linkage_rows
        self.set_rows_identity(design_matrix, 2 * self.linkage2matrix(i=0, j=-1, k=3), 2 * shift)
        self.set_rows_identity(
            design_matrix, 2 * self.linkage2matrix(i=num_linkage_rows, j=0, k=0), 2 * (shift + 1)
        )
        self.set_rows_identity(
            design_matrix,
            2 * self.linkage2matrix(i=num_linkage_rows - 1, j=num_linkage_cols, k=1),
            2 * (shift + 2),
        )
        self.set_rows_identity(
            design_matrix,
            2 * self.linkage2matrix(i=-1, j=num_linkage_cols - 1, k=2),
            2 * (shift + 3),
        )

        # BULK
        for i in range(num_linkage_rows):
            for j in range(num_linkage_cols):
                phi = phis[i, j]
                bottom_angle = phi - np.pi
                right_angle = phi

                row_ind_left, row_ind_bottom, row_ind_right, row_ind_top = [
                    2 * self.linkage2matrix(i=i, j=j, k=k) for k in range(4)
                ]

                self.set_rows(
                    design_matrix,
                    offsets[i, j],
                    row_ind_bottom,
                    row_ind_left,
                    row_ind_top,
                    bottom_angle,
                )
                self.set_rows(
                    design_matrix,
                    offsets[i, j],
                    row_ind_right,
                    row_ind_top,
                    row_ind_left,
                    right_angle,
                )

        # BOUNDARY
        for bound_ind in range(4):
            for z, (i, j) in enumerate(self.get_outer_boundary_linkages(bound_ind)):

                offset = boundary_offsets[bound_ind][z]
                growth_k = (bound_ind + 1) % 4
                existing_k = [(growth_k + 2) % 4, (growth_k + 1) % 4]

                phi = boundary_phis[bound_ind][z]
                if is_even(bound_ind):
                    angle = phi
                else:
                    angle = np.pi - phi
                growth_row_ind = 2 * self.linkage2matrix(i=i, j=j, k=growth_k)
                last_last_row_ind = 2 * self.linkage2matrix(i=i, j=j, k=existing_k[0])
                last_row_ind = 2 * self.linkage2matrix(i=i, j=j, k=existing_k[1])

                self.set_rows(
                    design_matrix, offset, growth_row_ind, last_row_ind, last_last_row_ind, angle
                )

        return design_matrix

    @staticmethod
    def set_rows(design_matrix, offset, row_ind_growth, row_ind_origin, row_ind_arm, angle):
        """
        Set matrix rows using rotation-based relationships for deployment.

        Args:
            design_matrix (ndarray): Design matrix to modify
            offset (float): Growth offset parameter
            row_ind_growth (int): Index for new point being positioned
            row_ind_origin (int): Index for rotation origin point
            row_ind_arm (int): Index for rotation arm point
            angle (float): Rotation angle in radians

        Operation:
            Sets 2x2 block in design matrix encoding rotational relationship:
            new_point = origin + (1 + offset) * rotation_matrix(angle) @ (arm - origin)

        Purpose:
            - Encodes kinematic relationships in deployed configuration
            - Rotation matrices capture folding motion constraints
            - Offset parameter allows local geometry variation

        Shape: Modifies 2x2 block starting at (row_ind_growth, various_cols)
        """

        rot_mat = rotation_matrix(angle)
        eye_mat = identity_matrix(2)

        ori_mat = eye_mat - (1.0 + offset) * rot_mat
        arm_mat = (1.0 + offset) * rot_mat

        origin_rows = design_matrix[row_ind_origin : row_ind_origin + 2, :]
        arm_rows = design_matrix[row_ind_arm : row_ind_arm + 2, :]

        design_matrix[row_ind_growth : row_ind_growth + 2, :] = np.dot(
            ori_mat, origin_rows
        ) + np.dot(arm_mat, arm_rows)

    @staticmethod
    def set_rows_identity(design_matrix, row_ind, col_ind):
        """
        Set 2x2 identity block in design matrix for seed points.

        Args:
            design_matrix (ndarray): Design matrix to modify
            row_ind (int): Starting row index for 2x2 block
            col_ind (int): Starting column index for 2x2 block

        Operation:
            Sets identity block: [[1,0],[0,1]] at specified location

        Purpose:
            - Establishes direct mapping for seed coordinates
            - Ensures seed points pass through unchanged
            - Creates anchor points for matrix interpolation

        Shape: Modifies 2x2 block at (row_ind:row_ind+2, col_ind:col_ind+2)
        """
        design_matrix[row_ind, col_ind] = 1.0  # x
        design_matrix[row_ind + 1, col_ind + 1] = 1.0  # y

    def design_matrix_dims(self):
        """
        Calculate dimensions of the deployed design matrix.

        Returns:
            tuple: (m, n) where:
                  m = 2 * total_vertices (x,y coordinates for each vertex)
                  n = 2 * seed_parameters (x,y coordinates for each seed point)

        Size calculation:
            - All dimensions doubled to handle (x,y) coordinate pairs
            - Maintains same logical structure as MatrixStructure
            - Enables full 2D coordinate generation

        Shape: (2*total_vertices, 2*seed_parameters)
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        num_seed_points = num_linkage_cols + num_linkage_rows
        num_bulk_points = 2 * num_linkage_rows * num_linkage_cols
        num_boundary_points = 2 * (num_linkage_rows + num_linkage_cols + 2)

        m = 2 * (num_seed_points + num_bulk_points + num_boundary_points)
        n = 2 * (num_linkage_rows + num_linkage_cols + 4)

        return m, n

    def build_structure(
        self, top_points, left_points, corners, offsets, boundary_offsets, phis, boundary_phis=None
    ):
        """
        Build deployed structure geometry from seed points and angles.

        Args:
            top_points (ndarray): Top edge seed points, shape (num_cols, 2)
            left_points (ndarray): Left edge seed points, shape (num_rows, 2)
            corners (ndarray): Corner seed points, shape (4, 2)
            offsets (ndarray): Interior offset parameters, shape (num_rows, num_cols)
            boundary_offsets (list): Boundary offset parameters for each edge
            phis (ndarray or float): Deployment angles
            boundary_phis (ndarray, optional): Boundary deployment angles

        Updates:
            self.points: All vertex coordinates in deployed configuration

        Process:
            1. Calculate deployment-aware design matrix
            2. Flatten seed points into coordinate vector [x1,y1,x2,y2,...]
            3. Matrix multiply to get all deployed coordinates
            4. Reshape result back to vertex array

        Shape transitions:
            - Seed points: (various, 2) -> Flattened coordinate vector
            - Matrix multiplication -> Full coordinate vector
            - Reshape -> (total_vertices, 2)
        """
        design_matrix = self.calculate_design_matrix(offsets, boundary_offsets, phis, boundary_phis)
        seed_points = np.hstack(
            (top_points.flatten("C"), left_points.flatten("C"), corners.flatten("C"))
        )
        points = np.dot(design_matrix, seed_points)
        points = points.reshape(int(len(points) / 2), 2)
        self.points = points

    def linear_inverse_design(
        self, boundary_points, corners, interior_offsets, boundary_offsets, phis, boundary_phis=None
    ):
        """
        Inverse design for deployed configuration: find seed points for target boundary.

        Args:
            boundary_points (ndarray): Target boundary coordinates in deployed state
            corners (ndarray): Fixed corner points, shape (4, 2)
            interior_offsets (ndarray): Interior offset parameters
            boundary_offsets (list): Boundary offset parameters
            phis (ndarray or float): Deployment angles
            boundary_phis (ndarray, optional): Boundary deployment angles

        Updates:
            self.points: Structure achieving target deployed boundary shape

        Process:
            1. Calculate deployed design matrix
            2. Extract boundary equations from matrix
            3. Solve linear system for top/left seed points
            4. Build full deployed structure

        Mathematical approach:
            - Accounts for deployment angles in inverse problem
            - Finds seed points that deploy to target boundary
            - More complex than flat inverse design due to rotations

        Shape transitions:
            - Target deployed boundary -> Computed seed points -> Full deployed structure
        """

        num_linkage_cols = self.num_linkage_cols
        num_linkage_rows = self.num_linkage_rows

        design_matrix = self.calculate_design_matrix(
            interior_offsets, boundary_offsets, phis, boundary_phis
        )

        bound_inds = []
        for bound_ind in range(4):
            bound_nodes = self.get_outer_boundary_node_inds(bound_ind)
            for ind in bound_nodes:
                bound_inds.append(2 * ind)
                bound_inds.append(2 * ind + 1)

        sub_design_matrix = design_matrix[bound_inds, : 2 * (num_linkage_cols + num_linkage_rows)]
        sub_design_matrix_inverse = np.linalg.inv(sub_design_matrix)

        inverted_seed_points = np.dot(sub_design_matrix_inverse, boundary_points.flatten("C"))
        top_points = inverted_seed_points[: 2 * num_linkage_cols].reshape(num_linkage_cols, 2)
        left_points = inverted_seed_points[
            2 * num_linkage_cols : 2 * (num_linkage_cols + num_linkage_rows)
        ].reshape(num_linkage_rows, 2)

        self.build_structure(
            top_points,
            left_points,
            corners,
            interior_offsets,
            boundary_offsets,
            phis,
            boundary_phis,
        )


def main():
    """
    Main function for module reloading during development.

    Returns:
        None

    Purpose:
        - Provides feedback when Structure module is reloaded
        - Useful for interactive development and debugging
        - Indicates successful module import/reload
    """
    print("reloading Structure")
    return
