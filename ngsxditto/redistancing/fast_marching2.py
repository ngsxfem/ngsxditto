from ngsolve import *
from .redistancing import *
from xfem import *
import numpy as np
import scipy.sparse as sp
from .helping_functions_redistancing import *
import ngsolve.webgui as ngw


class FastMarching2(BaseRedistancing):
    """
    Fast Marching redistancing using xfem functionality for efficiency.

    This implementation uses CutInfo from xfem to identify interface elements,
    making it significantly more efficient than the manual element-by-element search
    in LinearFastMarching.
    """

    def __init__(self, bandwidth: float = None):
        super().__init__(bandwidth)
        self.order = 1

    def SetOrder(self, order: int):
        self.order = order

    def Redistance(self, phip1: GridFunction, deformation=None):
        """
        Applies redistancing using Fast Marching method with xfem integration.

        Parameters:
        -----------
        phip1 : GridFunction
            The level set function to be redistanced
        """
        phip1_copy = GridFunction(phip1.space)
        phip1_copy.Set(phip1)
        l2_function = False
        if type(phip1_copy.space).__name__ == "L2":
            l2_function = True
            phip1_copy = l2_to_h1(phip1_copy)

        V = phip1_copy.space
        mesh = V.mesh

        # Use xfem's CutInfo to identify interface elements efficiently
        ci = CutInfo(mesh)
        ci.Update(phip1_copy)
        if_bitarray = ci.GetElementsOfType(IF)
        if_els = [el for el in V.Elements() if if_bitarray[el.nr]]

        # Get vertices on the interface
        levelset_vertices = vertices_of_element_set(if_els)

        # Build a mapping from dof to vertex for O(1) lookup
        dof_to_vertex = {}
        vertex_to_dof = {}
        for vertex in mesh.vertices:
            dof = V.GetDofNrs(vertex)[0]
            dof_to_vertex[dof] = vertex
            vertex_to_dof[vertex] = dof

        # Initialize distance dictionaries
        all_dofs = [dof for el in V.Elements() for dof in el.dofs]
        all_dofs = list(set(all_dofs))

        distance_dict = {dof: float('inf') for dof in all_dofs}
        nearest_point_dict = {vertex: None for vertex in levelset_vertices}

        # Step 1: Calculate distances from vertices on interface elements
        for el in if_els:
            zero_points = find_zero_points(phip1_copy, el)
            if len(zero_points) < 2:
                continue

            for vertex in el.vertices:
                point = mesh[vertex].point

                # Find minimum distance to zero points
                distances = [distance(point, zp) for zp in zero_points]
                min_dist = min(distances)
                nearest_point = zero_points[distances.index(min_dist)]

                dof = vertex_to_dof[vertex]
                if min_dist < distance_dict[dof]:
                    distance_dict[dof] = min_dist
                    nearest_point_dict[vertex] = nearest_point

        # Step 2: Fast Marching propagation using priority queue
        import heapq

        marked_dofs = set(vertex_to_dof[v] for v in levelset_vertices)
        finished_dofs = set()
        pq = [(distance_dict[dof], dof) for dof in marked_dofs if distance_dict[dof] != float('inf')]
        heapq.heapify(pq)

        nearest_levelset_point = {
            vertex: nearest_point_dict[vertex]
            for vertex in levelset_vertices
            if nearest_point_dict[vertex] is not None
        }

        while pq and (self.bandwidth is None or min(d for d, _ in pq) <= self.bandwidth / 2):
            min_dist, current_dof = heapq.heappop(pq)

            if current_dof in finished_dofs:
                continue

            # O(1) lookup instead of loop
            current_vertex = dof_to_vertex[current_dof]
            finished_dofs.add(current_dof)

            # Process neighbors
            for edge in mesh[current_vertex].edges:
                # Get opposite vertex
                edge_verts = mesh[edge].vertices
                next_vertex = edge_verts[1] if edge_verts[0] == current_vertex else edge_verts[0]
                next_dof = vertex_to_dof[next_vertex]

                if next_dof not in finished_dofs:
                    # Update distance via current vertex
                    if current_vertex in nearest_levelset_point:
                        new_dist = distance(
                            nearest_levelset_point[current_vertex],
                            mesh[next_vertex].point
                        )

                        if new_dist < distance_dict[next_dof]:
                            distance_dict[next_dof] = new_dist
                            nearest_levelset_point[next_vertex] = nearest_levelset_point[current_vertex]
                            heapq.heappush(pq, (new_dist, next_dof))

        # Step 3: Convert distances to signed distances
        signed_distances = np.zeros(V.ndof)

        for vertex in mesh.vertices:
            dof = vertex_to_dof[vertex]
            x, y = mesh[vertex].point

            # Evaluate phi at vertex to determine sign
            sign = 1.0 if phip1_copy(mesh(x, y)) > 0 else -1.0
            signed_distances[dof] = sign * distance_dict[dof]

        # Replace infinite distances with original values
        for dof in range(V.ndof):
            if np.isinf(distance_dict.get(dof, float('inf'))):
                signed_distances[dof] = phip1_copy.vec.data[dof]


        phip1_copy.vec.data = signed_distances
        if l2_function:
            phip1_copy = h1_to_l2(phip1_copy)

        phip1.vec.data = phip1_copy.vec.data