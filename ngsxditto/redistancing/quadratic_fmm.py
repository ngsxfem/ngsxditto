from ngsolve import *
from .redistancing import *
from .helping_functions_redistancing import *

class QuadraticFastMarching(BaseRedistancing):
    """
    The Quadratic Fast Marching algorithm.
    """
    def __init__(self, bandwidth: float=None):
        super().__init__(bandwidth)
        self.order = 2

    def Redistance(self, phi: GridFunction):
        """
        Redistance based on the quadratic fast marching algorithm.
        """
        phi_copy = GridFunction(phi.space)
        phi_copy.Set(phi)
        l2_function = False
        if type(phi_copy.space).__name__ == "L2":
            l2_function = True
            phi_copy = l2_to_h1(phi_copy)
        V = phi_copy.space

        # Find elements that the zero levelset crosses
        levelset_elements = find_levelset_elements(phi_copy)
        levelset_vertices = vertices_of_element_set(levelset_elements)
        levelset_edges = edges_of_element_set(levelset_elements)

        all_dofs = get_all_dofs(V)
        distance_dict = {dof: [float('inf')] for dof in all_dofs}
        nearest_point_dict = {dof: [] for dof in levelset_vertices + levelset_edges}

        # calculate the minimum distance from vertex to the zero levelset within the element
        for el in levelset_elements:
            v_coord = [V.mesh[v].point for v in el.vertices]
            edge_coord = {get_edge_midpoint(V.mesh, edge): edge for edge in el.edges}
            all_coord = v_coord + list(edge_coord.keys())
            zero_points = find_zero_points(phi_copy, el)

            for point in all_coord:
                distance_to_zeropoint1 = distance(point, zero_points[0])
                distance_to_zeropoint2 = distance(point, zero_points[1])

                projection = orth_projection(point, zero_points)

                if point_in_triangle(projection, v_coord):
                    distance_to_projection = distance(point, projection)
                    possible_nearest_points = [projection, zero_points[0], zero_points[1]]
                    point_distances_in_element = [distance_to_projection, distance_to_zeropoint1,
                                                  distance_to_zeropoint2]

                    min_distance_to_levelset = min(distance_to_projection, distance_to_zeropoint1,
                                                   distance_to_zeropoint2)
                else:
                    possible_nearest_points = zero_points
                    point_distances_in_element = [distance_to_zeropoint1, distance_to_zeropoint2]

                    min_distance_to_levelset = min(distance_to_zeropoint1, distance_to_zeropoint2)

                if point in v_coord:
                    nearest_point_dict[el.vertices[v_coord.index(point)]].append(
                        possible_nearest_points[point_distances_in_element.index(min_distance_to_levelset)])
                    distance_dict[V.GetDofNrs(el.vertices[v_coord.index(point)])[0]].append(min_distance_to_levelset)

                else:
                    nearest_point_dict[edge_coord[point]].append(
                        possible_nearest_points[point_distances_in_element.index(min_distance_to_levelset)])
                    distance_dict[V.GetDofNrs(edge_coord[point])[0]].append(min_distance_to_levelset)

        # calculate the minimum distance from vertex to the zero levelset globally
        min_distance_dict = {dof: min(distance_dict[dof]) for dof in distance_dict.keys()}

        # calculate distance for dofs further away, use dijkstra
        previous_points_dict = {dof: nearest_point_dict[dof][
            distance_dict[V.GetDofNrs(dof)[0]].index(min_distance_dict[V.GetDofNrs(dof)[0]]) - 1] for dof in
                                levelset_vertices + levelset_edges}
        nearest_levelset_point_dict = previous_points_dict

        marked_dofs = levelset_vertices
        finished_dofs = levelset_edges
        min_distance = 0

        def sc1():
            return min_distance > self.bandwidth/2

        def sc2():
            return marked_dofs == []

        if self.bandwidth is not None:
            stopping_criterion = sc1
        else:
            stopping_criterion = sc2

        while not stopping_criterion():
            next_marked_dofs = []
            min_distance = min([min_distance_dict[V.GetDofNrs(v)[0]] for v in marked_dofs])
            for vertex in marked_dofs:
                if min_distance_dict[V.GetDofNrs(vertex)[0]] == min_distance:
                    dof = vertex
                    break
            if dof.type == levelset_vertices[0].type:
                edges = V.mesh[dof].edges
                for edge in edges:
                    edge_point = get_edge_midpoint(V.mesh, edge)
                    if edge not in marked_dofs and edge not in finished_dofs:
                        if edge not in next_marked_dofs:
                            next_marked_dofs.append(edge)

                        new_distance = distance(nearest_levelset_point_dict[dof], get_edge_midpoint(V.mesh, edge))
                        if new_distance < min_distance_dict[V.GetDofNrs(edge)[0]]:
                            previous_points_dict[edge] = V.mesh[dof].point
                            nearest_levelset_point_dict[edge] = nearest_levelset_point_dict[dof]
                            min_distance_dict[V.GetDofNrs(edge)[0]] = distance(nearest_levelset_point_dict[edge],
                                                                               get_edge_midpoint(V.mesh, edge))

            if dof.type == levelset_edges[0].type:
                vs = V.mesh[dof].vertices
                for v in vs:
                    if v not in marked_dofs and v not in finished_dofs:
                        if v not in next_marked_dofs:
                            next_marked_dofs.append(v)
                        new_distance = distance(nearest_levelset_point_dict[dof], V.mesh[v].point)
                        if new_distance < min_distance_dict[V.GetDofNrs(v)[0]]:
                            previous_points_dict[v] = get_edge_midpoint(V.mesh, dof)
                            nearest_levelset_point_dict[v] = nearest_levelset_point_dict[dof]
                            min_distance_dict[V.GetDofNrs(v)[0]] = distance(nearest_levelset_point_dict[v],
                                                                            V.mesh[v].point)

            finished_dofs.append(dof)
            marked_dofs.remove(dof)
            marked_dofs = marked_dofs + next_marked_dofs

        # solve linear system to get basis coefficients
        matrix = get_fes_matrix(V)

        if self.bandwidth is not None:
            old_distances = np.array(matrix @ phi_copy.vec.data)
            min_distance_dict = {dof: dist if not math.isinf(dist) else abs(old_distances[dof]) for dof, dist in min_distance_dict.items()}
        signed_distances = get_signed_distance_vector(phi_copy, min_distance_dict)

        phi_copy.vec.data = sp.sparse.linalg.spsolve(matrix, signed_distances)
        if l2_function:
            phi_copy = h1_to_l2(phi_copy)

        phi.vec.data = phi_copy.vec.data
