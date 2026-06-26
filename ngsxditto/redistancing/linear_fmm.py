import heapq
from ngsolve import *
from .redistancing import *
from .helping_functions_redistancing import *

class LinearFastMarching(BaseRedistancing):
    """
    The Linear Fast Marching algorithm.
    """
    def __init__(self, bandwidth: float=None):
        super().__init__(bandwidth)
        self.order = 1

    def Redistance(self, phi: GridFunction):
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

        all_dofs = get_all_dofs(V)
        distance_dict = {dof: [float('inf')] for dof in all_dofs}
        nearest_point_dict = {vertex: [] for vertex in levelset_vertices}

        # calculate the minimum distance from vertex to the zero levelset within the element
        for el in levelset_elements:
            coord = [V.mesh[v].point for v in el.vertices]
            zero_points = find_zero_points(phi_copy, el)

            for point in coord:
                distance_to_zeropoint1 = distance(point, zero_points[0])
                distance_to_zeropoint2 = distance(point, zero_points[1])
                projection = orth_projection(point, zero_points)

                if point_in_triangle(projection, coord):
                    distance_to_projection = distance(point, projection)
                    possible_nearest_points = [projection, zero_points[0], zero_points[1]]
                    point_distances_in_element = [distance_to_projection, distance_to_zeropoint1,
                                                  distance_to_zeropoint2]

                    min_distance_to_levelset = min(point_distances_in_element)

                else:
                    possible_nearest_points = zero_points
                    point_distances_in_element = [distance_to_zeropoint1, distance_to_zeropoint2]

                    min_distance_to_levelset = min(point_distances_in_element)

                nearest_point_dict[el.vertices[coord.index(point)]].append(
                    possible_nearest_points[point_distances_in_element.index(min_distance_to_levelset)])
                distance_dict[V.GetDofNrs(el.vertices[coord.index(point)])[0]].append(min_distance_to_levelset)

        # calculate the minimum distance from vertex to the zero levelset globally
        min_distance_dict = {dof: min(distance_dict[dof]) for dof in distance_dict.keys()}

        # calculate distance for dofs further away, use dijkstra
        previous_points_dict = {vertex: nearest_point_dict[vertex][
            distance_dict[V.GetDofNrs(vertex)[0]].index(min_distance_dict[V.GetDofNrs(vertex)[0]]) - 1] for vertex
                                in levelset_vertices}
        nearest_levelset_point_dict = previous_points_dict

        # Dijkstra-style fast marching from the interface outward, heap-based -> O(N log N).
        # (The previous version did a linear min-search over the whole frontier *inside* the
        # loop plus list membership/removal -> O(N^2), which made redistancing the dominant
        # cost on fine meshes: ~1000 s/call at ne~1e4.) Each vertex enters the frontier exactly
        # once and its tentative distance is never lowered afterwards, so the heap keys are
        # stable (no decrease-key needed); the empty-frontier and bandwidth stops fall out
        # naturally. `cnt` is just a tiebreaker so heapq never compares the vertex objects.
        finished = set()
        in_frontier = set()
        heap = []
        cnt = 0
        for v in levelset_vertices:
            heapq.heappush(heap, (min_distance_dict[V.GetDofNrs(v)[0]], cnt, v))
            in_frontier.add(v); cnt += 1

        while heap:
            d, _, v = heapq.heappop(heap)
            if v in finished:
                continue
            if self.bandwidth is not None and d > self.bandwidth / 2:
                break
            finished.add(v)
            in_frontier.discard(v)
            for edge in V.mesh[v].edges:
                opposite_vertex = get_opposite_vertex(V.mesh, v, edge)
                if opposite_vertex in finished or opposite_vertex in in_frontier:
                    continue
                opp_dof = V.GetDofNrs(opposite_vertex)[0]
                new_distance = distance(nearest_levelset_point_dict[v], V.mesh[opposite_vertex].point)
                if new_distance < min_distance_dict[opp_dof]:
                    previous_points_dict[opposite_vertex] = V.mesh[v].point
                    nearest_levelset_point_dict[opposite_vertex] = nearest_levelset_point_dict[v]
                    min_distance_dict[opp_dof] = new_distance
                in_frontier.add(opposite_vertex)
                heapq.heappush(heap, (min_distance_dict[opp_dof], cnt, opposite_vertex))
                cnt += 1



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