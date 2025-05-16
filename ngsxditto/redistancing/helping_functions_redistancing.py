from ngsolve import *
from netgen.occ import *
import math
import numpy as np
import scipy as sp


def distance(x1, x2):
    """
    Returns the distance between two points.
    """
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** (1 / 2)


def same_sign(a, b):
    """
    Checks if two numbers have the same sign
    """

    if (a >= 0 and b >= 0) or (a <= 0 and b <= 0):
        return True
    else:
        return False


def orth_projection(point, line_points):
    """
    Returns the orthogonal projection of a point on a line. The line should be given by two points
    """

    x1, y1 = point
    x2, y2 = line_points[0]
    x3, y3 = line_points[1]
    dx = x3 - x2
    dy = y3 - y2
    dot_product = ((x1 - x2) * dx + (y1 - y2) * dy) / (dx ** 2 + dy ** 2)
    projection_x = x2 + dot_product * dx
    projection_y = y2 + dot_product * dy

    return (projection_x, projection_y)


def point_in_triangle(point, triangle):
    """
    Checks if a point is in a triangle. The triangle should be given by the three vertices.
    """

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, triangle[0], triangle[1]) < 0.0
    b2 = sign(point, triangle[1], triangle[2]) < 0.0
    b3 = sign(point, triangle[2], triangle[0]) < 0.0
    return (b1 == b2) and (b2 == b3)


def get_edge_midpoint(mesh, edge):
    """
    Returns the coordinates of the midpoint of an edge.

    Parameters
    ----------
        mesh: ngsolve.comp.Mesh
            The mesh the edge is defined on
        edge: ngsolve.comp.MeshNode
            The edge the midpoint should be calculated of

    Returns
    -------
        tuple:
            midpoint of the edge
    """

    vs = mesh[edge].vertices
    coord = [mesh[v].point for v in vs]
    midx = (1 / 2) * (coord[0][0] + coord[1][0])
    midy = (1 / 2) * (coord[0][1] + coord[1][1])
    return (midx, midy)


def find_levelset_elements(V, mesh, phi):
    """
    Loops through the elements of the Finite Element Space to find the elements that the zero levelset crosses.

    Parameters
    ----------
        V: ngsolve.comp.FESpace
            The Finite Element Space of the functions
        mesh: ngsolve.comp.Mesh
            The mesh on which the finite element spaces are defined
        phi: ngsolve.comp.GridFunction
            The levelset function

    Returns
    -------
        list:
            List of all elements (ngsolve.comp.FESpaceElement) that the zero levelset of the levelset function crosses
    """

    levelset_elements = []
    for el in V.Elements():
        vals = []
        vs = el.vertices
        for v in vs:
            x, y = mesh[v].point
            vals.append(phi(mesh(x, y)))
        if V.globalorder == 2:
            edges = el.edges
            for edge in edges:
                x, y = get_edge_midpoint(mesh, edge)
                vals.append(phi(mesh(x, y)))
        if max(vals) > 0 and min(vals) < 0:
            levelset_elements.append(el)
    return levelset_elements


def vertices_of_element_set(elements):
    """
    Finds all vertices that are part of at least one element of a set of elements.

    Parameters
    ----------
        elements: list
            List of elements (ngsolve.comp.FESpaceElement)

    Returns
    -------
        list:
            List of all vertices (ngsolve.comp.NodeId) that are part of at least one given element
    """

    vertices = set()
    for el in elements:
        vertices.update(set(el.vertices))
    return list(vertices)


def edges_of_element_set(elements):
    """
    Finds all edges that are part of at least one element of a set of elements.

    Parameters
    ----------
        elements: list
            List of elements (ngsolve.comp.FESpaceElement)

    Returns
    -------
        list:
            List of all edges (ngsolve.comp.NodeId) that are part of at least one given element
    """

    edges = set()
    for el in elements:
        edges.update(set(el.edges))
    return list(edges)


def get_all_dofs(V):
    """
    Returns all degrees of freedom of a Finite Element Space

    Parameters
    ----------
        V: ngsolve.comp.FESpace
            The Finite Element Space

    Returns
    -------
        list:
            List of dofs (int) of the FES
    """

    dofs = set()
    for el in V.Elements():
        dofs.update(set(el.dofs))
    return list(dofs)


def find_zero_points(V, mesh, element, phi):
    """
    Finds the points on the boundary of an element where the levelset function is zero. Exact only for Polynomial degree one.

    Parameters
    ----------
        V: ngsolve.comp.FESpace
            The Finite Element Space of the function
        mesh: ngsolve.comp.Mesh
            The mesh on which the element lives on
        element: ngsolve.comp.FESpaceElement
            The element we want to find the zero points on
        phi: ngsolve.comp.GridFunction
            The levelset function

    Returns
    -------
        list:
            List of the points as tuples of floats
    """

    vs = element.vertices
    coord = [mesh[v].point for v in vs]

    zero_points = []
    for j in range(3):
        for i in range(j):
            x_i, y_i = coord[i]
            x_j, y_j = coord[j]

            val_i = phi(mesh(x_i, y_i))
            val_j = phi(mesh(x_j, y_j))

            if V.globalorder == 1:
                if not same_sign(val_i, val_j):
                    t = abs(val_i) / abs(val_i - val_j)
                    zero_points.append(((1 - t) * x_i + t * x_j, (1 - t) * y_i + t * y_j))

            elif V.globalorder == 2:
                midx = (1 / 2) * (x_i + x_j)
                midy = (1 / 2) * (y_i + y_j)
                mid_val = phi(mesh(midx, midy))

                if not same_sign(val_i, mid_val):
                    t = abs(val_i) / abs(val_i - mid_val)
                    zero_points.append(((1 - t) * x_i + t * midx, (1 - t) * y_i + t * midy))
                if not same_sign(phi(x_j, y_j), mid_val):
                    t = abs(val_j) / abs(val_j - mid_val)
                    zero_points.append(((1 - t) * x_j + t * midx, (1 - t) * y_j + t * midy))

    return zero_points


def get_opposite_vertex(mesh, vertex1, edge):
    """
    Given a vertex, returns the vertex that is on the other side of the given edge.

    Parameters
    ----------
        mesh: ngsolve.comp.Mesh
            The mesh on which the vertices and edge are defined on
        vertex1: ngsolve.comp.NodeId
            The known vertex
        edge: ngsolve.comp.NodeId
            The connecting edge
    Returns
    -------
        ngsolve.comp.NodeId
            The vertex on the other end of the edge
    """

    edge_vertices = mesh.edges[edge.nr].vertices
    if vertex1 not in edge_vertices:
        raise Exception("vertex1 not part of the edge")
    if edge_vertices[0] == vertex1:
        return edge_vertices[1]
    else:
        return edge_vertices[0]


def get_length_of_edge(mesh, edge):
    """
    Returns the coordinates of the midpoint of an edge.

    Parameters
    ----------
        mesh: ngsolve.comp.Mesh
            The mesh the edge is defined on
        edge: ngsolve.comp.MeshNode
            The edge the distance should be calculated of

    Returns
    -------
        float:
            distance of the edge
    """

    edge_vertices = mesh.edges[edge.nr].vertices
    coord = [mesh[v].point for v in edge_vertices]

    return distance(coord[0], coord[1])


def get_fes_matrix(V, mesh):
    """
    Get the matrix that multiplied by the vector of the basis coefficients equals the values of the function at the vertices and edges.

    Parameters
    ----------
        V: ngsolve.comp.FESpace
            The Finite Element Space that defines the degrees of freedom
        mesh: ngsolve.comp.Mesh
            The mesh the vertices and edges are defined on

    Returns
    -------
        scipy.sparse.csr.csr_matrix
            The matrix that converts basis coefficients to values of a function
    """

    n = V.ndof

    if V.globalorder == 1:
        return sp.sparse.csr_matrix(np.eye(n))

    elif V.globalorder == 2:
        matrix = np.zeros((n, n))

        for i in range(n):
            if i < len(mesh.vertices):
                matrix[i][i] = 1

        for edge in mesh.edges:
            v1, v2 = mesh[edge].vertices
            matrix[V.GetDofNrs(edge)[0]][V.GetDofNrs(edge)[0]] = -1 / 8
            matrix[V.GetDofNrs(edge)[0]][V.GetDofNrs(v1)[0]] = 1 / 2
            matrix[V.GetDofNrs(edge)[0]][V.GetDofNrs(v2)[0]] = 1 / 2
        return sp.sparse.csr_matrix(matrix)

    else:
        raise NotImplementedError("Only implemented for polynomial degrees 1 and 2")


def get_signed_distance_vector(V, mesh, phi, distance_dict):
    """
    Get the vector that takes a dictionary of distances and converts it to a vector of the signed distances

    Parameters
    ----------
        V: ngsolve.comp.FESpace
            The Finite Element Space that defines the degrees of freedom
        mesh: ngsolve.comp.Mesh
            The mesh the vertices and edges are defined on

    Returns
    -------
        numpy.ndarray
            A vector of all signed distances to the levelset
    """

    n = len(distance_dict.items())
    vector = np.empty(n)

    for v in mesh.vertices:
        x, y = mesh[v].point
        if phi(mesh(x, y)) > 0:
            vector[V.GetDofNrs(v)[0]] = distance_dict[V.GetDofNrs(v)[0]]
        else:
            vector[V.GetDofNrs(v)[0]] = -distance_dict[V.GetDofNrs(v)[0]]

    if V.globalorder == 2:
        for edge in mesh.edges:
            x, y = get_edge_midpoint(mesh, edge)
            if phi(mesh(x, y)) > 0:
                vector[V.GetDofNrs(edge)[0]] = distance_dict[V.GetDofNrs(edge)[0]]
            else:
                vector[V.GetDofNrs(edge)[0]] = -distance_dict[V.GetDofNrs(edge)[0]]
    return vector


def l2_to_h1(phi):
    """
    Convert an L2 function to an H1 function.

    This function takes a L2 function `phi` and constructs an H1 finite element space `V_hat` with the same order as `V`.
    It then projects `phi` onto `V_hat` and returns the projected function.

    Parameters
    ----------
    phi: ngsolve.comp.GridFunction
        The function in the L2 space `V` to be converted to the H1 space.

    Returns
    -------
    phi_hat: ngsolve.comp.GridFunction
        The function `phi` projected onto the H1 space `V_hat`.
    """
    V_hat = H1(phi.space.mesh, order=phi.space.globalorder)
    phi_hat = GridFunction(V_hat)
    phi_hat.Set(phi)
    return phi_hat


def h1_to_l2(phi_hat):
    """
    Convert an H1 function to an L2 function on a given mesh.

    This function takes a H1 function `phi_hat` and constructs an L2 finite element space `V` with the same order as `V_hat`.
    It then projects `phi_hat` onto `V` and returns both the new L2 space and the projected function.

    Parameters
    ----------
    phi_hat: ngsolve.comp.GridFunction
        The function in the H1 space `V_hat` to be converted to the L2 space

    Returns
    -------
    phi: ngsolve.comp.GridFunction
        The function `phi_hat` projected onto the L2 space `V`
    """
    V = L2(phi_hat.space.mesh, order=phi_hat.space.globalorder, dgjumps=True)
    phi = GridFunction(V)
    phi.Set(phi_hat)
    return phi

