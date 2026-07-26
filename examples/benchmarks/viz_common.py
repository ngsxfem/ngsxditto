"""Shared helpers for the Hysing–Turek benchmark visualisations.

The plots in the accompanying notebook are produced from **VTK snapshots** that
carry the (P1) level set ``phi``, the high-order isoparametric mesh
``deform``-ation and — where relevant — the mean curvature ``H``. This module

* dumps such snapshots from *live* ``ngsxditto`` objects
  (:func:`dump_levelset`, :func:`dump_curvature`), and
* reads them back and extracts the **curved** geometry the solver integrates on
  (warp the points by ``deform``, then contour ``phi``), so the interface can be
  drawn faithfully with matplotlib.

All rendering is headless (matplotlib ``Agg`` / pyvista ``off_screen``); no
display or webgui is required. The per-figure scripts
``viz_isoparametric_interface.py``, ``viz_redistancing_gallery.py``,
``viz_curvature_on_interface.py`` and ``viz_qoi_convergence.py`` build on it.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri
import pyvista as pv
pv.OFF_SCREEN = True

try:  # only needed for the dump_* helpers (i.e. when called from a live run)
    from ngsolve import VTKOutput, VOL
except Exception:  # pragma: no cover - plotting works without ngsolve
    VTKOutput = None


# --------------------------------------------------------------------------- #
# Dumping snapshots from live ngsxditto objects
# --------------------------------------------------------------------------- #
def dump_levelset(levelset, filename, extra_coefs=None, extra_names=None):
    """Write ``phi`` (P1 level set), ``deform`` (isoparametric deformation) and
    optional extra coefficient functions of a ``LevelSetGeometry`` to ``filename``
    (a ``.vtu`` is produced). ``phi`` is the *continuous* field so its isolines
    can be drawn; pass ``levelset.lsetp1`` via ``extra_coefs`` if you want the
    exact P1 cut instead."""
    coefs = [levelset.field, levelset.deformation] + list(extra_coefs or [])
    names = ["phi", "deform"] + list(extra_names or [])
    VTKOutput(levelset.mesh, coefs=coefs, names=names, filename=filename,
              subdivision=levelset.order).Do(vb=VOL)


def dump_curvature(levelset, curvature_cf, filename):
    """Write ``phi`` (P1 cut), ``deform`` and the mean curvature ``H`` so the
    zero isoline can be coloured by |H| (see ``viz_curvature_on_interface.py``)."""
    VTKOutput(levelset.mesh, coefs=[levelset.lsetp1, levelset.deformation, curvature_cf],
              names=["phi", "deform", "H"], filename=filename,
              subdivision=levelset.order).Do(vb=VOL)


# --------------------------------------------------------------------------- #
# Reading snapshots and extracting the curved geometry
# --------------------------------------------------------------------------- #
def pvd_times(pvd):
    """Map each ``.vtu`` referenced by a ParaView ``.pvd`` collection to its
    physical time. Returns ``{}`` if the file is missing."""
    import os
    import xml.etree.ElementTree as ET
    if not os.path.exists(pvd):
        return {}
    root = ET.parse(pvd).getroot()
    d = os.path.dirname(pvd)
    return {os.path.join(d, ds.attrib["file"]): float(ds.attrib["timestep"])
            for ds in root.iter("DataSet")}


def read_warped(vtu, warp=True):
    """Read a snapshot ``.vtu`` and (by default) warp its points by the
    ``deform`` field, i.e. map the reference mesh to the isoparametric geometry
    the solver actually uses. Returns the pyvista mesh."""
    m = pv.read(vtu)
    if warp and "deform" in m.point_data:
        d = np.asarray(m.point_data["deform"])
        p = m.points.copy()
        p[:, 0] += d[:, 0]
        p[:, 1] += d[:, 1]
        m.points = p
    return m


def _polys_from_contour(c):
    """Split a pyvista line PolyData (after ``.strip``) into ordered (n,2) arrays."""
    c = c.strip(join=True)
    arr, i, polys = c.lines, 0, []
    while i < len(arr):
        n = int(arr[i])
        ids = arr[i + 1:i + 1 + n]
        polys.append(c.points[ids][:, :2])
        i += 1 + n
    return polys


def interface_polylines(vtu, warp=True, level=0.0, scalar="phi"):
    """Ordered polylines (list of (n,2) arrays) of the ``scalar=level`` isoline of
    a snapshot; ``warp`` applies the isoparametric deformation (curved interface),
    ``warp=False`` gives the raw P1 cut."""
    m = read_warped(vtu, warp=warp)
    c = m.contour([level], scalars=scalar)
    return _polys_from_contour(c) if c.n_points else []


def isoline_levels(vtu, levels, warp=True, scalar="phi"):
    """Dict ``{level: [polylines]}`` for several isolines at once — used to show
    the +/- 0.025*i level sets before/after redistancing."""
    m = read_warped(vtu, warp=warp)
    out = {}
    for lv in levels:
        c = m.contour([lv], scalars=scalar)
        out[lv] = _polys_from_contour(c) if c.n_points else []
    return out


def contour_with_scalar(vtu, scalar_on_line="H", warp=True, level=0.0, scalar="phi"):
    """Zero isoline as ``(points (n,2), values (n,))`` with ``scalar_on_line``
    interpolated onto it (e.g. the mean curvature ``H``)."""
    m = read_warped(vtu, warp=warp)
    c = m.contour([level], scalars=scalar)
    if c.n_points == 0:
        return np.empty((0, 2)), np.empty(0)
    return c.points[:, :2], _scalarise(c.point_data[scalar_on_line])


def _scalarise(a):
    """Reduce a possibly multi-component field to a scalar magnitude (e.g. the
    mean-curvature vector ``H`` = kappa*n is stored with 2/3 components)."""
    a = np.asarray(a)
    return np.linalg.norm(a, axis=1) if a.ndim > 1 else a


def contour_segments(vtu, scalar_on_line="H", warp=True, level=0.0, scalar="phi"):
    """Zero isoline as coloured line segments: ``(segments (m,2,2), values (m,))``
    where ``values`` is ``|scalar_on_line|`` averaged over each segment's
    endpoints — ready for a matplotlib ``LineCollection``. A vector field (such as
    the curvature vector ``H``) is reduced to its magnitude."""
    m = read_warped(vtu, warp=warp)
    c = m.contour([level], scalars=scalar)
    if c.n_points == 0:
        return np.empty((0, 2, 2)), np.empty(0)
    lines = c.lines.reshape(-1, 3)[:, 1:]        # (m,2) endpoint indices
    P = c.points[:, :2]
    val = _scalarise(c.point_data[scalar_on_line])
    segs = np.stack([P[lines[:, 0]], P[lines[:, 1]]], axis=1)
    return segs, 0.5 * (val[lines[:, 0]] + val[lines[:, 1]])


# --------------------------------------------------------------------------- #
# Background mesh
# --------------------------------------------------------------------------- #
def triangulation(meshref):
    """A matplotlib ``Triangulation`` of the (undeformed) background mesh, taken
    either from an ``.npz`` with ``verts``/``tris`` arrays or from a live
    ``ngsolve`` mesh."""
    if isinstance(meshref, str):
        d = np.load(meshref)
        verts, tris = d["verts"], d["tris"]
    elif hasattr(meshref, "vertices"):
        from ngsolve import VOL as _VOL
        verts = np.array([list(v.point) for v in meshref.vertices])
        tris = np.array([[w.nr for w in el.vertices] for el in meshref.Elements(_VOL)])
    else:
        verts, tris = meshref
    return mtri.Triangulation(verts[:, 0], verts[:, 1], tris)


def draw_mesh(ax, tri, color="#b9c0cc", lw=0.6):
    ax.triplot(tri, color=color, lw=lw)
