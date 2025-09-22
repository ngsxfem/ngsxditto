from matplotlib import pyplot as plt
import ngsolve.webgui as ngw
from ngsolve import *


class Visualization:
    def __init__(self, name=None, step_frequency=None, time_frequency=None):
        self.initialize_funcs = []
        self.adddata_funcs = []
        self.draw_funcs = []
        self.name = name
        self.step_frequency = step_frequency
        self.time_frequency = time_frequency

    def RegisterInitialize(self, func, *args):
        self.initialize_funcs.append((func, args))

    def RegisterAddData(self, func, *args):
        self.adddata_funcs.append((func, args))

    def RegisterDraw(self, func, *args):
        self.draw_funcs.append((func, args))

    def Initialize(self):
        for func, args in self.initialize_funcs:
            func(*args)

    def AddData(self):
        for func, args in self.adddata_funcs:
            func(*args)

    def Draw(self):
        for func, args in self.draw_funcs:
            func(*args)



class SphericityDiagram(Visualization):
    def __init__(self, lset, time, name=None, step_frequency=None, time_frequency=None):
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.time = time
        self.time_list = []
        self.surface_volume_ratio = []

    def Initialize(self):
        self.time_list = [self.time.Get()]
        self.surface_volume_ratio = [self.lset.surface_area / self.lset.volume]

    def AddData(self):
        self.time_list.append(self.time.Get())
        self.surface_volume_ratio.append(self.lset.surface_area/self.lset.volume)

    def Draw(self):
        plt.plot(self.time_list, self.surface_volume_ratio)
        plt.show()


class UnfittedNGSWebguiPlot(Visualization):
    def __init__(self, lset, cf_neg, cf_pos, order, time, end_time, name=None, step_frequency=None, time_frequency=None,
                 min=0, max=1, autoscale=True):
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.cf_neg = cf_neg
        self.cf_pos = cf_pos
        self.order = order
        self.time = time
        self.gf_vis = None
        self.gf_vis_tmp = None
        self.vis_last_time = None
        self.vis_time_increment = None
        self.end_time = end_time
        self.min = min
        self.max = max
        self.autoscale = autoscale

    def Initialize(self):
        self.gf_vis = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=4), multidim=0)
        self.gf_vis_tmp = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=4))
        self.vis_last_time = self.time.Get()
        self.vis_time_increment = (self.end_time - self.vis_last_time) / 16

    def AddData(self):
        if self.time.Get() >= self.vis_last_time + self.vis_time_increment:
            self.vis_last_time = self.time.Get()
            self.gf_vis_tmp.Set(
                CF((self.lset.field, self.cf_neg, self.cf_pos, 0)))
            self.gf_vis.AddMultiDimComponent(self.gf_vis_tmp.vec)

    def Draw(self):
        ngw.Draw(self.gf_vis, self.lset.mesh, "uhnorm", eval_function="value.x>0.0?value.z:value.y",
                 autoscale=self.autoscale, min=self.min, max=self.max, interpolate_multidim=True, animate=True)


### pyvista stuff

#try:
import logging
import tempfile
import pyvista as pv
from ngsolve import VTKOutput  # Assuming NGSolve is installed
from typing import Sequence, Union, Optional

class PyVistaVisualizer:
    """
    Convenience class for NGSolve -> PyVista visualization
    for clipping and visualization of coefficient functions
    -- prototypical class, not yet fully implemented
    """

    def __init__(
        self,
        mesh: Mesh,
        coefs: Union[CoefficientFunction, Sequence[CoefficientFunction]],
        coef_names: Union[str, Sequence[str]],
        subdivision: int = 5,
        export_on_enter: bool = True
    ) -> None:
        """
        Parameters
        ----------
        mesh : Mesh
            The NGSolve mesh to export.
        coefs : CoefficientFunction or Sequence[CoefficientFunction]
            Single CoefficientFunction or a list of CoefficientFunctions.
        coef_names : str or Sequence[str]
            Single name (if coefs is a single CoefficientFunction) or list of names.
        subdivision : int, optional
            Subdivision level for the primary VTK export (default: 5).
        export_on_enter : bool, optional
            Whether to export the data on entering the context manager (default: True).

        Raises
        ------
        ValueError
            If the number of names does not match the number of coefficients.
        """
        # Normalize to lists
        if isinstance(coefs, CoefficientFunction) and isinstance(coef_names, str):
            self.coefs = [coefs]
            self.coef_names = [coef_names]
        elif isinstance(coefs, Sequence) and isinstance(coef_names, Sequence):
            if len(coefs) != len(coef_names):
                raise ValueError(
                    f"Number of coef_names ({len(coef_names)}) "
                    f"does not match number of coefficients ({len(coefs)})."
                )
            self.coefs = list(coefs)
            self.coef_names = list(coef_names)
        else:
            raise ValueError(
                "Provide either a single CoefficientFunction with a single name, "
                "or sequences of both."
            )

        self.mesh: Mesh = mesh
        self.subdivision: int = subdivision

        # Temporary directory for VTK files
        self._tempdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self._vtk_file: str = f"{self._tempdir.name}/data.vtu"
        self._mesh_file: str = f"{self._tempdir.name}/mesh.vtu"

        self.export_on_enter = export_on_enter

    def export(self) -> None:
        """
        Export mesh and coefficient data to temporary VTK files.
        """
        vtk = VTKOutput(
            ma=self.mesh,
            coefs=self.coefs,
            names=self.coef_names,
            filename=self._vtk_file[:-4],
            subdivision=self.subdivision,
        )
        vtk.Do()

        vtk_mesh = VTKOutput(
            ma=self.mesh,
            filename=self._mesh_file[:-4],
            subdivision=0,
        )
        vtk_mesh.Do()

    def visualize(
        self,
        clip_name: Optional[str] = None,
        scalar_name: str = "u",
        clip_value: float = 0.0,
        cmap: str = "jet",
        wireframe_color: str = "black",
        show_edges: bool = False,
        background: str = "white",
        screenshot: Optional[str] = None,
    ) -> None:
        """
        Visualize the exported data. Supports screenshots and animations.

        Parameters
        ----------
        clip_name : str, optional
            Name of the coefficient to clip (must be one of `coef_names`).
        scaclar_name : str
            Name of the coefficient to visualize (must be one of `coef_names`).
        clip_value : float
            Clip the scalar field at this value (default: 0.0).
        cmap : str, optional
            Colormap for the scalar visualization (default: "jet").
        wireframe_color : str, optional
            Color of the wireframe mesh (default: "black").
        background : str, optional
            Background color for the plot (default: "white").
        screenshot : str, optional
            Path to save a screenshot instead of interactive display.

        Raises
        ------
        ValueError
            If clip_name is not found in coef_names.
        """

        if clip_name is not None and clip_name not in self.coef_names:
            raise ValueError(
                f"Scalar '{clip_name}' not found. Available: {self.coef_names}"
            )

        visobj = pv.read(self._vtk_file)
        visobj_mesh = pv.read(self._mesh_file)

        if clip_value is not None:
            visobj = visobj.clip_scalar(scalars=clip_name, value=clip_value)

        plot = pv.Plotter()
        plot.background_color = background
        plot.add_mesh(visobj_mesh, style="wireframe", color=wireframe_color)
        plot.add_mesh(visobj, scalars=clip_name, cmap=cmap)

        if screenshot:
            plot.show(screenshot=screenshot)
        else:
            plot.show()

    def cleanup(self) -> None:
        """Remove temporary files."""
        self._tempdir.cleanup()

    def __enter__(self) -> "PyVistaVisualizer":
        """Enable use as a context manager."""
        if self.export_on_enter:
            self.export()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Clean up temporary files on exit."""
        self.cleanup()
#except:
#    logger = logging.getLogger(__name__)
#    logger.warning("pyvista plotting not successful. No pyvista installed?")
