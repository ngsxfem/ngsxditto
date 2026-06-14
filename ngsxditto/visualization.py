from matplotlib import pyplot as plt
import ngsolve.webgui as ngw
from ngsolve import *
from ngsxditto import LevelSetGeometry
from ngsxditto.stepper import *
import numpy as np
from typing import Sequence, Union, Optional
from IPython.display import Image, display, HTML, IFrame
import tempfile
import logging

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
except ImportError:
    pv = None
    logger.warning("pyvista not available. PyVista-based visualizations "
                   "(PyVistaAnimation, PyVistaVisualizer) are replaced by no-op dummies; "
                   "for some visualizations it may be better to have pyvista installed.")


class Visualization(StatelessStepper):
    def __init__(self, name=None, step_frequency=None, time_frequency=None):
        super().__init__()
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

    def BeforeLoop(self):
        for func, args in self.initialize_funcs:
            func(*args)

    def ValidateStep(self):
        for func, args in self.adddata_funcs:
            func(*args)

    def AfterLoop(self):
        for func, args in self.draw_funcs:
            func(*args)

    def Step(self):
        pass



class SphericityDiagram(Visualization):
    """
    Plots the ratio of the surface area to the volume.
    """
    def __init__(self, lset, time, name=None, step_frequency=None, time_frequency=None):
        """
        Parameters:
        -----------
        lset: LevelSetGeometry
            The levelset that separates the domains
        time: Parameter
            The time parameter.
        name: str
            Name of the plot.
        step_frequency: int
            Only visualize every n-th step
        time_frequency: int
            Only visualize every time this amount of time has passed
        """
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.time = time
        self.time_list = []
        self.surface_volume_ratio = []

    def BeforeLoop(self):
        self.time_list.append(self.time.Get())
        self.surface_volume_ratio.append(self.lset.surface_area/self.lset.volume)

    def ValidateStep(self):
        self.time_list.append(self.time.Get())
        self.surface_volume_ratio.append(self.lset.surface_area/self.lset.volume)

    def AfterLoop(self):
        plt.plot(self.time_list, self.surface_volume_ratio, ".")
        plt.show()


class UnfittedNGSWebguiPlot(Visualization):
    """
    Animate the unfitted problem using the ngsolve webgui after all steps are finished.
    """
    def __init__(self, lset:LevelSetGeometry, cf_neg:CoefficientFunction, cf_pos:CoefficientFunction,
                 order:int, time:Parameter, end_time:float, name:str=None,
                 step_frequency:int=None, time_frequency:float=None,
                 min:float=0, max:float=1, autoscale:bool=True):
        """
        Parameters:
        -----------
        lset: LevelSetGeometry
            The levelset that separates the domains
        cf_neg: CoefficientFunction
            The function within the negative part of the levelset function.
        cf_pos: CoefficientFunction
            The function within the positive part of the levelset function.
        order: int
            The polynomial order.
        time: Parameter
            The time parameter.
        end_time: float
            The point in time until the animation runs.
        name: str
            Name of the plot.
        step_frequency: int
            Only visualize every n-th step
        time_frequency: int
            Only visualize every time this amount of time has passed
        min: float
            Minimimal value of the color map
        max: float
            Maximimimal value of the color map
        autoscale: bool
            Whether to autoscale the color map
        """
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

    def SetCFNeg(self, cf_neg):
        self.cf_neg = cf_neg

    def SetCFPos(self, cf_pos):
        self.cf_pos = cf_pos

    def BeforeLoop(self):
        self.gf_vis = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=3), multidim=0)
        self.gf_vis_tmp = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=3))
        self.vis_last_time = self.time.Get()
        self.gf_vis_tmp.Set(
            CF((self.lset.lsetp1, self.cf_neg, self.cf_pos)))
        self.gf_vis.AddMultiDimComponent(self.gf_vis_tmp.vec)

        self.vis_time_increment = (self.end_time - self.vis_last_time) / 16

    def ValidateStep(self):
        if self.time.Get() >= self.vis_last_time + self.vis_time_increment:
            self.vis_last_time = self.time.Get()
            self.gf_vis_tmp.Set(
                CF((self.lset.lsetp1, self.cf_neg, self.cf_pos)))
            self.gf_vis.AddMultiDimComponent(self.gf_vis_tmp.vec)

    def AfterLoop(self):
        ngw.Draw(self.gf_vis, self.lset.mesh, "uhnorm", eval_function="value.x>0.0?value.z:value.y",
                 autoscale=self.autoscale, min=self.min, max=self.max, interpolate_multidim=True, animate=True, eval=0)


class UnfittedNGSWebguiScene(Visualization):
    """
    Animate the unfitted scene using the ngsolve webgui while the steps by updating the scene after each step.
    """

    def __init__(self, lset, cf_neg, cf_pos, name=None, step_frequency=None, time_frequency=None,
                 min=0, max=1, autoscale=True):
        """
        Parameters:
        -----------
        lset: LevelSetGeometry
            The levelset that separates the domains
        cf_neg: CoefficientFunction
            The function within the negative part of the levelset function.
        cf_pos: CoefficientFunction
            The function within the positive part of the levelset function.
        name: str
            Name of the plot.
        step_frequency: int
            Only visualize every n-th step
        time_frequency: int
            Only visualize every time this amount of time has passed
        min: float
            Minimimal value of the color map
        max: float
            Maximimimal value of the color map
        autoscale: bool
            Whether to autoscale the color map
        """
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.cf_neg = cf_neg
        self.cf_pos = cf_pos
        self.min = min
        self.max = max
        self.autoscale = autoscale
        self.scene = None

    def BeforeLoop(self):
        self.scene = ngw.Draw(IfPos(self.lset.field, self.cf_pos, self.cf_neg), self.lset.mesh,
                              autoscale=self.autoscale, min=self.min, max=self.max, deformaion=self.lset.deformation)

    def ValidateStep(self):
        self.scene.Redraw()

    def AfterLoop(self):
        pass


### this is just a technical playground for now... 
# I think finally, finally, the pyvista visualizer should be able to draw
# CutFEM scenes based on a level set and two coefficient functions (neg/pos) (+ mesh deformation)
# and/or surface quantities based on level set and one coefficient function (+ mesh deformation)
# probably the two cases could be two separate classes inheriting from a pyvista base class



if pv is not None:

    class PyVistaAnimation(Visualization):
        def __init__(self,
                mesh: Mesh,
                cf_neg: CoefficientFunction,
                cf_pos: CoefficientFunction = None,
                lset: LevelSetGeometry=None,
                subdivision: int = 3,
                export_on_enter: bool = True,
                show_globally: bool = False,
                format: str = "gif",
                framerate: int = 24,
                clim=None,
                camera_bounds=None,
                camera_roll=None,
        ) -> None:
            """
            Parameters
            ----------
            mesh : Mesh
                The NGSolve mesh to export.
            cf_neg : CoefficientFunction
                The function to visualize (in the negative part of the levelset function.)
            cf_neg : CoefficientFunction
                The function to visualize (in the positive part of the levelset function.)
            lset : LevelSetGeometry
                The level set geometry to use.
            subdivision : int, optional
                Subdivision level for the primary VTK export (default: 5).
            export_on_enter : bool, optional
                Whether to export the data on entering the context manager (default: True).
            format : str, optional
                Animation format, "gif" (default) or "mp4" (requires imageio-ffmpeg).
            framerate : int, optional
                Frame rate for movie formats (default: 24).
            clim : tuple, optional
                Fixed (min, max) of the color map. By default the color map is
                rescaled to each frame's data range; pass a fixed range for a
                time-independent scale.
            """
            super().__init__()
            self.lset = lset
            self.cf_neg = cf_neg
            self.cf_pos = cf_pos
            if self.cf_pos is not None:
                self.uh = IfPos(self.lset.lsetp1, self.cf_pos, self.cf_neg)
            else:
                self.uh = self.cf_neg

            self.coefs = [self.lset.lsetp1, self.lset.deformation, self.uh]
            self.coef_names = ["P1-levelset", "deform", "uh"]

            self.mesh: Mesh = mesh
            self.subdivision: int = subdivision
            self.clim = clim

            # Temporary directory for VTK files
            self._tempdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
            self._vtk_files = []
            self.plot = pv.Plotter(notebook=False, off_screen=True)
            self._format = format
            self._anim_file = f"{self._tempdir.name}/animation.{format}"
            if format == "gif":
                self.plot.open_gif(self._anim_file)
            else:
                self.plot.open_movie(self._anim_file, framerate=framerate)
            self.plot.background_color = "white"
            self._mesh_actor = None
            self._text_actor = None

            self.export_on_enter = export_on_enter
            self.counter = 0
            self.show_globally = show_globally
            self.camera_bounds = camera_bounds
            self.camera_roll = camera_roll

        def export_current_step(self) -> None:
            """
            Export mesh and coefficient data to temporary VTK files.
            """
            current_file = f"{self._tempdir.name}/data{self.counter}.vtu"
            vtk = VTKOutput(
                ma=self.mesh,
                coefs=self.coefs,
                names=self.coef_names,
                filename=current_file[:-4],
                subdivision=self.subdivision,
            )
            vtk.Do()
            self._vtk_files.append(current_file)

        def visualize_current_step(self,) -> None:
            """
            Adds the current plot to the GIF.
            """
            visobj = pv.read(self._vtk_files[-1])
            deform = visobj.point_data["deform"]
            if deform.shape[1] == 2:
                deform3d = np.hstack([deform, np.zeros((deform.shape[0], 1))])
                visobj.point_data["deform"] = deform3d

            if not self.show_globally:
                visobj = visobj.clip_scalar(scalars="P1-levelset", value=0.0)

            if deform.shape[1] == 2:
                visobj = visobj.warp_by_vector(vectors="deform")

            # swap only the mesh actor and update the text in place instead
            # of rebuilding the whole scene every frame
            first_frame = self._mesh_actor is None
            if self._mesh_actor is not None:
                self.plot.remove_actor(self._mesh_actor, render=False)
            self._mesh_actor = self.plot.add_mesh(visobj, scalars="uh", cmap="jet",
                                                  clim=self.clim)
            if first_frame and self.mesh.dim == 2:
                # 2D simulation: top-down view instead of the 3D default
                self.plot.view_xy()
                if self.camera_bounds is not None:
                    self.plot.reset_camera(bounds=self.camera_bounds)
                if self.camera_roll is not None:
                    self.plot.camera.Roll(self.camera_roll)
            text = f"Step {self.counter}"
            if self._text_actor is None:
                self._text_actor = self.plot.add_text(text, font_size=10)
            else:
                try:
                    self._text_actor.SetText(2, text)  # 2 = vtkCornerAnnotation.UpperLeft
                except AttributeError:  # add_text did not return a corner annotation
                    self._text_actor = self.plot.add_text(text, font_size=10)
            self.plot.write_frame()  # save current frame to the GIF


        def cleanup(self) -> None:
            """Remove temporary files."""
            self._tempdir.cleanup()

        def __enter__(self) -> "PyVistaAnimation":
            """Enable use as a context manager."""
            if self.export_on_enter:
                self.export_current_step()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            """Clean up temporary files on exit."""
            self.cleanup()

        def BeforeLoop(self):
            pass

        def ValidateStep(self):
            self.export_current_step()
            self.visualize_current_step()
            self.counter += 1

        def AfterLoop(self):
            self.plot.close()
            if self._format == "gif":
                display(Image(filename=self._anim_file))
            else:
                from IPython.display import Video
                display(Video(self._anim_file, embed=True))
            self.cleanup()


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
            scalar_name : str
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
            deform = visobj.point_data["deform"]
            if deform.shape[1] == 2:
                deform3d = np.hstack([deform, np.zeros((deform.shape[0], 1))])
                visobj.point_data["deform"] = deform3d
            if clip_value is not None:
                visobj = visobj.clip_scalar(scalars=clip_name, value=clip_value)

            contour = visobj.contour(isosurfaces=[0.0], scalars=clip_name, rng=[-1, 1])

            #def_contour = contour.warp_by_vector(vectors="deform")
            plot = pv.Plotter()
            plot.background_color = background
            #plot.add_mesh(visobj_mesh, style="wireframe", color=wireframe_color)
            if deform.shape[1] == 2:
                plot.add_mesh(visobj, scalars=scalar_name, cmap=cmap)
            elif deform.shape[1] == 3:
                plot.add_mesh(contour, scalars=scalar_name, cmap=cmap)


            if screenshot:
                plot.show(screenshot=screenshot)
            else:
                if not pv.OFF_SCREEN:
                    plot.show()
                else:
                    # Export als interaktive HTML-Datei
                    html_file = "plot.html"
                    plot.export_html(html_file)
                    with open("plot.html", "r") as f:
                        html_content = f.read()
                    display(HTML(html_content))
                    IFrame(src='./plot.html', width=700, height=600)

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

else:
    class _PyVistaNotInstalledDummy(Visualization):
        """
        No-op stand-in used when pyvista is not installed: it can be registered
        in solver loops and used as a context manager, but all visualization
        calls do nothing.
        """
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning(f"pyvista is not installed: {type(self).__name__} does nothing. "
                           "For some visualizations it may be better to have pyvista installed.")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)

            def noop(*args, **kwargs):
                pass

            return noop


    class PyVistaAnimation(_PyVistaNotInstalledDummy):
        pass


    class PyVistaVisualizer(_PyVistaNotInstalledDummy):
        pass
