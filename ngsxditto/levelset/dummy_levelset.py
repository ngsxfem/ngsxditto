from .levelset import *
from ngsxditto.transport.known_solution_transport import *


class DummyLevelSet(LevelSetGeometry):
    """
    A dummy levelset that is negative everywhere.
    """
    def __init__(self, mesh):
        """
        Initialize the dummy levelset.
        Parameters:
        ----------
        mesh : Mesh
            The computational mesh
        """
        true_solution = CF(-1)
        transport = KnownSolutionTransport(mesh, true_solution)
        super().__init__(transport=transport, redistancing=None, autoredistancing=None, initial_levelset=true_solution)


