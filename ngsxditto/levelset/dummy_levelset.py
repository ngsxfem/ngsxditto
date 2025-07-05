from .levelset import *
from ngsxditto.transport.known_solution_transport import *


class DummyLevelSet(LevelSetGeometry):
    def __init__(self, mesh):
        true_solution = CF(-1)
        transport = KnownSolutionTransport(mesh, true_solution)
        super().__init__(transport=transport, redistancing=None, autoredistancing=None, initial_levelset=true_solution)


