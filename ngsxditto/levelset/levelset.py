from ngsolve import *
from ngsxditto.transport import *
from ngsxditto.redistancing import *


class LevelSetGeometry:
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing):
        self.transport = transport
        self.redistancing = redistancing
        self.mesh = self.transport.mesh


