import logging

logger = logging.getLogger(__name__)
logger.debug("importing ngsxditto")

direct_solver_spd = "sparsecholesky"
direct_solver_nonspd = "pardiso"

from .transport import *
from .fluid import *
from .levelset import *
from .redistancing import *
from .multistepper import *
from .gradient_tester import *
from .extension import *
from .callback import *
from .solver import *
from .visualization import *
from .stepper import *
from .two_phase import *
from .progress_info import *
