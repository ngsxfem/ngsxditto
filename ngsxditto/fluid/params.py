"""
This file introduces placeholder classes working just as an dictionary, holding corresponding parameters values for some fluid.
"""

from typing import Optional


class FluidParameters:
    """
    This class represents the fluid parameters as a dictionary.
    """
    def __init__(self, viscosity: float = 1e-3, density: float = 1, surface_tension_coeff: float = 0.072):
        """
            parameters:
                viscosity: viscosity 
                density: density
                surface_tension_coeff: surface_tension_coeff
        """ 
        # TODO: dynamic and kinematic viscosity
        self.viscosity = viscosity
        self.density = density
        self.surface_tension_coeff = surface_tension_coeff
        self.dictionary = {"viscosity": viscosity, "density": density, "surface_tension_coeff": surface_tension_coeff}

        # do sanity checks
        # ...

    def Update(self, viscosity: Optional[float] = None, density: Optional[float] = None, surface_tension_coeff: Optional[float] = None) -> dict:
        """
        This class represents wall parameters as a dictionary.
    
            parameters:
                viscosity: viscosity nu
                density: density rho
                surface_tension_coeff: surface tension coeff
        """
        self._UpdateDict(viscosity=viscosity, density=density, surface_tension_coeff=surface_tension_coeff)
        return self.dictionary

    def _UpdateDict(self, viscosity=None, density=None, surface_tension_coeff=None):
        # do not update parameter, get from current parameter
        for param in ["viscosity", "density", "surface_tension_coeff"]:
            if param == None:
                param = self.dictionary[param]
        self.dictionary = {"viscosity": viscosity, "density": density, "surface_tension_coeff": surface_tension_coeff}

    def __getitem__(self, param: str) -> float:
        return self.dictionary[param]


class WallParameters:
    """
    This class represents wall parameters as a dictionary.
    """
    def __init__(self, friction_coeff: float = 0, contact_angle: float = 0):
        """
            parameters:
                friction_coeff: friction coefficient
                contact_angle: contact angle
        """
        self.friction_coeff = friction_coeff
        self.contact_angle = contact_angle
        self.dictionary = {"friction_coeff": friction_coeff, "contact_angle": contact_angle}

    def Update(self, friction_coeff: Optional[float] = None, contact_angle: Optional[float] = None):
        """
            Updates WallParameters-object with given values. If no values were given for entry, use old one.

            parameters:
                friction_coeff: friction coefficient
                contact_angle: contact angle
        """
        self._UpdateDict(friction_coeff=friction_coeff, contact_angle=contact_angle)

    def _UpdateDict(self, friction_coeff: Optional[float] = None, contact_angle: Optional[float] = None):
        # do not update parameter, get from current parameter
        for param in ["friction_coeff", "contact_angle"]:
            if param == None:
                param = self.dictionary[param]
        self.dictionary = {"friction_coeff": friction_coeff, "contact_angle": contact_angle}

    def __getitem__(self, param):
        return self.dictionary[param]
