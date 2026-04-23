from ngsolve import CoefficientFunction


class BoundaryCondition:
    def __init__(self, region:str, values:CoefficientFunction):
        self.values = values
        self.region = region


class NitscheVelocityBC(BoundaryCondition):
    def __init__(self, region, values):
        super().__init__(region, values)

class StrongDirichletBC(BoundaryCondition):
    def __init__(self, region, values):
        super().__init__(region, values)


class StrongNeumannBC(BoundaryCondition):
    def __init__(self, region, values):
        super().__init__(region, values)


class NitscheNormalVelocityBC(BoundaryCondition):
    def __init__(self, region, values):
        super().__init__(region, values)


class BoundaryRegistry:
    def __init__(self):
        self.all_bc_dict = {}

        self.strong_dirichlet_dict = {}
        self.strong_neumann_dict = {}
        self.nitsche_velocity_dict = {}
        self.nitsche_normal_velocity_dict = {}
        self.dbnd = ""

    def AddBoundaryCondition(self, condition: BoundaryCondition):
        if isinstance(condition, StrongDirichletBC):
            self.all_bc_dict[condition.region] = {"function": condition.values, "type": StrongDirichletBC.__name__}
            self.strong_dirichlet_dict[condition.region] = condition.values
            if self.dbnd == "":
                self.dbnd = condition.region
            else:
                self.dbnd = "|".join([self.dbnd, condition.region])

        elif isinstance(condition, StrongNeumannBC):
            self.all_bc_dict[condition.region] = {"function": condition.values, "type": StrongNeumannBC.__name__}
            self.strong_neumann_dict[condition.region] = condition.values

        elif isinstance(condition, NitscheVelocityBC):
            self.all_bc_dict[condition.region] = {"function": condition.values, "type": NitscheVelocityBC.__name__}
            self.nitsche_velocity_dict[condition.region] = condition.values

        elif isinstance(condition, NitscheNormalVelocityBC):
            self.all_bc_dict[condition.region] = {"function": condition.values, "type": NitscheNormalVelocityBC.__name__}
            self.nitsche_normal_velocity_dict[condition.region] = condition.values
