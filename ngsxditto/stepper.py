from ngsolve import *


class Stateholder:
    def __init__(self, store_intermediate=True):
        self.past = None
        self.intermediate = None
        self.store_intermediate = store_intermediate


    def StoreState(self):
        self.past[:] = self.current.vec.data
        self.intermediate[:] = self.current.vec.data

    def StoreIntermediate(self):
        self.intermediate[:] = self.current.vec.data


    def ComputeDifference2Intermediate(self):
        raise NotImplementedError("ComputeDifference2Intermediate must be implemented by subclass")


    def Step(self):
        self.current.vec.data = self.past[:]
        self.UpdateStates()



        if self.store_intermediate:
            self.StoreIntermediate()


    def UpdateStates(self):
        raise NotImplementedError("UpdateStates must be implemented by subclasses.")

    @property
    def current(self):
        return None