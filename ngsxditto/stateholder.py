

class Stateholder:
    def __init__(self):
        self.past = None


    def StoreState(self):
        self.past[:] = self.current.vec.data


    def Step(self):
        self.current.vec.data = self.past[:]
        self.UpdateStates()



    def UpdateStates(self):
        raise NotImplementedError("UpdateStates must be implemented by subclasses.")