from .redistancing import *


class OnUpdateCallbacks:
    def __init__(self):
        self.callbacks = []


    def AddCallback(self, func, index=None):
        if callable(func):
            if index is None:
                self.callbacks.append(func)
            else:
                self.callbacks.insert(index, func)
        else:
            raise ValueError("Callback must be callable")


    def ProcessCallbacks(self):
        for callback in self.callbacks:
            callback()
