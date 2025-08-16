from .redistancing import *


class OnUpdateCallbacks:
    """
    This class handles callbacks that are automatically applied after a function call.
    """
    def __init__(self):
        """
        Initialize an empty callback list.
        """
        self.callbacks = []


    def AddCallback(self, func, index:int=None):
        """
        Adds a callback to the callback list.
        Parameters:
        ----------
        func: callable
            The function to add.
        index: int
            The index in the callback list the function is inserted (default: last).
        """
        if callable(func):
            if index is None:
                self.callbacks.append(func)
            else:
                self.callbacks.insert(index, func)
        else:
            raise ValueError("Callback must be callable")


    def ProcessCallbacks(self):
        """
        Runs all functions in the callback list.
        """
        for callback in self.callbacks:
            callback()
