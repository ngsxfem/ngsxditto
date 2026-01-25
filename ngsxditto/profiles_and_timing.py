import time
import threading
from contextlib import contextmanager
from collections import defaultdict



def timed_method(name=None):
    def decorator(fn):
        if name:
            section = name
        else:
            section = fn.__name__

        def wrapper(self, *args, **kwargs):
            # if object is registered in solver -> exclusive
            exclusive = getattr(self, "_solver", None) is not None
            with self.timer(section, exclusive=exclusive):
                return fn(self, *args, **kwargs)
        wrapper._timed_section = True
        return wrapper
    return decorator

_current_object = threading.local()


class Timed:
    def __init__(self):
        self.times = defaultdict(float)
        self._solver = None

    @contextmanager
    def timer(self, section, exclusive=True):
        now = time.perf_counter
        stack = getattr(_current_object, "stack", [])
        _current_object.stack = stack

        start = now()

        if exclusive:
            target_obj = self
            target_section = stack[-1][1] if (stack and stack[-1][0] is self and stack[-1][1] != "__total__") else section
        else:
            # Use top of stack if exists, otherwise self
            target_obj = stack[-1][0] if stack else self
            target_section = stack[-1][1] if stack else section
        # push this timer onto stack
        #stack.append((self, section, start, exclusive))
        stack.append((target_obj, target_section, start, exclusive))
        try:
            yield
        finally:
            end = now()
            obj, sec, obj_start, exclusive_flag = stack.pop()
            elapsed = end - obj_start
            target_obj.times[target_section] += elapsed

            if target_section != "__total__":
                target_obj.times["__total__"] += elapsed

            if stack:
                parent_obj, parent_sec, parent_start, parent_excl = stack[-1]
                if parent_excl:
                    #pause parent: shift start forward to exclude elapsed
                    stack[-1] = (parent_obj, parent_sec, end, parent_excl)

    def reset_times(self):
        self.times.clear()

    def TimeExtra(self, fn, name):
        def wrapper(*args, **kwargs):
            exclusive = self._solver is not None
            with self.timer(name, exclusive=exclusive):
                return fn(*args, **kwargs)

        return wrapper
