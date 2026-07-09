# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Concept: Steppers, `ValidateStep` and driving loops
#
# Everything that evolves in an `ngsxditto` simulation -- a transport solver, a
# level-set geometry, a fluid discretization -- is a `Stepper`. The contract is
# small:
#
# * `Step()`: compute a *candidate* for the new state,
# * `ValidateStep()`: commit the candidate (the outer loop accepted the step),
# * `AcceptIntermediate()`: keep the candidate as intermediate result of an
#   inner (sub-)iteration -- the step is not committed yet,
# * `RevertStep()`: discard the candidate and fall back to the committed state.
#
# A `StatefulStepper` backs this up with three states: `past` (last committed
# step), `intermediate` (last sub-iterate) and `current` (candidate). The
# workhorse is `GFStepper`, whose states are `ngsolve.GridFunction`s.
#
# **Single time authority:** a stepper never advances the time parameter
# itself -- only the *driving loop* does (`TimeLoop`, `MultiStepper`, or your
# own loop). This keeps several coupled steppers on one consistent clock.

# %%
from ngsolve import *
from ngsxditto import TimeLoop
from ngsxditto.stepper import GFStepper

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
fes = NumberSpace(mesh)                       # one global DOF -- state is a single number

class Doubler(GFStepper):
    """Toy stepper: the candidate state is twice the committed state."""
    def __init__(self, fes):
        super().__init__()
        self.past, self.intermediate, self.current = (GridFunction(fes) for _ in range(3))
        self.past.vec[0] = self.intermediate.vec[0] = self.current.vec[0] = 1.0

    def Step(self):
        self.current.vec[0] = 2 * self.past.vec[0]

def states(s):
    return f"past={s.past.vec[0]:4.1f}  intermediate={s.intermediate.vec[0]:4.1f}  current={s.current.vec[0]:4.1f}"

# %% [markdown]
# ## Part 1: the state machine
#
# `Step()` only touches `current`. Nothing is committed until the driving loop
# decides the step was good and calls `ValidateStep()` (current → past and
# intermediate). If the step was bad, `RevertStep()` resets current and
# intermediate back to `past` -- as if `Step()` had never happened.

# %%
s = Doubler(fes)
print("initial:       ", states(s))
s.Step()
print("after Step:    ", states(s))          # only `current` changed
s.ValidateStep()
print("after Validate:", states(s))          # candidate committed

s.Step()
print("after Step:    ", states(s))
s.RevertStep()
print("after Revert:  ", states(s))          # candidate discarded

# %% [markdown]
# `AcceptIntermediate()` is the third outcome, used *inside* sub-iteration
# loops (e.g. a Picard coupling): the candidate becomes the new `intermediate`
# so the next sub-iterate can measure its update against it
# (`ComputeDifference2Intermediate`), while `past` stays untouched until the
# whole sub-iteration converged and `ValidateStep()` commits it.

# %%
s.Step()
s.AcceptIntermediate()
print("after Accept:  ", states(s))          # intermediate follows, past does not

# %% [markdown]
# ## Part 2: manual time loops and the clock
#
# In a hand-written loop *you* are the driving loop: call `Step()` and
# `ValidateStep()`, then advance the clock yourself. A stepper's `Step()` does
# **not** move `time` (single time authority) -- forgetting the last line below
# is the classic way to write an infinite `while time < T_end` loop.

# %%
t = Parameter(0.0)
dt = 0.1

s = Doubler(fes)
s.time = t
while t.Get() < 0.3:
    s.Step()
    s.ValidateStep()
    t.Set(t.Get() + dt)   # the driving loop advances time, not the stepper
print(f"t={t.Get():.1f}, committed state={s.past.vec[0]}")

# %% [markdown]
# ## Part 3: `TimeLoop` as driving loop
#
# For anything beyond a quick experiment, register steppers in a `TimeLoop`
# (a `Solver` whose progress is measured by a time parameter). It calls
# `Step()` on every registered object each iteration, advances the clock, and
# routes each iteration to exactly one of the three outcomes:
#
# * `should_finalize()` true (default: always) → `ValidateStep()` on everyone,
# * `should_revert()` true → `RevertStep()` on everyone,
# * neither → `AcceptIntermediate()` and the *inner* loop continues.
#
# Plain functions can be registered too -- they are wrapped as stateless
# steppers. `step_frequency`/`time_frequency` let a stepper run only every
# n-th step or once per time interval (e.g. output writers).

# %%
calls = {"every_step": 0, "every_2nd": 0}

s = Doubler(fes)
def count_step():   calls["every_step"] += 1
def count_2nd():    calls["every_2nd"] += 1

time_loop = TimeLoop(time=Parameter(0.0), dt=0.1, end_time=1.0,
                     display_progress_bar=False, show_profiles=False)
time_loop.Register(s, name="doubler")
time_loop.Register(count_step, name="counter")
time_loop.Register(count_2nd, name="every 2nd step", step_frequency=2)
time_loop()

print(f"time after loop: {time_loop.time.Get():.1f}")
print(f"committed state: {s.past.vec[0]} (= 2^10)")
print(f"calls: {calls}")

# %% [markdown]
# ## Part 4: sub-iterations -- `should_finalize` and `AcceptIntermediate`
#
# With a `should_finalize` rule the loop sub-iterates: every iteration that is
# *not* finalized ends in `AcceptIntermediate()`, only the finalizing one in
# `ValidateStep()` (and only then the clock moves on). Here we finalize every
# second inner iteration and count what the stepper actually sees:

# %%
class CountingStepper(Doubler):
    def __init__(self, fes):
        super().__init__(fes)
        self.n_validate = self.n_accept = 0
    def ValidateStep(self):
        self.n_validate += 1
        super().ValidateStep()
    def AcceptIntermediate(self):
        self.n_accept += 1
        super().AcceptIntermediate()

s = CountingStepper(fes)
time_loop = TimeLoop(time=Parameter(0.0), dt=0.25, end_time=1.0,
                     display_progress_bar=False, show_profiles=False)
time_loop.SetFinalizeRule(lambda: time_loop.i_inner % 2 == 0)   # 2 sub-iterations per step
time_loop.Register(s, name="doubler")
time_loop()

print(f"validated steps:        {s.n_validate}")
print(f"accepted sub-iterates:  {s.n_accept}")
