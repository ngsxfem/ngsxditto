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
# # Concept: `Extrapolator` and `ExtrapolatorSource`
#
# Many handover quantities in a coupled (sub-iterated) time loop are needed at
# a time *inside* the current step interval, not just at its endpoint -- e.g.
# the transport wind, which the level-set update integrates over the whole
# interval $[t^n, t^{n+1}]$. `ngsxditto.extrapolation` provides two small,
# general-purpose building blocks for this:
#
# * `Extrapolator`: fit a polynomial through the last fed `(time, state)`
#   samples and evaluate it at any other time.
# * `ExtrapolatorSource`: a stepper mixin that feeds an `Extrapolator`
#   automatically, at the right point in a solver loop.

# %%
import numpy as np
from ngsolve import *
from ngsxditto.extrapolation import Extrapolator, ExtrapolatorSource
from ngsxditto.stepper import GFStepper

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
fes = NumberSpace(mesh)                      # single global DOF -- a "field" is just a number

def field(t):
    """Synthetic time-dependent scalar w(t) = sin(t) -- stand-in for e.g. a wind."""
    g = GridFunction(fes)
    g.vec[0] = sin(t)
    return g

exact = np.sin                                # exact value of `field(t)`

# %% [markdown]
# ## Part 1: `Extrapolator`
#
# An `Extrapolator(order=1)` keeps the last two distinct `(time, state)`
# samples and fits a straight line through them. Evaluating *between* the
# stored times is interpolation; evaluating *outside* is extrapolation -- the
# same object does both, depending only on where the requested time falls.

# %%
for order in [0,1,2,3]:
    wind = Extrapolator(order=order)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        wind.Feed(t, field(t))                    # ring keeps only the last order+1 samples

    for t_eval in [0.66, 1.5]:                     # 1.66 interpolates, 2.5 extrapolates
        kind = "interpolation" if t_eval < 2.0 else "extrapolation"
        wind.Evaluate(t_eval)
        print(f"t={t_eval:4.2f} ({kind}, order={order}): estimate={wind.vec[0]:.4f}  exact={exact(t_eval):.4f}")

# %% [markdown]
# Feeding a time that (numerically) coincides with an *already stored* node
# overwrites that node in place instead of appending a new one -- this is what
# lets a Picard sub-iteration refine an endpoint estimate without growing the
# ring or double-counting it:

# %%
print("ring size before re-feed of t=1.0:", wind.EffectiveOrder() + 1)
wind.Feed(1.0, field(2.0))                    # same t=1.0 node -> refines it in place
print("ring size  after re-feed of t=1.0:", wind.EffectiveOrder() + 1)

# %% [markdown]
# ## Part 2: `ExtrapolatorSource`
#
# `ExtrapolatorSource` is a mixin for steppers that *produce* the fed state.
# It feeds registered extrapolators automatically: every (sub-)iteration via
# `AcceptIntermediate` by default, or only on validated steps if
# `only_on_validate=True` is passed to `FeedInto`. This is exactly the
# extrapolate -> interpolate transition sketched above, wired up for you.
#
# Below, a toy stepper reproduces `field(t)` above. We feed *two* extrapolators
# from it -- one updated every sub-iteration, one only on validated steps --
# and run a hand-rolled Picard loop (2 sub-iterations per step) to watch them
# behave differently.

# %%
class ToyWind(ExtrapolatorSource, GFStepper):
    def __init__(self, fes):
        super().__init__()
        self.current, self.intermediate, self.past = (GridFunction(fes) for _ in range(3))

    def Step(self, t):
        self.current.vec[0] = sin(t)

t = Parameter(0.0)
producer = ToyWind(fes)
producer.Step(t.Get())                                    # w^0

every_iter = Extrapolator(order=1)
validated_only = Extrapolator(order=1)
producer.FeedInto(every_iter, time=t)
producer.FeedInto(validated_only, time=t, only_on_validate=True)
producer.SeedExtrapolators()                                # feed w^0 to both, once

# %%
dt = 0.5
for step in range(2):
    t.Set(t.Get() + dt)
    for sub_iter in range(2):                               # toy Picard sweep
        producer.Step(t.Get())
        mid = t.Get() - dt / 2
        print(f"step {step} sub {sub_iter}: "
              f"every_iter={every_iter.Evaluate(mid).vec[0]:.4f}  "
              f"validated_only={validated_only.Evaluate(mid).vec[0]:.4f}  "
              f"exact={exact(mid):.4f}")
        producer.AcceptIntermediate()                       # feeds `every_iter` only
    producer.ValidateStep()                                 # feeds both (validated_only catches up)