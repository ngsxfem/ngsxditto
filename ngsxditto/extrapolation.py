"""Time extrapolation/interpolation of state history for coupled problems.

:class:`Extrapolator` stores states with associated time and allows to 
extra-/interpolate (through polynomial interpolation) at a different time.

It is fed with ``(time, state)`` and stores up to ``order + 1`` states 
in a ring buffer. 

At startup (too few states yet) only available data is used reducing 
the *effective* order to ``len(history) - 1``
(see :meth:`Extrapolator.EffectiveOrder`).

Typical behaviour in a coupled loop:

* At the start of a new time step only validated history ``..., w^{n-1}, w^n``
  is available, so evaluating at, e.g., the midpoint ``t^{n+1/2}`` is an
  *extrapolation*.
* Under strong (Picard) coupling every sub-iteration may feed the freshest
  endpoint iterate back in at time ``t^{n+1}``. Because :meth:`Feed` *overwrites*
  a same-time entry, the endpoint node is refined in place, and from the second
  sub-iteration on the very same midpoint evaluation becomes an *interpolation*
  between ``w^n`` and the (now available) ``w^{n+1}``.

A :class:`Stepper` is a good candidate to be an `Extrapolator` at the same time.
A :class:`ExtrapolatorSource` implements such a superclass:  
* at the end of each (sub-)iteration, feeds its state to the extrapolator.
The default behavior is to do that *every* iteration, but with the 
``only_on_validate`` flag feeding is only done on validated steps.
"""
import bisect

from ngsolve import GridFunction, BaseVector

__all__ = ["Extrapolator", "ExtrapolatorSource"]


class Extrapolator:
    """Polynomial time extrapolation/interpolation over a ring of states.

    The extrapolator owns its storage: it copies the vectors of the states it
    is fed and holds its own output :class:`~ngsolve.GridFunction` (same space
    as the states) whose vector is overwritten on every evaluation. Hand
    :attr:`gf` to the consumer once; re-evaluating updates it in place.

    Parameters
    ----------
    order: int
        Target (polynomial) order. The ring keeps the last ``order + 1`` states;
        with fewer states the effective order drops accordingly (startup).
    atol, rtol: float
        A fed time is considered to coincide with a stored time (and then
        *overwrites* that node in place, instead of appending a new one) when
        ``abs(time - t_i) <= atol + rtol * abs(t_i)`` for some stored ``t_i``
        -- no two stored nodes ever share a (near-)coincident time. This makes
        the Picard re-feed of the endpoint at a fixed ``t^{n+1}`` robust
        against the floating-point noise of the loop clock (``t += dt`` /
        ``t -= dt``). Nodes need not be fed in chronological order.
    """

    def __init__(self, order: int, atol: float = 1e-10, rtol: float = 1e-9):
        if order < 0:
            raise ValueError("Extrapolator: order must be >= 0")
        self.order = order
        self.atol = atol
        self.rtol = rtol
        self._times = []          # stored times, oldest first, len <= order+1
        self._vecs = []           # owned bare vectors, parallel to _times
        self._space = None        # FESpace, known once a GF is fed or SetSpace is called
        self._out = None          # output: GridFunction if _space is known, else a bare vector
        self._last_evaluated_time = None  # cache key; invalidated by any Feed

    # --- introspection ------------------------------------------------------
    @property
    def gf(self):
        """The output GridFunction (overwritten by :meth:`Evaluate`)."""
        if self._space is None:
            raise RuntimeError(
                "Extrapolator: no space set yet -- feed a GridFunction once, "
                "or call SetSpace(...) explicitly, before accessing .gf")
        return self._out

    @property
    def vec(self):
        """The output as a bare vector (overwritten by :meth:`Evaluate`).

        Available regardless of whether a space is known.
        """
        if self._out is None:
            raise RuntimeError("Extrapolator: no state has been fed yet")
        return self._out.vec if self._space is not None else self._out

    def SetSpace(self, space):
        """Register the FESpace of the fed states, enabling :attr:`gf`.

        Called automatically by :meth:`Feed` whenever a GridFunction is fed.
        Call it explicitly up front if you only ever feed bare vectors but
        still want GridFunction output.
        """
        if self._space is space:
            return
        self._space = space
        out = GridFunction(space)
        if self._out is not None:
            out.vec.data = self._out
        self._out = out

    def EffectiveOrder(self) -> int:
        """Polynomial order actually used for the next evaluation.

        ``len(history) - 1`` capped at the target ``order``: with a single
        state this is 0 (piecewise constant), and it grows as history builds up.
        """
        return max(0, len(self._times) - 1)

    def _match_index(self, time):
        """Index of a stored node coinciding with ``time`` (see ``atol``/``rtol``),
        or ``None``. Checked against *all* stored nodes, not just the last."""
        for i, t in enumerate(self._times):
            if abs(time - t) <= self.atol + self.rtol * abs(t):
                return i
        return None

    # --- feeding ------------------------------------------------------------
    def Feed(self, time: float, state):
        """Register a ``(time, state)`` sample.

        ``state`` may be a :class:`~ngsolve.GridFunction` or a bare
        :class:`~ngsolve.la.BaseVector`. If it is a GridFunction its space is
        registered via :meth:`SetSpace` (enabling :attr:`gf`); if it is a
        bare vector, the space must already be known (via a prior GF feed or
        an explicit :meth:`SetSpace` call) for :attr:`gf` to work, but is not
        otherwise required.

        A copy of ``state``'s vector is stored. If ``time`` coincides with any
        stored time (see ``atol``/``rtol``) that node is *overwritten* in
        place (endpoint refinement during Picard iterations) -- no two stored
        nodes ever share a (near-)coincident time; otherwise a new node is
        inserted (nodes need not be fed chronologically) and, once
        ``order + 1`` nodes are stored, the node with the smallest (oldest)
        time drops out of the ring.
        """
        if isinstance(state, GridFunction):
            self.SetSpace(state.space)
            vec = state.vec
        else:
            vec = state

        if self._out is None:
            self._out = vec.CreateVector()

        self._last_evaluated_time = None  # ring changed -> invalidate Evaluate's cache

        i = self._match_index(time)
        if i is not None:
            self._vecs[i].data = vec
            self._times[i] = time
            return

        if len(self._times) == self.order + 1:
            # evict the oldest (smallest-time) buffer to avoid per-step allocations
            evict = min(range(len(self._times)), key=self._times.__getitem__)
            buf = self._vecs.pop(evict)
            self._times.pop(evict)
            buf.data = vec
        else:
            buf = vec.CreateVector()
            buf.data = vec

        pos = bisect.bisect_left(self._times, time)
        self._times.insert(pos, time)
        self._vecs.insert(pos, buf)

    # --- evaluation ---------------------------------------------------------
    def Evaluate(self, time: float):
        """Evaluate the history polynomial at ``time``, overwriting :attr:`gf`/:attr:`vec`.

        Fits the polynomial through all stored nodes (degree
        :meth:`EffectiveOrder`) and evaluates it at ``time`` via the Lagrange
        form. Returns the (updated) output -- a GridFunction if the space is
        known, otherwise a bare vector.

        A repeated call with the same ``time`` is a no-op (returns the cached
        output) as long as the ring hasn't changed since the last evaluation
        -- any :meth:`Feed` invalidates the cache.
        """
        if self._out is None:
            raise RuntimeError("Extrapolator.Evaluate: no state has been fed yet")

        if time == self._last_evaluated_time:
            return self._out

        ts = self._times
        n = len(ts)
        # Lagrange weights L_i(time) = prod_{j!=i} (time - t_j)/(t_i - t_j)
        weights = []
        for i in range(n):
            li = 1.0
            for j in range(n):
                if j != i:
                    li *= (time - ts[j]) / (ts[i] - ts[j])
            weights.append(li)

        out_vec = self._out.vec if self._space is not None else self._out
        out_vec.data = weights[0] * self._vecs[0]
        for i in range(1, n):
            out_vec.data += weights[i] * self._vecs[i]
        self._last_evaluated_time = time
        return self._out


class ExtrapolatorSource:
    """Mixin: a stepper that feeds :class:`Extrapolator`\\ s from its state.

    The feeding is deliberately owned by the *producing* object and tied to the
    point where it commits its state in the loop (``AcceptIntermediate`` for a
    continued inner iteration, ``ValidateStep`` at the end of the time step),
    so the extrapolate->interpolate transition works out. Mix this in *before*
    the concrete stepper base so its ``AcceptIntermediate``/``ValidateStep`` run
    first and still forward to ``super()``::

        class LevelsetBasedExtension(ExtrapolatorSource, StatelessStepper):
            ...

    Usage example::

        wind = Extrapolator(order=1)
        velocity_extension.FeedInto(wind, time=t, state=velocity_extension.field)
        transport.SetWind(wind.gf)

    ``time`` is a ``Parameter`` read at feed time (the loop is the single time
    authority); ``state`` defaults to the stepper's ``current`` state.
    """

    def FeedInto(self, extrapolator: "Extrapolator", time, state=None,
                 only_on_validate: bool = False):
        """Associate ``extrapolator`` to be fed from ``state`` at ``time``.

        Parameters
        ----------
        extrapolator: Extrapolator
            The target to feed.
        time: ngsolve.Parameter
            The (loop-owned) time parameter; ``time.Get()`` is used as the node
            time when feeding.
        state: GridFunction or BaseVector, optional
            The state to feed; defaults to ``self.current``. May be a bare
            vector instead of a GridFunction, see :meth:`Extrapolator.Feed`.
        only_on_validate: bool
            If ``True``, feed only on validated steps (explicit predictor,
            no coupling requirement). Default ``False``: feed every iteration.
        """
        if not hasattr(self, "_extrapolator_feeds"):
            self._extrapolator_feeds = []
        self._extrapolator_feeds.append((extrapolator, time, state, only_on_validate))
        return extrapolator

    def SeedExtrapolators(self):
        """Feed the current state to all associated extrapolators once (e.g.
        the initial value ``w^0`` before the loop starts)."""
        self._feed_extrapolators(validated=True)

    def _feed_extrapolators(self, validated: bool):
        for extrapolator, time, state, only_on_validate in getattr(
                self, "_extrapolator_feeds", []):
            if validated or not only_on_validate:
                src = state if state is not None else self.current
                extrapolator.Feed(time.Get(), src)

    # --- lifecycle hooks (forward to the concrete stepper base) -------------
    def AcceptIntermediate(self):
        self._feed_extrapolators(validated=False)
        super().AcceptIntermediate()

    def ValidateStep(self):
        self._feed_extrapolators(validated=True)
        super().ValidateStep()
