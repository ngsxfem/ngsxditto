ngsxditto concepts
==================

`ngsxditto` is built around a few small, reusable pieces of infrastructure --
the `Stepper` protocol that drives every simulation loop, the `Extrapolator`
used for higher order accuracy in time, and (planned) profiling/timing
helpers. The notebooks here explain each of them at a small, prototypical
example, decoupled from any concrete PDE -- useful as a quick introduction
before diving into a :doc:`module_examples` or :doc:`application_examples`.

.. toctree::
   :maxdepth: 2
   :caption: Concepts:

   concepts_stepper.ipynb
   concepts_extrapolator.ipynb
