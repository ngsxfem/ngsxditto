Module examples
================

The notebooks here introduce individual `ngsxditto` classes -- the building
blocks of a simulation, such as `LevelSetGeometry`, redistancing, transport,
extension or the unfitted Stokes discretizations -- each on its own, without
combining them into a coupled problem. See :doc:`application_examples` for
how these modules are combined to solve (simple) coupled problems, and
:doc:`concepts` for the underlying `Stepper` / `Extrapolator` infrastructure
they are built on.

.. toctree::
   :maxdepth: 2
   :caption: Module examples:

   ditto_lset.ipynb

   mean_curv_ditto.ipynb

   transport_and_redistancing.ipynb

   element_based_extension.ipynb

   narrow_band_transport_implicitdg.ipynb

   basic_stokes.ipynb

   moving_stokes.ipynb
