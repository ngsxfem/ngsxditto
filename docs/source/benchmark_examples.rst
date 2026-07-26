Benchmark examples
==================

The :doc:`module_examples` and :doc:`application_examples` are lightweight,
*interactive* notebooks meant to be run and explored directly -- they render
live ``webgui`` scenes and finish in seconds.

The **benchmarks** collected here are different: they are heavier, quantitative
*validation studies* (mesh/time-step convergence against published reference
values) that take minutes and are therefore **not executed by the
continuous-integration documentation build**. The code is shown for reference
and stays fully runnable, but the reported numbers and all figures/animations
are *pre-computed and embedded* so the CI pipeline remains lean. Large image
and animation assets are stored with git-LFS.

.. toctree::
   :maxdepth: 2
   :caption: Benchmark examples:

   hysing_turek_case1.ipynb
