Test coverage
=============

The `pytest-cov` HTML report from the test suite (``coverage.xml`` / ``htmlcov/``,
built by the ``test`` job in CI) is embedded below.

.. raw:: html

    <iframe src="_static/coverage/index.html"
            style="width:100%;height:80vh;border:1px solid #ccc;border-radius:4px;">
    </iframe>

If the frame above is empty, the report was not available at build time -- this
happens for a local ``generate_docs.sh`` run unless ``docs/htmlcov/`` is
populated first (see the ``build_pages`` / ``build-pages`` CI job, which copies
the ``test`` job's coverage artifact there before building).
