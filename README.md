# ngsxditto

**ngsxditto** is a high-level library for solving PDEs on moving domains using level-set-based methods and unfitted finite element techniques. Inspired by the flexibility and smooth boundaries of shape-shifting forms, the library emphasizes robust numerical schemes, including ghost penalty stabilization, to handle evolving interfaces.

> *"Like its namesake, ngsxditto smoothly adapts to evolving shapes — with mathematical rigor."*

## 🚀 Features

- Level-set based interface capturing
- Geometrically unfitted FEM (CutFEM/XFEM-style)
- Stabilization via ghost penalty and aggregation techniques
- Higher order methods (in geometry and field approximation)
- Modular design for extension to multi-physics problems
- Compatible with [ngsolve](https://ngsolve.org)

Features aimed at:
- space-time methods


## Installation without cloning
When using this library from outside the repo install it first via `pip3 install --user git+https://gitlab.gwdg.de/ngsuite/ngsxditto`.

Or install from the package registry:
`pip install ngsxditto --index-url https://__token__:<your_personal_token>@gitlab.gwdg.de/api/v4/projects/45482/packages/pypi/simple` with your personal access token put in here.

## Installation from repo
The first two commands are optional.
However, we assume you have a python installation available.
  * `python3 -m venv .venv`
  * `source .venv/bin/activate`
  * `pip3 install . --user` (uses the `pyproject.toml` to install the source code)

With the following command, you can verify the installation works
  * `pytest tests/test_*.py`

## Using / developing
If you are working in the library directly (working on a module, geometry, test or example) you can use `pip install -e .` so that the installation only links to your source files and source file changes have immediate impact.

## Available modules:
  * `transport`: level-set transport solvers — explicit DG, implicit DG and implicit SUPG,
    all usable on narrow bands via `active_elements`; `KnownSolutionTransport` for manufactured solutions
  * `levelset`: `LevelSetGeometry` — combines transport, redistancing and isoparametric mesh
    deformation; provides cut information (`hasif`, ...) and integration measures (`dS`, `dCut`-based)
  * `redistancing`: fast marching (level-set order 1 and 2), with periodic auto-redistancing
  * `extension` / `velocity_extension`: element-based and levelset-based extension of fields
    into the bulk / narrow band
  * `fluid`: fitted and unfitted Stokes discretizations (Taylor–Hood, Scott–Vogelius,
    H1-conforming with aggregation), mean curvature computation
  * `two_phase`: two-phase Stokes flow (e.g. oscillating droplet, second order in time via BDF2)
  * infrastructure: `Stepper` / `Solver` / `TimeLoop` framework (validated/reverted steps,
    sub-iterations, profiling) and polynomial-in-time `Extrapolator`

Planned / in progress:
  * space-time discretizations

## Examples

see in the `examples` directory.

The examples are stored as plain python files (jupytext percent format) — these are the only versioned source. Matching jupyter notebooks can be generated from them and edited notebooks can be synced back into the python files (`.ipynb` files are git-ignored):

  * install jupytext once: `pip install jupytext`
  * generate/update the notebooks (and sync notebook edits back): `jupytext --sync examples/*.py`

The pairing is configured in `examples/jupytext.toml`. If you work in Jupyter(Lab) with jupytext installed, paired files are kept in sync automatically on save, and the `.py` files can be opened directly as notebooks.

## ⚠️ Disclaimer

The name *ngsxditto* is inspired by the general concept of smooth, shape-shifting geometries.  
This project is **not affiliated with, endorsed by, or associated with Nintendo, Game Freak, or the Pokémon franchise**.  
The term “ditto” is used in a mathematical and descriptive context only.

All artwork and visualizations are original and generative in nature.  
No copyrighted Pokémon imagery or characters are used.

## Contributors:

Paul Schwering (Narrow band transport)