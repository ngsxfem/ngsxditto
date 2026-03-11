# ngsxditto

**ngsxditto** is a high-level library for solving PDEs on moving domains using level-set-based methods and unfitted finite element techniques. Inspired by the flexibility and smooth boundaries of shape-shifting forms, the library emphasizes robust numerical schemes, including ghost penalty stabilization, to handle evolving interfaces.

> *"Like its namesake, ngsxditto smoothly adapts to evolving shapes — with mathematical rigor."*

## 🚀 Features

- Level-set based interface capturing
- Geometrically unfitted FEM (CutFEM/XFEM-style)
- Stabilization via ghost penalty and aggregation techniques
- Modular design for extension to multi-physics problems
- Compatible with [ngsolve](https://ngsolve.org)

Features aimed at:
- higher order methods (in geometry and field approximation) 
- and space-time methods 


## Installation without cloning
When using this library from outside the repo install it first via `pip3 install --user git+https://gitlab.gwdg.de/ngsuite/ngsxditto`.

Or install from the package registry:
`pip install ngsxditto --index-url https://__token__:<your_personal_token>@gitlab.gwdg.de/api/v4/projects/45482/packages/pypi/simple` with your personal access token put in here.

## Installation from repo
The first two commands are optional.
However, we assume you have a python installation available.
  * `python3 -m venv .venv`
  * `source .venv/bin/activate`
  * `pip3 install . --usre` (uses the `pyproject.toml` to install the source code)

With the following command, you can verify the installation works
  * `pytest tests/test_*.py`

## Using / developing
If you are working in the library directly (working on a module, geometry, test or example) you can use `pip install -e .` so that the installation only links to your source files and source file changes have immediate impact.

## Available (and planned) modules:
  * levelset (WIP): 
    This module is work in progress. Don't use it, yet. 
    Here, several algorithms related to the levelset description of (typically) moving domain problems are gathered. It has several submodules:
    * transport (only preliminary state)...
    * redistancing (preliminary state)...
  * extensions:
    * normal extensions: First version in `goengs.module.levelset.normalextension`
    * bulk extensions (not implemented yet)...
  * mean curvature computation (not implemented yet)...
  * (unfitted) Stokes (not implement yet)

## Examples

see in the `examples` directory

## ⚠️ Disclaimer

The name *ngsxditto* is inspired by the general concept of smooth, shape-shifting geometries.  
This project is **not affiliated with, endorsed by, or associated with Nintendo, Game Freak, or the Pokémon franchise**.  
The term “ditto” is used in a mathematical and descriptive context only.

All artwork and visualizations are original and generative in nature.  
No copyrighted Pokémon imagery or characters are used.

## Contributors:

Paul Schwering (Narrow band transport)