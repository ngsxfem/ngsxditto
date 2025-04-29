# Installation
When using this library from outside the repo install it first via `pip3 install --user git+https://gitlab.gwdg.de/ngsuite/ngsxditto` or after cloning the repo install it via `python3 setup.py install --user` or `pip3 install . --user`.


# Using / developing
If you are working in the library directly (working on a module, geometry, test or example) we use the hack to add the local goengs files to the python path `import sys`, `sys.path.append('../goengs')`.

# Available (and planned) modules:
  * levelset (WIP): 
    This module is work in progress. Don't use it, yet. 
    Here, several algorithms related to the levelset description of (typically) moving domain problems are gathered. It has several submodules:
    * transport (only preliminary state)...
    * redistancing (not implemented yet)... 
    * mean curvature computation (not implemented yet)...
    * normal extensions: First version in `goengs.module.levelset.normalextension`
    * bulk extensions (not implemented yet)...
  * ...

# Examples
