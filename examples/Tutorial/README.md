This tutorial was originall developed by W Chuck Witt for the 2023 GAP/(M)ACE User and Developer Meeting.

To install the tutorial
- install Julia 1.10
- create a new folder
- active a new Julia project in this folder 
- add the `ACEpotentials` package to the folder 
- From the Julia repl run `using ACEpotentials; ACEpotentials.install_tutorial(@__DIR__())`; this will install `Literate` and `IJulia`, copy the notebook to the current folder in Literate format, then convert it to Jupyter format. 
- Exit Julia, open Jupyter, open the `ACEpotentials-Tutorial.ipynb` notebook

