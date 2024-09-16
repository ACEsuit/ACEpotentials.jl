This tutorial was originall developed by W Chuck Witt for the 2023 GAP/(M)ACE User and Developer Meeting.

To install the tutorial
- install Julia 1.10
- create a new folder
- active a new Julia project in this folder, add `ACEpotentials` and run `using ACEpotentials; ACEpotentials.copy_tutorial()`. Or copy-paste the following into a terminal 
```
j110 --project=. -e 'using Pkg; Pkg.add("ACEpotentials"); using ACEpotentials; ACEpotentials.copy_tutorial(@__DIR__())'
```
this will install `Literate` and `IJulia`, copy the notebook to the current folder in Literate format, then convert it to Jupyter format. 
- Exit Julia, open Jupyter (e.g. `jupyter notebook`), open the `ACEpotentials-Tutorial.ipynb` notebook, start reading...

