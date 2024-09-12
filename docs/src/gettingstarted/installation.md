
# Installation

### Short Version

These short instructions are intended for users who are already familiar with Julia. 
If these instructions don't make sense please see the detailed instructions below. 

1. Install Julia (1.10) if you haven't already. Make sure the `General` registry is installed and up to date. 

2. Setup a new project: create a folder to develop your new project, and `cd` into the folder. This folder will track the packages and versions which the `ACEpotentials` code requires. Start julia, activate the project and add `ACEregistry` that includes `ACEpotentials`, which is the package that we want to install:

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.Registry.add("General")  # only needed when installing Julia for the first time
   Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
   Pkg.add("ACEpotentials")
   ```

   `ACEpotentials` will come with the most important packages you need, in particular `ACEfit.jl` and various `AtomsBase.jl` related packages. 

3. You need to activate a project folder when starting julia.
This can be done by starting julia with `julia --project=pathtoproject` command,
using an environment variable `export JULIA_PROJECT=pathtoproject` or, after starting julia, calling
```julia
using Pkg
pkg"activate pathtoproject"
```


### Detailed Instructions

If you have any difficulties with the following setup process, please file an issue. We highly recommend familiarizing oneself with the [Julia package manager](https://github.com/JuliaLang/Pkg.jl) and how Project management is best done in Julia (there is also a summary in [this section](pkg.md) of these docs). In particular all projects should manage their own `Project.toml` file with appropriate version bounds, and where appropriate the `Manifest.toml` file can be tracked in order to guarantee reproducibility of results.

#### Installing Julia

Download and unpack [Julia](https://julialang.org). We require v1.10 or upwards. Add the `julia` executable to your path with something like `export PATH=<julia-directory>/bin:$PATH`.

Start the Julia REPL (type `julia` followed by Enter), switch to package manager by typing `]`, then install the General registry and the [`ACE` registry](https://github.com/ACEsuit/ACEregistry):
```julia
registry add https://github.com/JuliaRegistries/General
registry add https://github.com/ACEsuit/ACEregistry
```
Press Backspace or `Ctrl-c` to exit the package manager. Use `Ctrl-d`, or `exit()` followed by Enter, to close the Julia REPL.

#### Setting up a new `ACEpotentials.jl` project

Create a folder for your new project and change to it. Start the Julia REPL and activate a new project by switching to the package manager with `]`, and then running
```julia 
activate .
```
Now you can install `ACEpotentials`. Remaining in the package manager, use
```julia
add ACEpotentials
```

Depending on your usage you may also need to add other packages, e.g. `AtomsBase`, `Molly`, `DFTK` etc.


#### Returning to a project

When returning to a project, there are several methods for reactivating it. One is to simply `activate .` in the package manager, as above. Alternatively set the `JULIA_PROJECT` environment variable to the directory with `Project.toml` before starting julia, or call julia as `julia --project=<dir>`. Special syntax like `JULIA_PROJECT=@.` or `julia --project=@.` searches the current directory and its parents for a `Project.toml` file.


### Setting up the Python ASE calculator

!!! warning 
    The current version of ACEpotentials does not have a tested ASE interface. If you need an ASE interface, consider using a version 
    < 0.8 of ACEpotentials.



We use a wrapper called `pyjulip` to call julia and evaluate ACE potentials. In a terminal, with the correct julia project and python environment selected, run the following code:

```
python -m pip install julia
python -c "import julia; julia.install()"
```

Make sure to use the correct python and pip, e.g. the ones that are in the correct Conda environment.
Then, to set up `pyjulip`:

```
git clone https://github.com/casv2/pyjulip.git
cd pyjulip
pip install .
``` 


### Troubleshooting

There are currently no known recurring problems with `ACEpotentials` installation.
