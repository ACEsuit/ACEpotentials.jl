
# Installation

### Short Version

These short instructions are intended for users who are already familiar with Julia. 
If these instructions don't make sense please see the detailed instructions below. 

1. Install Julia if you haven't already. Make sure the `General` registry is installed and up to date. 

2. Setup a new project: create a folder to develop your new project, and `cd` into the folder. This folder will track the packages and versions which the `ACEpotentials` code requires. Start julia, activate the project and add `ACEregistry` that includes `ACEpotentials`, which is the package that we want to install:

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.Registry.add("General")  # only needed when installing Julia for the first time
   Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
   Pkg.add("ACEpotentials")
   ```

   `ACEpotentials` will come with the most important packages you need, in particular `ACE1.jl` and `ACEfit.jl`.

3. You need to activate the project folder when starting julia.
This can be done by starting julia with `julia --project=pathtoproject` command,
using environment variable `export JULIA_PROJECT=pathtoproject` or by after starting julia calling

```julia
using Pkg
pkg"activate pathtoproject"
```

### Setting up the Python ASE calculator

!!! warning
    At present, it is necessary to have `ASE`, `JuLIP` and `ACE1` installed in your Julia project to use `pyjulip`.


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


### Detailed Instructions

If you have any difficulties with the following setup process, please file an issue. We highly recommend familiarizing oneself with the [Julia package manager](https://github.com/JuliaLang/Pkg.jl) and how Project management is best done in Julia (there is also a summary in [this section](pkg.md) of these docs). In particular all projects should manage their own `Project.toml` file with appropriate version bounds, and where appropriate the `Manifest.toml` file can be tracked in order to guarantee reproducibility of results.

#### Installing Julia

Download and unpack [Julia](https://julialang.org). We require v1.9 or upwards. Add the `julia` executable to your path with something like `export PATH=<julia-directory>/bin:$PATH`.

Start the Julia REPL (type `julia` followed by Enter), switch to package manager by typing `]`, then install the General registry and the [`ACE` registry](https://github.com/ACEsuit/ACEregistry):
```julia
registry add https://github.com/JuliaRegistries/General
registry add https://github.com/ACEsuit/ACEregistry
```
Press Backspace or `Ctrl-c` to exit the package manager. Use `Ctrl-d`, or `exit()` followed by Enter, to close the Julia REPL.

#### Setting up a new `ACE1.jl` project

Create a folder for your new project and change to it. Start the Julia REPL and activate a new project by switching to the package manager with `]`, and then running
```julia 
activate .
```
Now you can install `ACEpotentials`. Remaining in the package manager, use
```julia
add ACEpotentials
```

Depending on your usage you may also need to add other packages. 

```@raw html
<!-- the following packages: `ACE1, JuLIP, ACEfit, ASE`. -->
```

#### Returning to a project

When returning to a project, there are several methods for reactivating it. One is to simply `activate .` in the package manager, as above. Alternatively set the `JULIA_PROJECT` environment variable to the directory with `Project.toml` before starting julia, or call julia as `julia --project=<dir>`. Special syntax like `JULIA_PROJECT=@.` or `julia --project=@.` searches the current directory and its parents for a `Project.toml` file.

### Trouble-shooting

* On some systems `ASE.jl` and `ACEfit.jl` is unable to automatically install python dependencies. We found that installing [Anaconda](https://anaconda.org) and then pointing `PyCall.jl` to the Anaconda installation (cf [PyCall Readme](https://github.com/JuliaPy/PyCall.jl)) resolves this. After installing Anaconda, it should then be sufficient to build `ASE.jl` again.
* If you cannot use Anaconda python, or if the last point failed, then you can try to install the python dependencies manually before trying to build `ASE.jl` again. Specifically, it should be sufficient to just install the [ase](https://wiki.fysik.dtu.dk/ase/) package. Please follow the installation instructions on their website.
