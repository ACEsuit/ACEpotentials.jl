
# Installation Instructions

### Short Version

First, you will need to install julia (see below for instructions). 

Create a folder to hold your julia `ACE` project, and `cd` into the fodler. This folder will track the packages and versions which the `ACE1pack` code requires. For example:
```
mkdir ~/ACE1project
cd ~/ACE1project
```

From within this folder type `julia` to enter the Julia REPL. Then run
```julia
using Pkg; Pkg.activate("."); pkg"registry add https://github.com/JuliaRegistries/General"; pkg"registry add https://github.com/JuliaMolSim/MolSim.git"; pkg"add ACE1pack, ACE1, JuLIP, IPFitting, ASE"
```

Before working on an ACE1 project in the `ACE1project` folder you will need to activate the Julia environment you just created in that folder. This can be done by starting julia with `julia --project=pathtoproject`, or from the [package manager](pkg.mk), or by exporting the environment variable `JULIA_PROJECT` set to the path to this folder. For example, `export JULIA_PROJECT=~/ACE1project`. 
<!-- You can add this to your `.bashrc`/`.bash_profile` and not touch it again.  -->

### Setting up the Python ASE calculator

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

Download and unpack [Julia](https://julialang.org). We recommend v1.6 or upwards. Add the `julia` executable to your path with something like `export PATH=<julia-directory>/bin:$PATH`.

Start the Julia REPL (type `julia` followed by Enter), switch to package manager by typing `]`, then install the General registry and the [`MolSim` registry](https://github.com/JuliaMolSim/MolSim):
```julia
registry add https://github.com/JuliaRegistries/General
registry add https://github.com/JuliaMolSim/MolSim.git
```
Press Backspace or `Ctrl-c` to exit the package manager. Use `Ctrl-d`, or `exit()` followed by Enter, to close the Julia REPL.

#### Setting up a new `ACE1.jl` project

Create a folder for your new project and change to it. Start the Julia REPL and activate a new project by switching to the package manager with `]`, and then running
```julia 
activate .
```
Now you can install `ACE1pack`. Remaining in the package manager, use
```julia
add ACE1pack
```

you will also need to add the following packages: `ACE1, JuLIP, IPFitting, ASE`.

#### Returning to a project

When returning to a project, there are several methods for reactivating it. One is to simply `activate .` in the package manager, as above. Alternatively set the `JULIA_PROJECT` environment variable to the directory with `Project.toml` before starting julia, or call julia as `julia --project=<dir>`. Special syntax like `JULIA_PROJECT=@.` or `julia --project=@.` searches the current directory and its parents for a `Project.toml` file.

### Trouble-shooting

* On some systems `ASE.jl` (a dependency of `IPFitting.jl`) is unable to automatically install python dependencies. We found that installing [Anaconda](https://anaconda.org) and then pointing `PyCall.jl` to the Anaconda installation (cf [PyCall Readme](https://github.com/JuliaPy/PyCall.jl)) resolves this. After installing Anaconda, it should then be sufficient to build `ASE.jl` again.
* If you cannot use Anaconda python, or if the last point failed, then you can try to install the python dependencies manually before trying to build `ASE.jl` again. Specifically, it should be sufficient to just install the [ase](https://wiki.fysik.dtu.dk/ase/) package. Please follow the installation instructions on their website.
