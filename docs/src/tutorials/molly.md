# Using ACE potentials in Molly

[Molly](https://github.com/JuliaMolSim/Molly.jl) is pure Julia MD program that is in development.
ACE support for Molly is currently in [ACEmd](https://github.com/ACEsuit/ACEmd.jl) package,
which is exported by ACEpotentials.

## Things to know about Molly

Molly expects units to be defined. Our fitting procedure does not define units (implicitly we use eV for energy and Å length),
so in order to use Molly, units need to be defined. This is done by wrapping potentials to
a structure that holds units in addition to the potential. The units used are defined in [Unitful](https://github.com/PainterQubits/Unitful.jl), which is exported by default.

To wrap units for a potential you can use `load_ace_model` function, which can take in
a potential you have just fitted as an input. You can also load `json` or `yace` potential
files exported from `ACEpotentials.jl` or `ACE1.jl`.

```julia
using ACEpotentials

# Load potential from file
potential = load_potential( "path to potential file"; new_format=true )
```

The default units are eV for energy and Å for length. You can change these with

```julia
pot_new_units = ACEpotential(
    pot.potentials;
    energy_unit = u"hartree",
    length_unit = u"bohr",
    cutoff_unit = u"pm"
)
```

## System setup

To start Molly you need to prepare the Molly system. There are still some ACE specific complications with this. But please refer to Molly documentation

```julia
using Molly
using ACEpotentials
using AtomsIO

# Load initial structure
data = AtomsIO.load_system("initial structure file")
# or use whatever AtomsBase structure
# need to have velocity return other than missing

# Load ACE potential
pot = load_potential("some ace potential"; new_format=true)

# Pack data to Molly compatible format
sys = Molly.System(data, pot)

# Set up temperature and velocities
temp = 298.0u"K"
vel = random_velocities!(sys, temp)

# Add loggers
# need at least Molly v0.17 for this
sys = Molly.System(
    sys;
    loggers=(temp=TemperatureLogger(100),) # add more loggers here
)
```

You can also customize system more. For details refer [Molly documentation](https://juliamolsim.github.io/Molly.jl/stable/).

## Set up simulation

To setup Molly simulation you need to create simulation object

```julia
# Set up simulator
simulator = VelocityVerlet(
    dt=1.0u"fs",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)
```

After this you can run the simulation by

```julia
# Perform MD for 1000 steps
simulate!(sys, simulator, 1000)
```
