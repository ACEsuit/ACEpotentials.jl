# Using ACE potentials in Molly

[Molly](https://github.com/JuliaMolSim/Molly.jl) is pure Julia MD program that is in development.
ACE support for Molly is currently in [ACEmd](https://github.com/ACEsuit/ACEmd.jl) package,
which is exported by ACEpotentials. 


## Things to know about Molly

Molly expects units to be defined. Our fitting procedure does not define units,
so in order to use Molly, units need to be defined. This is done by wrapping potentials to
a structure that holds units in addition to the potential. The units used are defined in [Unitful](https://github.com/PainterQubits/Unitful.jl), which is exported by default.

To wrap units for a potential you can use `load_ace_model` function, which can take in
a potential you have just fitted as an input. You can also load `json` or `yace` potential
files.

```julia
pot_with_units = load_ace_model( potential_with_no_units )

# Load potential files
pot_with_units = load_ace_model( "path to potential file" )
```

The default units are eV for energy and Å for length. You can change these with

```julia
load_ace_model( "path to potential file";
                energy_unit = u"hartree",
                length_unit = u"bohr",
                cutoff_unit = u"pm" )
```

## System setup

To start Molly you need to prepare Molly system. With ACE there are still some ACE specific complications with this. But please refer for Molly documentation 

```julia
using Molly
using ACEpotentials # or ACEmd
using ExtXYZ

# Load initial structure
data = FastSystem(ExtXYZ.Atoms(read_frame("initial structure in xyz file")))
# or use whatever AtomsBase structure

# Load ACE potential
pot = load_ace_model("some ace potential")

# Prepare data to Molly compatible format
atoms = [Molly.Atom( index=i, mass=AtomsBase.atomic_mass(data, i) ) for i in 1:length(data) ]

# Create boundary conditions
boundary = begin
    box = bounding_box(data)
    CubicBoundary(box[1][1], box[2][2], box[3][3])
end

# Prepare atomic number data. This is an ACE specific customization for older Molly versions.
atom_data = [ (; :Z=>z,:element=>s)  for (z,s) in zip(AtomsBase.atomic_number(data), AtomsBase.atomic_symbol(data))  ]

# Set up temperature and velocities
temp = 298.0u"K"
velocities = [random_velocity(m, temp) for m in atomic_mass(data)]

# Set up Molly system it self
sys = System(
           atoms=atoms,
           atoms_data = atom_data,
           coords=position(data),
           velocities=velocities,
           general_inters = (pot,),
           boundary=boundary,
           loggers=(temp=TemperatureLogger(100),), # add more loggers here
           energy_units=u"eV",  # Molly simulation units
           force_units=u"eV/Å",
       )
```

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
