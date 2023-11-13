# AtomsBase Interface

ACEpotentials has a new [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) based interface in addition to [JuLIP](https://github.com/JuliaMolSim/JuLIP.jl) based interface.

AtomsBase allows easy communication between different Julia programs. In the future this interface will became the default one and JuLIP interface will be retired.

AtomsBase interface has not been rigorously tested yet. So, expect some issues here and there. But you are recommended to give it a try.

## Loading Training Data

With AtomsBase you can use [AtomsIO](https://github.com/mfherbst/AtomsIO.jl) to load training data

```julia
using AtomsIO

data = load_trajectory("path to training data")
```

You can also use ExtXYZ directly (exported by ACEpotentials) to load training data

```julia
data = ExtXYZ.load("path to training data")
```

To use AtomsBase with the examples you can load the training data in AtomsBase format by giving a keyword argument `atoms_base=true`

```julia
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial"; atoms_base=true)
```

## Training with AtomsBase

There are no changes to the training methods when using AtomsBase. You only need to have the training data in AtomsBase format for the training to work. Here is the `acemodel` style training tutorial using AtomsBase

```julia
using ACEpotentials


data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial"; atoms_base=true)

model = acemodel(
    elements = [:Ti, :Al],
	order = 3,
	totaldegree = 6,
	rcut = 5.5,
	Eref = [:Ti => -1586.0195, :Al => -105.5954]
)
@show length(model.basis);


weights = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 )
);

solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
data_train = data[1:5:end]
P = smoothness_prior(model; p = 4) 

acefit!(model, data_train; solver=solver, weights=weights, prior = P);

@info("Training Error Table")
ACEpotentials.linear_errors(data_train, model; weights=weights);
```

### Weights with AtomsBase

With AtomsBase weights are stored in AtomsBase structures and there is no need to create `AtomsData` structures. Weights can be given for either energy, forces or virial and for structure itself. Each of these have an associated keyword

- `:energy_weight` for energy
- `:force_weight` for forces
- `virial_weight` for virial
- `:weight` a general weight for the structure that multiply other weights

To access the weights you can call

```julia
data_point[:energy_weight]
data_point[:force_weight]
data_point[:virial_weight] 
data_point[:weight]
```

To set a weight by hand on an individual structure you can use

```julia
# Set general weight
data_point[:weight] = 60

# Set weight for energy
data_point[:energy_weight] = 60

# etc.
```

When you use acemodel interface (call `acefit!`) the weights are applied by overwriting any existing weights.

You can look for what keys AtomsBase structures support by calling

```julia
# whole structure features
keys(data_point)

# features per atom
atomkeys(data_point)
```

You can give keyword `group_key` to `acefit!` to determine what group weights are used. Make sure that corresponding `haskey` call returns true.

### New Assemble Backend

AtomsBase interface uses new assemble backend that is faster than the old one. You can you the new backend with old JuLIP interface by giving keyword `new_assembly=true`. E.g.

```julia
# acefit interface
acefit!(model, data_train; solver=solver, weights=weights, prior = P, new_assembly=true);

# acebasis interface
ACEfit.assemble(data_train, basis; new_assembly=true)
```

## Calculations with AtomsBase

To use AtomsBase structures as an input for calculations you need to create `ACEpotential` structure. If you use acemodel interface (like above) you can do this with

```julia
pot = ACEpotential(model)
```

If you use acebasis interface you can create potential with

```julia
pot_1 = ACEpotential(basis, results["C"])
```

After this you can use commands

```julia
E = ace_energy(system, pot)
F = ace_forces(system, pot)
V = ace_virial(system, pot)
```

where `system` is an AtomsBase structure.

For more details on using AtomsBase look [ACEmd](https://github.com/ACEsuit/ACEmd.jl).
