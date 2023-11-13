# Saving and Loading Potentials

## Saving Potentials for Julia use

To save potentials for future Julia use can use

```julia
save_potential("my-potential-file.json", potential)
```

This will save the potential in `json` format. You can also use `yml` and `yace` suffixes.

The format used for saving can be either ACEmodel from `acemodel` function, `JuLIP` style potentials or `ACEmd` style `ACEpotential`.

## Loading Potentials

To load potential use

```julia
potential = load_potential("my-potential-file.json")
```

By default this should print information about versions in use when the potential was saved. E.g. like following

```txt
This potential was saved with following versions:

JuLIP v0.14.5
ACEbase v0.4.3
ACE1x v0.1.8
ACE1 v0.11.16
ACEmd v0.1.7
ACEpotentials v0.6.3
ACEfit v0.1.4

If you have problems using this potential, pin your installation to above versions.
```

If you have problems with the potential, you can use the given version numbers to build an installation that should have the potential working.

By default the loaded potential is in JuLIP style format. To load a new `ACEmd` style `ACEpotential` you can give keyword `new_format=true`

```julia
potential = load_potential("my-potential-file.json"; new_format=true)
```

## Exporting Potentials for Other Programs

LAMMPS export is described in section [ACEpotentials potentials in LAMMPS](@ref).