# python-`ase`

An `ACEpotentials.jl` model can be used in python as an `ase` calculator. To do this, you will need to install two python packages, `julia` and `pyjulip` as documented on the [installation page](../gettingstarted/installation.md). 

Python reads the `potential.json` file directly. To load an ACE potential as an `ase` calculator, use the following syntax:

```python
import pyjulip
calc = pyjulip.ACE1("first_potential.json")
```

Using that calculator, we can then evaluatuate energies, forces, etc, e.g., 
```python
ats = ase.io.read('atoms_object.xyz')
ats.calc = calc
print(ats.get_potential_energy())
```

See the `ase` [documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators) for more details.

### Another option: ASE's LAMMPSlib calculator

Alternatively, to avoid direct Julia-Python interaction, one can export to LAMMPS (see [LAMMPS](lammps.md)) and utilize ASE's [`LAMMPSlib` calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/lammpslib.html).
