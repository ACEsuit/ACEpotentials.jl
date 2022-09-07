# python-`ase`

An ACE1 potential can be used in python as an `ase` calculator. To do this, you will need to install two python packages, `julia` and `pyjulip` as documented on the [installation page](../gettingstarted/installation.md). 

Python reads the `potential.json` file directly. To load and use an ACE potential, use the following syntax:

```python
import pyjulip
calc = pyjulip.ACE1("first_potential.json")
```

evaluation is then:
```python
ats = ase.io.read('atoms_object.xyz')
ats.calc = calc
print(ats.get_potential_energy())
```

See the `ase` [documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators) for more details. 
