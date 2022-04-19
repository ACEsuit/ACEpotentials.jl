# LAMMPS interface

The ACE1 potentials can be exported as shown in the TiAl tutorial. Here we will demonstrate how this potential can be used using the LAMMPs `pace pairstyle`. Before using this make sure a LAMMPs executable with `ML-PACE` is compiled (https://github.com/lammps/lammps)

```python
import os
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read
import time
```

For LAMMPs in Python the `ASE_LAMMPSRUN_COMMAND` needs to be set pointing to a LAMMPs built using PACE.
```python
os.environ["ASE_LAMMPSRUN_COMMAND"]="~/gits/lammps/build/lmp"
```

Setting up the LAMMPs calculator using traditional LAMMPs commands.
```python
parameters = {'pair_style': 'pace',
             'pair_coeff': ['* * TiAl_tutorial_pot.yace Ti Al']}

files = ["TiAl_tutorial_pot.yace"]

calc1 = LAMMPS(parameters=parameters, files=files)
```

Using the LAMMPS calculator to evaluate energies/forces/virials from Python
```python
at = read("./TiAl_tutorial_DB.xyz", ":")[0]

at.set_calculator(calc1)

print(at.get_potential_energy())
print(at.get_forces())
print(at.get_stress())
```
