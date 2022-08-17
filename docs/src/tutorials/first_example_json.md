# TiAl potential (command line JSON)

In this tutorial we will fit a TiAl potential using the ACE fitting script of ACE1pack and a JSON file with all the mandatory and non-default parameters. The fitting itself is simply

```bash
julia .../ACE1pack.jl/scripts/ace_fit.jl --params ace_params.json
```

For this example, the parameters file contains the following. For more details see section on "Command line interface" and "ACE1pack Internals". 

```json
{
    "ACE_fname": "ACE.json",
    "e0": {
        "Ti": -1586.0195,
        "Al": -105.5954},
    "weights": {
        "FLD_TiAl": {
            "V": 1.0,
            "E": 30.0,
            "F": 1.0},
        "TiAl_T5000": {
            "V": 1.0,
            "E": 5.0,
            "F": 1.0}
    },
    "P": {"type": "laplacian"},
    "data": {
        "force_key": "force",
        "energy_key": "energy",
        "fname": "TiAl_tutorial.xyz",
        "virial_key": "virial"},
    "solver": {
        "lsqr_damp": 0.01,
        "type": "lsqr"},
    "basis": {
        "main_ace": {
            "N": 3,
            "maxdeg": 6,
            "radial": {
                "rcut": 5.5,
                "rin": 1.728},
            "type": "ace",
            "transform": {
                "r0": 2.88,
                "type": "polynomial"},
            "species": ["Ti", "Al"]},
        "main_pair": {
            "rcut": 7.0,
            "rin": 0.0,
            "maxdeg": 6,
            "type": "pair",
            "transform": {
                "r0": 2.88,
                "type": "polynomial"},
            "species": ["Ti", "Al"]
        }
    }
}
```

Brief explanation of the main entries:

* `ACE_fname` - filename for the fitted potential. The script will produce a file called `ACE.json` which can be read by julia and python, and a file called `ACE.yace` which can be read by lammps. 
* `e0` - isolated atom energies
* `weights` - weights for the loss function, for the specific structures. The labels (e.g. "TiAl_T5000") correspond to the "config_type" entry in the .xyz's "info" field. 
* `P` - specifies (a part of) the regularisation (optional).  
* `data` - `.xyz` filename with atomic structures and the specific energy/force/virial entries to fit to. 
* `solver` - specifies the LSQR solver
* `basis 
   - "main_ace" - defines the ACE basis
   - "main_pair" - defines the pair basis