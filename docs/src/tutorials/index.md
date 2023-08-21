
# Tutorials Overview 

### Fitting potentials using the Julia interface

These tutorials use the direct Julia interface provided by `ACE1.jl, ACE1x.jl, ACEfit.jl, ACEpotentials.jl`. They are provided in [Literate.jl](https://github.com/fredrikekre/Literate.jl) format in `ACEpotentials/tutorials/`.

* [First Example - Model Interface](../literate_tutorials/first_example_model.md)
* [TiAl Potential - Model Interface](../literate_tutorials/TiAl_model.md)
* [TiAl Potential - Basis Interface](../literate_tutorials/TiAl_basis.md)

### Fitting potentials from the command line

ACE potentials can be fitted from the command line using a dictionary stored in a `.json` or `.yaml` file to specify the parameters:

* [TiAl Potential (command line JSON)](./first_example_json.md)

### Fitting potentials using the ACEpotentials JSON interface from Julia

This tutorial shows how to use ACEpotentials to fit ACE potentials, using the same dictionary structure:

* [TiAl Potential (ACEpotentials)](../literate_tutorials/ACEpotentials_TiAl.md)


### Using ACE1 Potentials in External Software

* [ACE potentials in LAMMPS](lammps.md)
* [ACE potentials in python with `ase`](python_ase.md)

### Structure analysis with ACE1 descriptors

* [ACE descriptors](../literate_tutorials/descriptor.md)

### Committee Potentials

* [Committee Potentials](../literate_tutorials/committee.md)
