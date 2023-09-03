
# Tutorials

### Fitting potentials from Julia scripts

These tutorials use the direct Julia interface provided by `ACEpotentials.jl` (interfacing with `ACE1.jl, ACE1x.jl, ACEfit.jl`). They are provided in [Literate.jl](https://github.com/fredrikekre/Literate.jl) format and can also be run as scripts if that is preferred. 

* [First Example](../literate_tutorials/first_example_model.md)
* [Model Interface](../literate_tutorials/TiAl_model.md)
* [Basis Interface](../literate_tutorials/TiAl_basis.md)

The next two tutorials show some additional techniques to better understand how to make good hyperparameter choices. 

* [Smoothness Priors](../literate_tutorials/smoothness_priors.md) : a basic introduction to smoothness priors 
* [Dataset Analysis](../literate_tutorials/dataset_analysis.md) : some basic techniques to visualize training datasets and correlate such observations to the choice of geometric priors


### Using `ACEpotentials` Potentials in External Software

* [LAMMPS](lammps.md)
* [Python with `ase`](python_ase.md)

### Structure analysis with ACE1 descriptors

* [ACE descriptors](../literate_tutorials/descriptor.md)

### Committee Potentials

* [Committee Potentials](../literate_tutorials/committee.md)


```@raw html 
<!---
### JSON Interface and Command line (OUTDATED)

ACE potentials can be fitted from the command line using a dictionary stored in a `.json` or `.yaml` file to specify the parameters:

* [TiAl Potential (ACEpotentials)](../literate_tutorials/ACEpotentials_TiAl.md)
* [TiAl Potential (command line JSON)](../outdated/first_example_json.md)

-->
```
