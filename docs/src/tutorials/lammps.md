# LAMMPS

An ACE1 potential can be used LAMMPS via the ML-PACE LAMMPS package. General installation details from LAMMPS are [here](https://docs.lammps.org/Build_extras.html#ml-pace) and [here](https://github.com/ICAMS/lammps-user-pace). However, at the moment, one must install LAMMPS as follows:
```
git clone https://github.com/lammps/lammps
cd lammps
mkdir build
cd build
wget -O libpace.tar.gz https://github.com/wcwitt/lammps-user-pace/archive/main.tar.gz
cmake \
    -D BUILD_SHARED_LIBS=yes \
    -D LAMMPS_EXCEPTIONS=yes \
    -D PKG_ML-PACE=yes \
    -D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}') \
    ../cmake
make -j 4
```


## Export the model to the `'yace` format

To run an ACE potential in LAMMPS, you can export a potential in the `.yace` format. When fitting from julia, the potential must be exported by doing (see [this example](../literate_tutorials/TiAl.md)):

```julia
ACE1pack.ExportMulti.export_ACE("potenial.yace", IP)
```

If you have fitted from command line using the `aec_fit.jl` script, the `.yace` potential file will be made automatically.

## The PACE pair style

The syntax for the PACE pair style in LAMMPS, for a potential for I, Cs and Pb, is:

```
pair_style      pace
pair_coeff      * * potential.yace I Cs Pb
```
The order of the species after `pair_coef` must be the numerical ordering in the `.data` geometry file. 

### Notes

### 1. Exporting the pair potential via a spline lookup table

The ACE potential has a two-body component and a many body component. There is the option to export the two-body component as a spline lookup table which LAMMPs reads directly. To do this, include `export_pairpot_as_table=true` when calling `export_ACE`.

This creates a many body `potential.yace` file, and a two-body `potential_pairpot.table` file. The `.table` file contains a set of lookup tables with a fixed number `N` (written in the file) of interpolation points. To use this in LAMMPS, read N from the file and use the syntax:

```
pair_style      hybrid/overlay pace table linear <N>
pair_coeff      * * pace potential.yace I Cs Pb
pair_coeff      1 1 table potential_pairpot.table I_I
pair_coeff      1 2 table potential_pairpot.table I_Cs
pair_coeff      1 3 table potential_pairpot.table I_Pb
pair_coeff      2 2 table potential_pairpot.table Cs_Cs
pair_coeff      2 3 table potential_pairpot.table Cs_Pb
pair_coeff      3 3 table potential_pairpot.table Pb_Pb
```

where we are using the ordering I, Cs, Pb.

### 2. Calling LAMMPS from python

Calling LAMMPS from python to evaluate an ACE potential is not recommended, but can be done. When calling LAMMPS pace from python, the species must be specified in alphabetical order in the `pair_coef` command. This is because python does not expect the species to appear as a string literal in the `pair_coef`, which would be specied like this in python:

```python
parameters = {'pair_style': 'pace',
             'pair_coeff': ['* * potential.yace I Cs Pb']}

files = ["potential.yace"]

calc1 = LAMMPS(parameters=parameters, files=files)
```

When python calls lammps, it makes a .data file in which the numeric atom types correspond to alphabetically ordered species by chemical symbol. To make python agree, the species must therefore be in alphabetical order (Cs, I, Pb)
