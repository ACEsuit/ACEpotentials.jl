# ACE1pack potentials in LAMMPS

### Install LAMMPS with the ML-PACE package

An ACE1pack potential can be used in LAMMPS if the latter is built with the ML-PACE package. However, at present, a patched version of that package is required and one must install LAMMPS as follows:
```
git clone -b stable https://github.com/lammps/lammps
cd lammps
mkdir build
cd build
wget -O libpace.tar.gz https://github.com/wcwitt/lammps-user-pace/archive/main.tar.gz
cmake \
    -D BUILD_MPI=yes \
    -D BUILD_SHARED_LIBS=yes \
    -D LAMMPS_EXCEPTIONS=yes \
    -D PKG_ML-PACE=yes \
    -D PACELIB_MD5=$(md5sum libpace.tar.gz | awk '{print $1}') \
    ../cmake
make -j 4
```
The `BUILD_SHARED_LIBS` and `LAMMPS_EXCEPTIONS` options allow use of LAMMPS from Python.

### Convert an ACE1pack model to `yace` format

The ML-PACE package requires a potential in the `.yace` format. To convert a model saved as `.json` use the following:

```julia
using ACE1pack
potential_json = "Si.json"    # example json filename
potential_yace = "Si.yace"    # example yace filename
export2lammps(potential_yace, read_dict(load_dict(potential_json)["IP"]))
```

### Using `yace` potentials in LAMMPS

The syntax for the PACE pair style in LAMMPS, for a potential for I, Cs and Pb, is:
```
pair_style      pace
pair_coeff      * * potential.yace I Cs Pb
```
The species ordering after `pair_coeff` must match the numerical ordering in any `.data` geometry file. 

By default, ACE1pack models have separate two-body and many-body components.
At present, the two-body component is exported via a lookup table which LAMMPs reads directly, meaning two files are created: a `potentialname_pairpot.table` file for the two-body contribution and a `potentialname.yace` file for the many-body contribution.
The `.table` file contains a set of lookup tables with a fixed number `N` (written in the file) of interpolation points. To use the full model in LAMMPS, read N from the file and use the syntax:
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
