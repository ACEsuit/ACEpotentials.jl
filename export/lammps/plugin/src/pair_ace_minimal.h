/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_ace_minimal.h - ACE potential plugin using minimal export approach

   This pair style loads ACE potential models exported using the minimal
   export approach from ACEpotentials.jl. Models are loaded at runtime
   via Julia's C API and the ACE_C_Interface_Minimal module.

   Key differences from pair_ace.h:
   - Uses Julia C API instead of dlopen
   - Loads model directories (not .so files)
   - Calls ACE_C_Interface_Minimal functions
   - Requires libjulia in LD_LIBRARY_PATH

   Usage:
     pair_style ace/minimal
     pair_coeff * * /path/to/model/ Si O ...

   Where /path/to/model/ is a directory containing:
     - ace_model.jl (wrapper module)
     - ace_model_data.jls (serialized model)
     - Project.toml (dependencies)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ace/minimal,PairACEMinimal);
// clang-format on
#else

#ifndef LMP_PAIR_ACE_MINIMAL_H
#define LMP_PAIR_ACE_MINIMAL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairACEMinimal : public Pair {
 public:
  PairACEMinimal(class LAMMPS *);
  ~PairACEMinimal() override;

  // Required Pair interface
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  // Julia runtime state
  static bool julia_initialized;
  static bool julia_finalized;

  // Model ID from ACE_C_Interface_Minimal
  int model_id;

  // Model information (queried from Julia)
  double cutoff;               // Cutoff radius from model
  int n_species;               // Number of species in model
  int *species_Z;              // Atomic numbers supported by model [n_species]

  // Element mapping
  int *type_to_Z;              // LAMMPS type -> atomic number (-1 if not mapped)
  char **element_names;        // Element names from pair_coeff

  // Work arrays for per-atom evaluation
  int maxneigh;                // Current allocation size
  int *neighbor_Z;             // Neighbor atomic numbers [maxneigh]
  double *neighbor_Rij;        // Neighbor displacement vectors [maxneigh*3]
  double *site_forces;         // Site force output [maxneigh*3]

  // Helper methods
  void allocate();
  void load_model(const char *model_dir);
  void unload_model();
  void init_julia();
  int element_to_Z(const char *elem);
  void grow_neighbor_arrays(int n);
};

}    // namespace LAMMPS_NS

#endif
#endif
