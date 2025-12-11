/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_ace.h - ACE potential plugin for LAMMPS

   This pair style loads ACE potential models compiled from ACEpotentials.jl
   using Julia's juliac --trim feature. Models are dynamically loaded at
   runtime via dlopen.

   Usage:
     pair_style ace
     pair_coeff * * /path/to/model.so Si O ...
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ace,PairACE);
// clang-format on
#else

#ifndef LMP_PAIR_ACE_H
#define LMP_PAIR_ACE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairACE : public Pair {
 public:
  PairACE(class LAMMPS *);
  ~PairACE() override;

  // Required Pair interface
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  // Dynamic library handle for loaded model
  void *model_handle;

  // Function pointer types for ACE C API
  typedef double (*ace_site_efv_fn)(int, int, int*, double*, double*, double*);
  typedef double (*ace_get_cutoff_fn)();
  typedef int (*ace_get_n_species_fn)();
  typedef int (*ace_get_species_fn)(int);

  // Function pointers (resolved from loaded model)
  ace_site_efv_fn ace_site_energy_forces_virial;
  ace_get_cutoff_fn ace_get_cutoff;
  ace_get_n_species_fn ace_get_n_species;
  ace_get_species_fn ace_get_species;

  // Model information
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
  double site_virial[6];       // Site virial output [6]

  // Helper methods
  void allocate();
  void load_model(const char *filename);
  void unload_model();
  int element_to_Z(const char *elem);
};

}    // namespace LAMMPS_NS

#endif
#endif
