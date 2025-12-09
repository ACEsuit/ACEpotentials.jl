/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_ace.cpp - ACE potential plugin for LAMMPS

   This pair style loads ACE potential models compiled from ACEpotentials.jl
   using Julia's juliac --trim feature. Models are dynamically loaded at
   runtime via dlopen.

   Usage:
     pair_style ace
     pair_coeff * * /path/to/model.so Si O ...
------------------------------------------------------------------------- */

#include "pair_ace.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <dlfcn.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairACE::PairACE(LAMMPS *lmp) : Pair(lmp)
{
  // Single cutoff for all interactions
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  // Initialize pointers
  model_handle = nullptr;
  ace_site_energy_forces_virial = nullptr;
  ace_get_cutoff = nullptr;
  ace_get_n_species = nullptr;
  ace_get_species = nullptr;

  cutoff = 0.0;
  n_species = 0;
  species_Z = nullptr;
  type_to_Z = nullptr;
  element_names = nullptr;

  maxneigh = 0;
  neighbor_Z = nullptr;
  neighbor_Rij = nullptr;
  site_forces = nullptr;
}

/* ---------------------------------------------------------------------- */

PairACE::~PairACE()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }

  unload_model();

  memory->destroy(species_Z);
  memory->destroy(type_to_Z);

  if (element_names) {
    for (int i = 0; i < atom->ntypes + 1; i++)
      delete[] element_names[i];
    delete[] element_names;
  }

  memory->destroy(neighbor_Z);
  memory->destroy(neighbor_Rij);
  memory->destroy(site_forces);
}

/* ---------------------------------------------------------------------- */

void PairACE::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  // Initialize setflag to 0
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
}

/* ----------------------------------------------------------------------
   Load ACE model from shared library
------------------------------------------------------------------------- */

void PairACE::load_model(const char *filename)
{
  // Unload any existing model
  unload_model();

  // Load the shared library
  model_handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (!model_handle) {
    char errmsg[1024];
    snprintf(errmsg, sizeof(errmsg), "Cannot load ACE model '%s': %s",
             filename, dlerror());
    error->all(FLERR, errmsg);
  }

  // Clear any existing error
  dlerror();

  // Resolve required symbols
  ace_site_energy_forces_virial = (ace_site_efv_fn)dlsym(model_handle,
      "ace_site_energy_forces_virial");
  const char *err = dlerror();
  if (err) {
    char errmsg[1024];
    snprintf(errmsg, sizeof(errmsg),
             "Cannot find ace_site_energy_forces_virial in model: %s", err);
    error->all(FLERR, errmsg);
  }

  ace_get_cutoff = (ace_get_cutoff_fn)dlsym(model_handle, "ace_get_cutoff");
  err = dlerror();
  if (err) {
    char errmsg[1024];
    snprintf(errmsg, sizeof(errmsg),
             "Cannot find ace_get_cutoff in model: %s", err);
    error->all(FLERR, errmsg);
  }

  ace_get_n_species = (ace_get_n_species_fn)dlsym(model_handle,
      "ace_get_n_species");
  err = dlerror();
  if (err) {
    char errmsg[1024];
    snprintf(errmsg, sizeof(errmsg),
             "Cannot find ace_get_n_species in model: %s", err);
    error->all(FLERR, errmsg);
  }

  ace_get_species = (ace_get_species_fn)dlsym(model_handle, "ace_get_species");
  err = dlerror();
  if (err) {
    char errmsg[1024];
    snprintf(errmsg, sizeof(errmsg),
             "Cannot find ace_get_species in model: %s", err);
    error->all(FLERR, errmsg);
  }

  // Get model parameters
  cutoff = ace_get_cutoff();
  n_species = ace_get_n_species();

  // Get species list (atomic numbers)
  memory->destroy(species_Z);
  memory->create(species_Z, n_species, "pair:species_Z");
  for (int i = 0; i < n_species; i++) {
    species_Z[i] = ace_get_species(i + 1);  // 1-based indexing in Julia
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "ACE: Loaded model from {}\n", filename);
    utils::logmesg(lmp, "ACE: Cutoff = {} Angstrom\n", cutoff);
    utils::logmesg(lmp, "ACE: Number of species = {}\n", n_species);
    std::string species_str = "ACE: Supported elements (Z):";
    for (int i = 0; i < n_species; i++) {
      species_str += " " + std::to_string(species_Z[i]);
    }
    utils::logmesg(lmp, "{}\n", species_str);
  }
}

/* ----------------------------------------------------------------------
   Unload model shared library
------------------------------------------------------------------------- */

void PairACE::unload_model()
{
  if (model_handle) {
    dlclose(model_handle);
    model_handle = nullptr;
  }
  ace_site_energy_forces_virial = nullptr;
  ace_get_cutoff = nullptr;
  ace_get_n_species = nullptr;
  ace_get_species = nullptr;
}

/* ----------------------------------------------------------------------
   Convert element symbol to atomic number
------------------------------------------------------------------------- */

int PairACE::element_to_Z(const char *elem)
{
  // Common elements - extend as needed
  static const struct { const char *name; int Z; } elements[] = {
    {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6},
    {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12},
    {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18},
    {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24},
    {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36},
    {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42},
    {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48},
    {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54},
    {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60},
    {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66},
    {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72},
    {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78},
    {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84},
    {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90},
    {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {nullptr, 0}
  };

  for (int i = 0; elements[i].name != nullptr; i++) {
    if (strcmp(elem, elements[i].name) == 0)
      return elements[i].Z;
  }

  return -1;  // Unknown element
}

/* ----------------------------------------------------------------------
   Global settings (none for this pair style)
------------------------------------------------------------------------- */

void PairACE::settings(int narg, char **/*arg*/)
{
  if (narg != 0)
    error->all(FLERR, "Illegal pair_style ace command");
}

/* ----------------------------------------------------------------------
   Set coefficients for one I,J pair
   Format: pair_coeff * * model.so elem1 elem2 ...
------------------------------------------------------------------------- */

void PairACE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // Minimum: pair_coeff * * model.so elem1
  if (narg < 4)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure wildcard atom types
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for ace must use * * model.so elem1 elem2 ...");

  // Number of elements should match number of atom types
  int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR, "Number of elements must match number of atom types");

  // Load the model
  load_model(arg[2]);

  // Allocate and populate element names
  if (element_names) {
    for (int i = 0; i <= ntypes; i++)
      delete[] element_names[i];
    delete[] element_names;
  }
  element_names = new char*[ntypes + 1];
  for (int i = 0; i <= ntypes; i++)
    element_names[i] = nullptr;

  // Allocate type_to_Z mapping
  memory->destroy(type_to_Z);
  memory->create(type_to_Z, ntypes + 1, "pair:type_to_Z");

  // Map LAMMPS atom types to atomic numbers
  for (int i = 1; i <= ntypes; i++) {
    const char *elem = arg[2 + i];

    // Handle NULL element (type not used)
    if (strcmp(elem, "NULL") == 0) {
      type_to_Z[i] = -1;
      element_names[i] = nullptr;
      continue;
    }

    // Store element name
    element_names[i] = new char[strlen(elem) + 1];
    strcpy(element_names[i], elem);

    // Convert to atomic number
    int Z = element_to_Z(elem);
    if (Z < 0) {
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg), "Unknown element '%s'", elem);
      error->all(FLERR, errmsg);
    }

    // Verify model supports this element
    bool supported = false;
    for (int j = 0; j < n_species; j++) {
      if (species_Z[j] == Z) {
        supported = true;
        break;
      }
    }
    if (!supported) {
      char errmsg[256];
      snprintf(errmsg, sizeof(errmsg),
               "Element '%s' (Z=%d) not supported by model", elem, Z);
      error->all(FLERR, errmsg);
    }

    type_to_Z[i] = Z;
  }

  // Set all type pairs as active
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
    }
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "ACE: Element mapping:\n");
    for (int i = 1; i <= ntypes; i++) {
      if (type_to_Z[i] > 0)
        utils::logmesg(lmp, "ACE:   Type {} -> {} (Z={})\n",
                       i, element_names[i], type_to_Z[i]);
      else
        utils::logmesg(lmp, "ACE:   Type {} -> NULL\n", i);
    }
  }
}

/* ----------------------------------------------------------------------
   Initialize style-specific settings
------------------------------------------------------------------------- */

void PairACE::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style ace requires atom IDs");

  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ace requires newton pair on");

  // Request a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   Initialize one type pair i,j
------------------------------------------------------------------------- */

double PairACE::init_one(int /*i*/, int /*j*/)
{
  return cutoff;
}

/* ----------------------------------------------------------------------
   Main compute function
------------------------------------------------------------------------- */

void PairACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double cutsq_local = cutoff * cutoff;

  // Loop over center atoms
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itype = type[i];
    int z0 = type_to_Z[itype];

    // Skip NULL types
    if (z0 < 0) continue;

    double xi = x[i][0];
    double yi = x[i][1];
    double zi = x[i][2];

    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    // Count neighbors within cutoff and resize work arrays if needed
    int nneigh = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[j][0] - xi;
      double dy = x[j][1] - yi;
      double dz = x[j][2] - zi;
      double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq < cutsq_local && rsq > 1e-10) {
        nneigh++;
      }
    }

    // Resize work arrays if needed
    if (nneigh > maxneigh) {
      maxneigh = nneigh + 16;  // Add some padding
      memory->destroy(neighbor_Z);
      memory->destroy(neighbor_Rij);
      memory->destroy(site_forces);
      memory->create(neighbor_Z, maxneigh, "pair:neighbor_Z");
      memory->create(neighbor_Rij, maxneigh * 3, "pair:neighbor_Rij");
      memory->create(site_forces, maxneigh * 3, "pair:site_forces");
    }

    // Build neighbor arrays (displacement vectors and atomic numbers)
    // Also store j indices for force accumulation
    int neighbor_j[nneigh];  // VLA for neighbor indices
    nneigh = 0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      int jtype = type[j];
      int zj = type_to_Z[jtype];
      if (zj < 0) continue;  // Skip NULL types

      double dx = x[j][0] - xi;
      double dy = x[j][1] - yi;
      double dz = x[j][2] - zi;
      double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq < cutsq_local && rsq > 1e-10) {
        neighbor_j[nneigh] = j;
        neighbor_Z[nneigh] = zj;
        neighbor_Rij[nneigh * 3 + 0] = dx;
        neighbor_Rij[nneigh * 3 + 1] = dy;
        neighbor_Rij[nneigh * 3 + 2] = dz;
        nneigh++;
      }
    }

    // Skip if no neighbors
    if (nneigh == 0) continue;

    // Initialize site forces and virial
    for (int k = 0; k < nneigh * 3; k++)
      site_forces[k] = 0.0;
    for (int k = 0; k < 6; k++)
      site_virial[k] = 0.0;

    // Call ACE model evaluation
    double site_energy = ace_site_energy_forces_virial(
        z0, nneigh, neighbor_Z, neighbor_Rij, site_forces, site_virial);

    // Accumulate energy
    if (eflag_global)
      eng_vdwl += site_energy;
    if (eflag_atom)
      eatom[i] += site_energy;

    // Accumulate forces
    // ACE returns forces ON neighbors, so:
    //   f[j] += site_forces[k]
    //   f[i] -= sum(site_forces)
    double fxi = 0.0, fyi = 0.0, fzi = 0.0;

    for (int k = 0; k < nneigh; k++) {
      int j = neighbor_j[k];
      double fx = site_forces[k * 3 + 0];
      double fy = site_forces[k * 3 + 1];
      double fz = site_forces[k * 3 + 2];

      f[j][0] += fx;
      f[j][1] += fy;
      f[j][2] += fz;

      fxi -= fx;
      fyi -= fy;
      fzi -= fz;
    }

    f[i][0] += fxi;
    f[i][1] += fyi;
    f[i][2] += fzi;

    // Accumulate virial
    // ACE returns virial as sum(r*f) in Voigt notation: [xx, yy, zz, yz, xz, xy]
    // LAMMPS expects virial as -sum(r*f) in order: [xx, yy, zz, xy, xz, yz]
    // So we negate the ACE virial values
    if (vflag_global) {
      virial[0] -= site_virial[0];  // xx
      virial[1] -= site_virial[1];  // yy
      virial[2] -= site_virial[2];  // zz
      virial[3] -= site_virial[5];  // xy (ACE index 5)
      virial[4] -= site_virial[4];  // xz (ACE index 4)
      virial[5] -= site_virial[3];  // yz (ACE index 3)
    }

    // Per-atom virial (distribute to center atom)
    if (vflag_atom) {
      vatom[i][0] -= site_virial[0];
      vatom[i][1] -= site_virial[1];
      vatom[i][2] -= site_virial[2];
      vatom[i][3] -= site_virial[5];
      vatom[i][4] -= site_virial[4];
      vatom[i][5] -= site_virial[3];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
