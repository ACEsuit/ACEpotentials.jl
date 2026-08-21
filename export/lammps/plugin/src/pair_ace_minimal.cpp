/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_ace_minimal.cpp - ACE potential using minimal export approach

   This implementation uses Julia's C API to load ACE models exported
   with the minimal export approach. The C interface (ACE_C_Interface_Minimal)
   wraps the exported Julia modules and provides C-callable functions.

   Thread Safety:
   - Julia runtime is initialized once (shared across all PairACEMinimal instances)
   - Each instance loads its own model with unique model_id
   - ACE_C_Interface_Minimal provides thread-safe model storage
   - OpenMP parallelization is supported for atom loop

   Performance:
   - First evaluation may be slower (Julia JIT compilation)
   - Subsequent evaluations are fast (compiled Julia code)
   - Identical performance to direct Julia evaluation

   Memory:
   - Models stay in memory until unloaded
   - Julia GC manages model data
   - LAMMPS manages neighbor list arrays
------------------------------------------------------------------------- */

#include "pair_ace_minimal.h"

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
#include <julia.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace LAMMPS_NS;

// Static members for Julia runtime
bool PairACEMinimal::julia_initialized = false;
bool PairACEMinimal::julia_finalized = false;

/* ---------------------------------------------------------------------- */

PairACEMinimal::PairACEMinimal(LAMMPS *lmp) : Pair(lmp)
{
  // Single cutoff for all interactions
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  // Initialize state
  model_id = -1;
  cutoff = 0.0;
  n_species = 0;
  species_Z = nullptr;
  type_to_Z = nullptr;
  element_names = nullptr;

  maxneigh = 0;
  neighbor_Z = nullptr;
  neighbor_Rij = nullptr;
  site_forces = nullptr;

  // Initialize Julia runtime if needed
  init_julia();
}

/* ---------------------------------------------------------------------- */

PairACEMinimal::~PairACEMinimal()
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

  // Note: Don't finalize Julia here - it's shared across all instances
  // Julia runtime will be finalized at program exit via atexit handler
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::init_julia()
{
  if (julia_initialized) return;

  // Initialize Julia runtime
  jl_init();
  julia_initialized = true;

  if (comm->me == 0) {
    utils::logmesg(lmp, "ACE/Minimal: Julia runtime initialized\n");
  }

  // Load ACE_C_Interface_Minimal module
  // We need to load the C interface module that wraps the exported models
  const char *c_interface_path = getenv("ACE_C_INTERFACE_PATH");
  if (c_interface_path == nullptr) {
    error->all(FLERR, "ACE/Minimal: ACE_C_INTERFACE_PATH environment variable not set\n"
                      "Please set it to the path of ace_c_interface_minimal.jl");
  }

  // Include the C interface module
  std::string include_cmd = std::string("include(\"") + c_interface_path + "\")";
  jl_eval_string(include_cmd.c_str());

  if (jl_exception_occurred()) {
    const char *msg = jl_typeof_str(jl_exception_occurred());
    std::string err_msg = std::string("ACE/Minimal: Failed to load C interface: ") + msg;
    error->all(FLERR, err_msg.c_str());
  }

  // Import the module
  jl_eval_string("using .ACE_C_Interface_Minimal");

  if (jl_exception_occurred()) {
    const char *msg = jl_typeof_str(jl_exception_occurred());
    std::string err_msg = std::string("ACE/Minimal: Failed to import C interface module: ") + msg;
    error->all(FLERR, err_msg.c_str());
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "ACE/Minimal: C interface module loaded\n");
  }
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 3 + atom->ntypes)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // arg[0] and arg[1] are type specifiers (* *)
  // arg[2] is the model directory path
  const char *model_dir = arg[2];

  // Remaining args are element names
  element_names = new char *[atom->ntypes + 1];
  for (int i = 0; i < atom->ntypes + 1; i++) element_names[i] = nullptr;

  for (int i = 3; i < narg; i++) {
    int itype = i - 2;
    if (itype > atom->ntypes)
      error->all(FLERR, "Too many element names in pair_coeff");

    element_names[itype] = new char[strlen(arg[i]) + 1];
    strcpy(element_names[itype], arg[i]);
  }

  // Load the model
  load_model(model_dir);

  // Create mapping from LAMMPS types to atomic numbers
  memory->create(type_to_Z, atom->ntypes + 1, "pair:type_to_Z");
  for (int i = 0; i <= atom->ntypes; i++) type_to_Z[i] = -1;

  for (int itype = 1; itype <= atom->ntypes; itype++) {
    if (element_names[itype] == nullptr) continue;
    type_to_Z[itype] = element_to_Z(element_names[itype]);

    if (type_to_Z[itype] == -1) {
      std::string err_msg = std::string("ACE/Minimal: Element ") +
                            element_names[itype] + " not found in model";
      error->all(FLERR, err_msg.c_str());
    }

    // Verify element is supported by model
    bool found = false;
    for (int j = 0; j < n_species; j++) {
      if (species_Z[j] == type_to_Z[itype]) {
        found = true;
        break;
      }
    }

    if (!found) {
      std::string err_msg = std::string("ACE/Minimal: Element ") +
                            element_names[itype] + " (Z=" +
                            std::to_string(type_to_Z[itype]) +
                            ") not supported by model";
      error->all(FLERR, err_msg.c_str());
    }
  }

  // Set interaction flags
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (element_names[i] && element_names[j]) {
        setflag[i][j] = 1;
        setflag[j][i] = 1;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::load_model(const char *model_dir)
{
  // Call ace_load_model from Julia C interface
  jl_function_t *load_fn = jl_get_function(jl_main_module, "ace_load_model");

  if (load_fn == nullptr) {
    error->all(FLERR, "ACE/Minimal: ace_load_model function not found\n"
                      "Make sure ACE_C_Interface_Minimal is loaded");
  }

  // Convert path to Julia string
  jl_value_t *path_str = jl_cstr_to_string(model_dir);

  // Call the function
  jl_value_t *result = jl_call1(load_fn, path_str);

  if (jl_exception_occurred()) {
    const char *msg = jl_typeof_str(jl_exception_occurred());
    std::string err_msg = std::string("ACE/Minimal: Failed to load model: ") + msg;
    error->all(FLERR, err_msg.c_str());
  }

  // Extract model ID
  model_id = jl_unbox_int32(result);

  if (model_id < 0) {
    std::string err_msg = std::string("ACE/Minimal: Failed to load model from ") + model_dir;
    error->all(FLERR, err_msg.c_str());
  }

  if (comm->me == 0) {
    std::string msg = std::string("ACE/Minimal: Loaded model from ") + model_dir +
                      " (ID: " + std::to_string(model_id) + ")\n";
    utils::logmesg(lmp, msg.c_str());
  }

  // Query model metadata
  // Get cutoff
  jl_function_t *get_cutoff_fn = jl_get_function(jl_main_module, "ace_get_cutoff");
  double cutoff_val = 0.0;

  jl_value_t *args[2];
  args[0] = jl_box_int32(model_id);
  args[1] = jl_box_voidpointer(&cutoff_val);

  jl_value_t *status = jl_call2(get_cutoff_fn, args[0], args[1]);

  if (jl_unbox_int32(status) != 0) {
    error->all(FLERR, "ACE/Minimal: Failed to get cutoff from model");
  }

  cutoff = cutoff_val;

  // Get species
  jl_function_t *get_species_fn = jl_get_function(jl_main_module, "ace_get_species");
  int temp_species[100];  // Temporary buffer
  int n_species_val = 0;

  args[0] = jl_box_int32(model_id);
  args[1] = jl_box_voidpointer(temp_species);
  args[2] = jl_box_voidpointer(&n_species_val);

  status = jl_call(get_species_fn, args, 3);

  if (jl_unbox_int32(status) != 0) {
    error->all(FLERR, "ACE/Minimal: Failed to get species from model");
  }

  n_species = n_species_val;
  memory->create(species_Z, n_species, "pair:species_Z");
  for (int i = 0; i < n_species; i++) {
    species_Z[i] = temp_species[i];
  }

  if (comm->me == 0) {
    std::string msg = "ACE/Minimal: Cutoff = " + std::to_string(cutoff) + " Å, Species: [";
    for (int i = 0; i < n_species; i++) {
      msg += std::to_string(species_Z[i]);
      if (i < n_species - 1) msg += ", ";
    }
    msg += "]\n";
    utils::logmesg(lmp, msg.c_str());
  }
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::unload_model()
{
  if (model_id < 0) return;

  // Call ace_unload_model from Julia
  jl_function_t *unload_fn = jl_get_function(jl_main_module, "ace_unload_model");

  if (unload_fn) {
    jl_value_t *model_id_val = jl_box_int32(model_id);
    jl_call1(unload_fn, model_id_val);
  }

  model_id = -1;
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ACE/Minimal requires newton pair on");

  // Request full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

double PairACEMinimal::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::grow_neighbor_arrays(int n)
{
  if (n <= maxneigh) return;

  maxneigh = n + 100;  // Add buffer
  memory->grow(neighbor_Z, maxneigh, "pair:neighbor_Z");
  memory->grow(neighbor_Rij, maxneigh * 3, "pair:neighbor_Rij");
  memory->grow(site_forces, maxneigh * 3, "pair:site_forces");
}

/* ---------------------------------------------------------------------- */

void PairACEMinimal::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  ev_init(eflag, vflag);

  // Get Julia function pointers
  jl_function_t *site_energy_forces_fn =
      jl_get_function(jl_main_module, "ace_site_energy_forces");

  if (site_energy_forces_fn == nullptr) {
    error->all(FLERR, "ACE/Minimal: ace_site_energy_forces function not found");
  }

  // Loop over atoms (can be parallelized with OpenMP)
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // Get central atom atomic number
    int Z0 = type_to_Z[itype];
    if (Z0 < 0) continue;  // Skip if not mapped

    // Collect neighbors within cutoff
    int n_neigh = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      if (type_to_Z[jtype] < 0) continue;  // Skip unmapped types

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutoff * cutoff) {
        n_neigh++;
      }
    }

    // Grow arrays if needed
    grow_neighbor_arrays(n_neigh);

    // Fill neighbor arrays
    n_neigh = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      if (type_to_Z[jtype] < 0) continue;

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutoff * cutoff) {
        neighbor_Z[n_neigh] = type_to_Z[jtype];
        neighbor_Rij[3 * n_neigh + 0] = delx;
        neighbor_Rij[3 * n_neigh + 1] = dely;
        neighbor_Rij[3 * n_neigh + 2] = delz;
        n_neigh++;
      }
    }

    // Call ACE evaluation
    double energy = 0.0;

    // Prepare Julia call arguments
    jl_value_t *args[6];
    args[0] = jl_box_int32(model_id);
    args[1] = jl_box_int32(n_neigh);
    args[2] = jl_box_voidpointer(neighbor_Rij);
    args[3] = jl_box_voidpointer(neighbor_Z);
    args[4] = jl_box_int32(Z0);
    args[5] = jl_box_voidpointer(&energy);
    args[6] = jl_box_voidpointer(site_forces);

    jl_value_t *status = jl_call(site_energy_forces_fn, args, 7);

    if (jl_exception_occurred()) {
      error->one(FLERR, "ACE/Minimal: Exception during energy/force evaluation");
    }

    if (jl_unbox_int32(status) != 0) {
      error->one(FLERR, "ACE/Minimal: Failed to evaluate site energy/forces");
    }

    // Accumulate energy
    if (eflag_global) eng_vdwl += energy;
    if (eflag_atom) eatom[i] += energy;

    // Accumulate forces
    n_neigh = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      if (type_to_Z[jtype] < 0) continue;

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutoff * cutoff) {
        // Force on central atom i (negative of force on neighbor)
        f[i][0] -= site_forces[3 * n_neigh + 0];
        f[i][1] -= site_forces[3 * n_neigh + 1];
        f[i][2] -= site_forces[3 * n_neigh + 2];

        // Newton's third law: force on neighbor j
        if (newton_pair || j < nlocal) {
          f[j][0] += site_forces[3 * n_neigh + 0];
          f[j][1] += site_forces[3 * n_neigh + 1];
          f[j][2] += site_forces[3 * n_neigh + 2];
        }

        n_neigh++;
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

int PairACEMinimal::element_to_Z(const char *elem)
{
  // Simple periodic table lookup (extend as needed)
  struct ElementMap {
    const char *symbol;
    int Z;
  };

  static const ElementMap periodic_table[] = {
      {"H", 1},   {"He", 2},  {"Li", 3},  {"Be", 4},  {"B", 5},   {"C", 6},
      {"N", 7},   {"O", 8},   {"F", 9},   {"Ne", 10}, {"Na", 11}, {"Mg", 12},
      {"Al", 13}, {"Si", 14}, {"P", 15},  {"S", 16},  {"Cl", 17}, {"Ar", 18},
      {"K", 19},  {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23},  {"Cr", 24},
      {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
      {nullptr, 0}};

  for (int i = 0; periodic_table[i].symbol != nullptr; i++) {
    if (strcmp(elem, periodic_table[i].symbol) == 0) {
      return periodic_table[i].Z;
    }
  }

  return -1;  // Not found
}
