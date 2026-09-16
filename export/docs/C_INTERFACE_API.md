# C Interface API

**There are TWO C interfaces in this repository, and they are not the same ABI.**

| | exported/compiled library | minimal export |
|---|---|---|
| produced by | `export_ace_model(...; for_library = true)` + `juliac --trim=safe` | *no longer in the tree* -- see the note below |
| one model per | shared library (`libace_<model>.so`) | `model_id`, several per process |
| consumed by | the LAMMPS `pair_style ace` plugin, `ase_ace.ACELibraryCalculator` | in-process Julia embedding |
| documented in | **[the section immediately below](#compiled-library-abi-ccallable)** | the rest of this file |

Everything from "C Interface API for Minimal Export" onward describes the SECOND one. If you
are writing a LAMMPS pair style or a ctypes wrapper against a `.so`, the first is what you
want.

**The minimal export's implementation was deleted in Task 6 and its documentation is kept for
reference only.** `export/src/ace_c_interface.jl` was `include`d by nothing, and the minimal
plugin (`pair_ace_minimal.cpp`) looks for a differently-named `ace_c_interface_minimal.jl`
that has never existed in this tree. What it carried was a copy of the `ace_site_*` entry
points **without the workspace handle** -- precisely the stale ABI that the tagged handles in
the compiled library now exist to reject -- so leaving it there was a standing invitation to
copy the wrong signatures. If the minimal path is revived, write it against the ABI documented
above.

---

# Compiled-library ABI (`@ccallable`)

The symbols `Base.@ccallable`-exported by a generated model file. All of them are plain C:
no Julia runtime call is needed beyond loading the library.

## Workspaces and re-entrancy

Since the per-neighbour kernel (B2) the library holds **no mutable global state**. All scratch
lives in a *workspace*, which the caller obtains and passes to every evaluation entry point as
its **first argument**:

```c
void  *ace_workspace_new(void);   /* NULL when the pool is exhausted */
void   ace_workspace_free(void *ws);
int    ace_max_workspaces(void);  /* size of the pool, fixed when the library was built */
```

* The library is **re-entrant given one workspace per concurrent caller**. It is **not**
  thread-safe with a shared workspace, and there is no internal locking that would make it so:
  two threads in one workspace corrupt each other's `A`, `AA` and `∂A`.
* **The pool is FIXED SIZE and is built into the library.** `ace_max_workspaces()` returns it
  (currently 32 for a model exported by this generator). `ace_workspace_new` returns **NULL**
  once that many are outstanding — check the return value, and size your thread pool with
  `ace_max_workspaces()`. It is not a bound on anything physical; to raise it, change
  `MAX_WORKSPACES` in `export/src/write_c_interface.jl` and re-export the model.
* A workspace is **sized from the MODEL** (`N_A`, `N_AA`, `N_BASIS`) and never from the
  neighbour count. It does not grow, and it is never reallocated. There is consequently **no
  maximum neighbour count**: the pre-B2 `MAX_NEIGHBORS = 256` cap is gone, and a site of any
  size uses the same buffers.
* `ace_workspace_new` / `ace_workspace_free` take an internal lock (they flip a slot's
  *taken* flag in the shared pool), so they may be called from anywhere, including inside a
  parallel region. The evaluation entries take **no** lock. Allocate every workspace *before*
  concurrent use all the same.
* Handles are **tagged opaque integers**, not pointers into the library's heap, and they are
  validated on every call: passing something that is not a handle is rejected rather than
  served. Freeing is not required before `dlclose()` (the pool is part of the image), but it
  is good practice and it returns the slot.

WHY THE POOL IS IN THE IMAGE, in case a future change is tempted to allocate workspaces
lazily: a runtime-allocated Julia object held across C calls is **not reliably rooted** in a
`juliac --trim` library. Measured, twice — `pointer_from_objref` gives a `TypeError` on the
first garbage collection, and an index into a runtime-populated global instead gives a smashed
malloc arena. Both survive a single force evaluation and die in a real run. See the comment
block at the top of the workspace section in `export/src/write_c_interface.jl`.

The LAMMPS plugin allocates `workspaces[t]` for `t < omp_get_max_threads()` in `init_style()`
and passes `workspaces[omp_get_thread_num()]`; `ase_ace.ACELibrary` owns exactly one per
instance.

## Evaluation

```c
double ace_site_energy(void *ws, int z0, int nneigh,
                       const int *neighbor_z, const double *neighbor_Rij);

double ace_site_energy_forces(void *ws, int z0, int nneigh,
                              const int *neighbor_z, const double *neighbor_Rij,
                              double *forces);

double ace_site_energy_forces_virial(void *ws, int z0, int nneigh,
                                     const int *neighbor_z, const double *neighbor_Rij,
                                     double *forces, double *virial);

int    ace_site_basis(void *ws, int z0, int nneigh,
                      const int *neighbor_z, const double *neighbor_Rij,
                      double *basis_out);

void   ace_batch_energy_forces_virial(void *ws, int natoms, const int *z,
                                      const int *neighbor_counts,
                                      const int *neighbor_offsets,
                                      const int *neighbor_z, const double *neighbor_Rij,
                                      double *energies, double *forces, double *virials);
```

* `neighbor_Rij` is `nneigh * 3` doubles, **displacement vectors** `R_j - R_i` in Å, not
  positions.
* `forces` is `nneigh * 3` doubles: the force **on each neighbour**, `-dE_i/dR_j`. The force
  on the centre atom is minus their sum (Newton's third law); LAMMPS does that itself.
* `virial` is 6 doubles in Voigt order `xx, yy, zz, yz, xz, xy`.
* The returned energy is the **site energy including E0** of the centre species (and the pair
  term, when the model has one). `nneigh == 0` returns E0 and writes zeros.
* `ace_site_energy_forces` is the same as the `_virial` entry without the per-edge outer
  product; call it when no virial is wanted.
* `ace_batch_*` is sequential inside one call and uses the one workspace it is given. To
  evaluate concurrently, split the atom range across threads, one workspace each.

## Metadata (no workspace, no state)

```c
double ace_get_cutoff(void);      /* Å */
int    ace_get_n_species(void);
int    ace_get_species(int idx);  /* 1-based; atomic number, or -1 out of range */
int    ace_get_n_basis(void);
unsigned long long ace_build_id(void);   /* provenance; see below */
```

## Version and provenance checks

A model compiled before the workspace API exports `ace_site_*` **without** the handle
argument. Calling such a library through the signatures above passes `z0` where the handle
belongs; the handle is tagged and validated, so the call is *rejected* rather than served, but
you still get no result. Both the LAMMPS plugin and `ACELibrary` therefore resolve
`ace_workspace_new` first and **refuse to load** a library that does not export it. Do not
paper over that: re-export and re-compile the model.

`ace_build_id()` returns a 64-bit hash of the generated source the library was compiled from
(everything above the `ACE EXPORT BUILD STAMP` marker line in the `.jl`). Nothing else about a
`.so` records its source, so this is the only way to answer "was this library built from that
model file?" — `export_build_id(<file>)` in `export/src/build_stamp.jl` recomputes it, and
`export/test/runtests.jl` refuses to run the library test groups on a mismatch. Read it out of
process (ctypes, or any C caller): `ccall`ing into a `juliac --trim` library from a host Julia
process aborts that process.

---

# C Interface API for Minimal Export

This document describes the C-compatible API for ACE models exported using the minimal export approach.

## Overview

The minimal C interface provides functions to:
- Load exported ACE models
- Compute site energies
- Compute site energies and forces
- Compute ACE basis vectors
- Query model metadata (cutoff, species, basis size)
- Unload models

All functions are thread-safe and support multiple loaded models simultaneously.

---

## Quick Start

```julia
# Load the C interface module
include("export/src/ace_c_interface_minimal.jl")
using .ACE_C_Interface_Minimal

# Load a model
model_path = Base.unsafe_convert(Cstring, Base.cconvert(Cstring, "/path/to/model"))
model_id = ace_load_model(model_path)

# Compute site energy
Rs_flat = [3.0, 0.0, 0.0]  # Neighbor position (x, y, z)
Zs = Int32[14]             # Neighbor atomic number
Z0 = Int32(14)             # Central atom

E = ace_site_energy(model_id, Int32(1), pointer(Rs_flat), pointer(Zs), Z0)

# Unload when done
ace_unload_model(model_id)
```

---

## API Functions

### Model Management

#### `ace_load_model(model_path::Cstring) -> Cint`

Load an exported ACE model from a directory.

**Parameters:**
- `model_path`: C string (null-terminated) pointing to the model directory

**Returns:**
- Positive integer (model ID) on success
- `-1` on failure

**Example:**
```julia
model_path = Base.unsafe_convert(Cstring, Base.cconvert(Cstring, "/path/to/model"))
model_id = ace_load_model(model_path)

if model_id == -1
    error("Failed to load model")
end
```

**Notes:**
- Models are stored in thread-safe global storage
- Multiple models can be loaded simultaneously
- Each model gets a unique ID

---

#### `ace_unload_model(model_id::Cint) -> Cint`

Unload a previously loaded model.

**Parameters:**
- `model_id`: Model ID returned by `ace_load_model`

**Returns:**
- `0` on success
- `-1` on failure (model not loaded)

**Example:**
```julia
status = ace_unload_model(model_id)
```

---

### Model Metadata

#### `ace_get_cutoff(model_id::Cint, cutoff_ptr::Ptr{Float64}) -> Cint`

Get the cutoff radius for a loaded model.

**Parameters:**
- `model_id`: Model ID
- `cutoff_ptr`: Pointer to store cutoff radius (in Ångströms)

**Returns:**
- `0` on success
- `-1` on failure

**Example:**
```julia
cutoff_ref = Ref{Float64}(0.0)
status = ace_get_cutoff(model_id, Base.unsafe_convert(Ptr{Float64}, cutoff_ref))
println("Cutoff: $(cutoff_ref[]) Å")
```

---

#### `ace_get_species(model_id::Cint, species_ptr::Ptr{Cint}, n_species_ptr::Ptr{Cint}) -> Cint`

Get the list of atomic species (atomic numbers) supported by the model.

**Parameters:**
- `model_id`: Model ID
- `species_ptr`: Pointer to array to store species (must be pre-allocated!)
- `n_species_ptr`: Pointer to store number of species

**Returns:**
- `0` on success
- `-1` on failure

**Example:**
```julia
species_arr = zeros(Int32, 10)
n_species_ref = Ref{Int32}(0)

status = ace_get_species(
    model_id,
    pointer(species_arr),
    Base.unsafe_convert(Ptr{Int32}, n_species_ref)
)

species = species_arr[1:n_species_ref[]]
println("Species: $species")
```

---

#### `ace_get_n_basis(model_id::Cint, n_basis_ptr::Ptr{Cint}) -> Cint`

Get the ACE basis size for a loaded model.

**Parameters:**
- `model_id`: Model ID
- `n_basis_ptr`: Pointer to store basis size

**Returns:**
- `0` on success
- `-1` on failure

**Example:**
```julia
n_basis_ref = Ref{Int32}(0)
status = ace_get_n_basis(model_id, Base.unsafe_convert(Ptr{Int32}, n_basis_ref))
println("Basis size: $(n_basis_ref[])")
```

---

### Energy and Force Evaluation

#### `ace_site_energy(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64}, Zs_ptr::Ptr{Cint}, Z0::Cint) -> Float64`

Compute site energy for a central atom and its neighbors.

**Parameters:**
- `model_id`: Model ID
- `n_neigh`: Number of neighbors
- `Rs_ptr`: Pointer to neighbor positions (flat array: `[x1, y1, z1, x2, y2, z2, ...]`)
- `Zs_ptr`: Pointer to neighbor atomic numbers
- `Z0`: Central atom atomic number

**Returns:**
- Site energy in eV
- `NaN` on error

**Example:**
```julia
# Single neighbor at (3.0, 0.0, 0.0) Å
Rs_flat = [3.0, 0.0, 0.0]
Zs = Int32[14]  # Silicon
Z0 = Int32(14)  # Silicon center

E = ace_site_energy(model_id, Int32(1), pointer(Rs_flat), pointer(Zs), Z0)
println("Energy: $E eV")
```

**Notes:**
- Positions are relative to the central atom (central atom is at origin)
- Positions must be in Ångströms
- For isolated atoms (no neighbors), pass `n_neigh = 0` (returns E0 value)

---

#### `ace_site_energy_forces(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64}, Zs_ptr::Ptr{Cint}, Z0::Cint, energy_ptr::Ptr{Float64}, forces_ptr::Ptr{Float64}) -> Cint`

Compute site energy and forces for a central atom and its neighbors.

**Parameters:**
- `model_id`: Model ID
- `n_neigh`: Number of neighbors
- `Rs_ptr`: Pointer to neighbor positions (flat array)
- `Zs_ptr`: Pointer to neighbor atomic numbers
- `Z0`: Central atom atomic number
- `energy_ptr`: Pointer to store energy (in eV)
- `forces_ptr`: Pointer to store forces (flat array: `[fx1, fy1, fz1, fx2, fy2, fz2, ...]`, in eV/Å)

**Returns:**
- `0` on success
- `-1` on failure

**Example:**
```julia
# Two neighbors
Rs_flat = [
    3.0, 0.0, 0.0,  # Neighbor 1
    0.0, 3.0, 0.0   # Neighbor 2
]
Zs = Int32[14, 14]
Z0 = Int32(14)

energy_ref = Ref{Float64}(0.0)
forces_flat = zeros(Float64, 6)  # 2 neighbors × 3 components

status = ace_site_energy_forces(
    model_id,
    Int32(2),
    pointer(Rs_flat),
    pointer(Zs),
    Z0,
    Base.unsafe_convert(Ptr{Float64}, energy_ref),
    pointer(forces_flat)
)

println("Energy: $(energy_ref[]) eV")
println("Force on neighbor 1: [$(forces_flat[1]), $(forces_flat[2]), $(forces_flat[3])] eV/Å")
println("Force on neighbor 2: [$(forces_flat[4]), $(forces_flat[5]), $(forces_flat[6])] eV/Å")
```

**Notes:**
- Forces are on neighbors, not on the central atom
- To get force on central atom: `F_center = -sum(F_neighbors)`
- Forces must be pre-allocated with size `3 * n_neigh`

---

### Basis Evaluation

#### `ace_site_basis(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64}, Zs_ptr::Ptr{Cint}, Z0::Cint, basis_ptr::Ptr{Float64}, n_basis_ptr::Ptr{Cint}) -> Cint`

Compute ACE basis vector for a central atom and its neighbors.

**Parameters:**
- `model_id`: Model ID
- `n_neigh`: Number of neighbors
- `Rs_ptr`: Pointer to neighbor positions (flat array)
- `Zs_ptr`: Pointer to neighbor atomic numbers
- `Z0`: Central atom atomic number
- `basis_ptr`: Pointer to store basis vector (must be pre-allocated!)
- `n_basis_ptr`: Pointer to store basis size

**Returns:**
- `0` on success
- `-1` on failure

**Example:**
```julia
# Get basis size first
n_basis_ref = Ref{Int32}(0)
ace_get_n_basis(model_id, Base.unsafe_convert(Ptr{Int32}, n_basis_ref))

# Allocate basis array
basis_arr = zeros(Float64, n_basis_ref[])
n_basis_out = Ref{Int32}(0)

# Compute basis
Rs_flat = [3.0, 0.0, 0.0]
Zs = Int32[14]
Z0 = Int32(14)

status = ace_site_basis(
    model_id,
    Int32(1),
    pointer(Rs_flat),
    pointer(Zs),
    Z0,
    pointer(basis_arr),
    Base.unsafe_convert(Ptr{Int32}, n_basis_out)
)

println("Basis size: $(n_basis_out[])")
println("Basis: $basis_arr")
```

**Notes:**
- Basis vector must be pre-allocated with correct size (use `ace_get_n_basis`)
- Basis vector is the ACE descriptors before weight contraction
- Useful for linear fitting or active learning

---

## Thread Safety

All functions are thread-safe through internal locking. Multiple threads can safely:
- Load/unload different models simultaneously
- Evaluate different model IDs concurrently
- Query metadata from different models

**Note:** Evaluations on the *same* model ID are serialized through Julia's threading model. For parallel evaluation, load multiple copies of the model (different IDs).

---

## Error Handling

Functions return error codes or `NaN` to indicate failures:

**Integer Return Values:**
- `>= 0`: Success (model ID for `ace_load_model`, `0` for other functions)
- `-1`: Failure

**Float Return Values:**
- Finite value: Success
- `NaN`: Failure

**Common Errors:**
- Model not found (bad path)
- Invalid model ID
- Missing model file or corrupted data
- Out of memory

Errors are logged to Julia's logger with `@error` macros.

---

## Memory Management

**User Responsibilities:**
- Pre-allocate output arrays for forces, basis
- Ensure arrays are large enough (`3 * n_neigh` for forces, `n_basis` for basis)
- Unload models when done (`ace_unload_model`)

**Automatic:**
- Model data is managed by Julia's GC
- Internal allocations handled automatically
- No manual memory management needed

---

## Performance Notes

1. **First Evaluation:** May be slower due to Julia JIT compilation
2. **Subsequent Evaluations:** Fast (compiled code)
3. **Batch Evaluation:** For many atoms, call site functions in a loop
4. **Memory:** Models stay in memory until unloaded

---

## Integration Examples

### LAMMPS Pair Style

```cpp
// In pair_ace.cpp
extern "C" {
    int jl_ace_load_model(const char* path);
    double jl_ace_site_energy(int model_id, int n_neigh,
                               double* Rs, int* Zs, int Z0);
}

void PairACE::coeff(int narg, char **arg) {
    model_id = jl_ace_load_model(model_path);
    if (model_id < 0) error->all(FLERR, "Failed to load ACE model");
}

void PairACE::compute(int eflag, int vflag) {
    for (int ii = 0; ii < inum; ii++) {
        // Collect neighbors...
        double E = jl_ace_site_energy(model_id, n_neigh, Rs, Zs, Z0);
        eng_vdwl += E;
    }
}
```

### Python via PyJulia

```python
from julia import Julia, Main

# Initialize Julia
jl = Julia(compiled_modules=False)

# Load C interface
Main.include("export/src/ace_c_interface_minimal.jl")

# Load model
from julia import ACE_C_Interface_Minimal as ace
model_id = ace.ace_load_model("/path/to/model")

# Evaluate
import numpy as np
Rs_flat = np.array([3.0, 0.0, 0.0])
Zs = np.array([14], dtype=np.int32)
Z0 = np.int32(14)

E = ace.ace_site_energy(model_id, 1, Rs_flat.ctypes.data,
                         Zs.ctypes.data, Z0)
print(f"Energy: {E} eV")
```

---

## Testing

Run the test suite:
```bash
julia --project=. export/test/test_c_interface_minimal.jl
```

Tests verify:
- ✅ Model loading/unloading
- ✅ Metadata queries
- ✅ Site energy evaluation
- ✅ Site energy and forces
- ✅ Basis evaluation
- ✅ Multiple neighbors
- ✅ Isolated atoms
- ✅ Error handling

---

## Comparison with Old Approach

| Feature | Old (Code Generation) | New (Minimal) |
|---------|----------------------|---------------|
| Lines of code | ~3,500 | ~600 (interface + minimal export) |
| Accuracy | Bugs in generated code | Perfect (uses actual ACEpotentials) |
| Maintenance | Must sync with ACEpotentials | Automatic |
| API stability | Generated code changes | Stable C interface |
| Memory usage | Large generated files | Serialized data |
| Load time | Fast (compiled) | Fast (deserialize) |

---

## Troubleshooting

### Model fails to load

**Symptoms:** `ace_load_model` returns `-1`

**Solutions:**
- Check that model directory exists
- Verify `.jl` wrapper file is present
- Check Julia logs for detailed error messages
- Ensure dependencies are installed (`Pkg.instantiate()` in model directory)

### Energy is NaN

**Symptoms:** `ace_site_energy` returns `NaN`

**Solutions:**
- Check that model_id is valid (returned from `ace_load_model`)
- Verify neighbor positions are finite
- Check atomic numbers are in model's species list
- Ensure cutoff radius is respected

### Segmentation fault

**Symptoms:** Crash during evaluation

**Solutions:**
- Verify pointer arguments are valid
- Ensure arrays are allocated with correct size
- Check that `n_neigh` matches array sizes
- Use `Base.unsafe_convert` for Julia Ref types

---

## Future Enhancements

Potential additions:
- Virial/stress computation
- Descriptor gradients for fitting
- Batch evaluation API
- MPI support for distributed evaluation
- GPU acceleration (when upstream supports it)
