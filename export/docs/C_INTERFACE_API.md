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
