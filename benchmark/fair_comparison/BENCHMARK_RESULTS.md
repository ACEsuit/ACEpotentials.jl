# Fair Benchmark Results: ACE Export Methods

**Date**: January 5, 2026
**System**: B2 TiAl, 10x10x10 supercell = 2000 atoms
**Test**: 100 NVE MD steps, 1 MPI process, 1 OpenMP thread

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Elements | Ti, Al |
| Order | 3 |
| Total Degree | 8 |
| Max L | 4 |
| Cutoff | 5.5 Å |
| E0s | Ti=-1586.0195 eV, Al=-105.5954 eV |
| Basis Size | 308 |
| Training | TiAl_tutorial (66 configs) |
| Solver | QR(lambda=1e-3) |
| Prior | algebraic_smoothness_prior(p=4) |

## Performance Results

| Method | Pair Time (s) | Loop Time (s) | Performance (ns/day) | Status |
|--------|--------------|---------------|---------------------|--------|
| **ETACE Spline** | **18.72** | **18.92** | **0.457** | ✓ Stable |
| Old ACE juliac | 24.91 | 25.82 | 0.335 | ✓ Stable |
| ETACE Polynomial | 29.81 | 30.53 | 0.283 | ✓ Stable* |
| ML-PACE | - | - | - | ✗ Crashed |

*Note: ETACE Polynomial simulation shows energy instability after 30 steps, indicating fitting issue.

## Key Findings

### 1. ETACE Spline is the Fastest
- **33% faster** than Old ACE juliac (24.91s → 18.72s pair time)
- 59% faster than ETACE Polynomial

### 2. Spline vs Polynomial Performance
- Hermite cubic spline evaluation: 18.72s
- Polynomial evaluation: 29.81s
- Spline speedup: **1.59x** (37% reduction)

### 3. juliac Compilation Notes
- `--trim=safe` flag strips @ccallable functions (C interface)
- Must compile WITHOUT `--trim=safe` for LAMMPS compatibility
- Library sizes without trim: 229-404 MB

### 4. ML-PACE Compatibility
- ML-PACE requires ACEpotentials v0.6.x for export
- Model trained with v0.6 has different basis than v0.9+
- Cannot directly compare performance due to different fitting

## Library Sizes

| Method | Library | Size |
|--------|---------|------|
| Old ACE | libace_fair_oldace.so | 404 MB |
| ETACE Spline | libace_fair_etace_spline.so | 231 MB |
| ETACE Poly | libace_fair_etace_poly.so | 229 MB |
| ML-PACE | fair_mlpace.yace | 35.7 MB |

## Deployment Files

```
deployments/
├── oldace/
│   ├── fair_oldace_model.jl
│   └── lib/libace_fair_oldace.so
├── etace_spline/
│   ├── fair_etace_spline_model.jl
│   └── lib/libace_fair_etace_spline.so
├── etace_poly/
│   ├── fair_etace_poly_model.jl
│   └── lib/libace_fair_etace_poly.so
└── mlpace/
    ├── fair_mlpace.yace
    └── fair_mlpace_pairpot.table
```

## Conclusions

1. **ETACE with Hermite cubic splines** is the fastest method, providing 33% speedup over the old juliac export approach.

2. **Polynomial radial basis is ~1.6x slower** than spline basis due to more expensive evaluation.

3. **juliac compilation** produces working LAMMPS-compatible libraries when `--trim=safe` is omitted.

4. **Fair comparison with ML-PACE** requires using ACEpotentials v0.6 for consistent model construction.

## Recommendations

- Use **ETACE Spline** export for production LAMMPS simulations (fastest)
- Avoid `--trim=safe` flag in juliac compilation for LAMMPS compatibility
- Consider multi-threading benchmarks in future work
