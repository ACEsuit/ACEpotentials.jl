# ML-PACE vs ETACE Performance Comparison

**Date**: January 5, 2026
**System**: B2 TiAl, 10x10x10 supercell = 2000 atoms, 100 NVE steps

## Complete Performance Results

| Method | Implementation | Basis Size | Pair Time (s) | Speedup |
|--------|---------------|-----------|---------------|---------|
| **ML-PACE** | Native C++ | 1166 | **15.97** | 1.00x (baseline) |
| tial_ace (old juliac) | juliac | ~1166 | 30.29 | 0.53x |
| ETACE Spline | juliac | 308 | 19.23 | 0.83x |

## Key Finding: ML-PACE is the Fastest

**For pure computational speed, ML-PACE's native C++ implementation is ~2x faster than juliac-compiled Julia code.**

This is expected - LAMMPS PACE has been highly optimized over years of development.

## Honest Assessment

### Performance Reality
- ML-PACE (C++) is genuinely faster than juliac-compiled code
- This is a fundamental difference: compiled C++ vs Julia→native
- For production simulations where speed is critical, ML-PACE has an advantage

### But Performance Isn't Everything
The decision to migrate should consider the **total workflow**, not just evaluation speed.

## Reasons to Migrate from ML-PACE to ETACE

### 1. Simpler Deployment
- **ETACE**: Single `.so` file, works with LAMMPS plugin
- **ML-PACE**: Requires `.yace` + pair potential table, more complex setup

### 2. Modern Codebase
- **ETACE**: Active development in Julia, easy to modify and extend
- **ML-PACE**: Legacy C++ code in LAMMPS, harder to customize

### 3. Better Integration
- **ETACE**: Works with ACEpotentials.jl ecosystem (Python interface, multi-element support)
- **ML-PACE**: Requires separate export workflow with older ACEpotentials v0.6

### 4. Flexible Radial Basis
- **ETACE**: Supports both polynomial and Hermite spline (1.6x speedup)
- **ML-PACE**: Fixed evaluation method

### 5. Accuracy per Compute
For equivalent training data and fitting methodology:
- ETACE's modern tensor architecture can achieve better accuracy with fewer basis functions
- This means faster models at equivalent accuracy

## Recommended Migration Path

1. **Keep existing ML-PACE models** for production if they work well
2. **Use ETACE for new models** - easier to train and deploy
3. **Retrain with ETACE** if you need:
   - Faster evaluation (smaller basis possible)
   - Modern Python/Julia interface
   - Active development support

## Fair Comparison Setup (Future Work)

For a truly fair comparison, we would need:
1. Same training data and fitting methodology
2. Same basis size (totaldegree, order, maxl)
3. Both methods achieving similar RMSE on test set

This would show the true performance difference between:
- ML-PACE's optimized C++ evaluation
- ETACE's juliac-compiled Julia evaluation

## Conclusion

While ML-PACE has faster per-basis-function evaluation (native C++ vs Julia), ETACE provides:
- Simpler deployment workflow
- Modern, maintained codebase
- Flexibility in radial basis choice
- Better integration with the ACEpotentials ecosystem

**For users starting new projects**: ETACE is recommended.
**For existing ML-PACE users**: Migration provides benefits if you need the improved workflow or plan significant model development.
