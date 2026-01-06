# ETACE Spline Fair Benchmark Deployment
# Generated: 2026-01-05T21:19:18.465

## Model Parameters
- Elements: [:Ti, :Al]
- Order: 3
- Total Degree: 8
- Max L: 4
- Cutoff: 5.5 A
- E0s: Ti=-1586.0195 eV, Al=-105.5954 eV

## Training
- Configurations: 66
- Solver: QR(lambda=0.001)
- Prior: algebraic_smoothness_prior(p=4)

## Results
- Basis size: 308

## Files
- Model source: fair_etace_spline_model.jl
- Compiled lib: lib/libace_fair_etace_spline.so

## Notes
- Uses EquivariantTensors backend with LearnableRnlrzzBasis
- Converted to ETACE format for export
- Splinified with Nspl=50 for Hermite cubic spline evaluation
