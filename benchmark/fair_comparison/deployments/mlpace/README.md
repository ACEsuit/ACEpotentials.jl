# ML-PACE Fair Benchmark Deployment
# Generated: 2026-01-05T20:39:07.716
# ACEpotentials version: 0.6.12

## Model Parameters
- Elements: [:Ti, :Al]
- Order: 3
- Total Degree: 8
- Cutoff: 5.5 Å
- E0s: Ti=-1586.0195 eV, Al=-105.5954 eV

## Training
- Configurations: 66
- Solver: QR(lambda=0.001)
- Prior: smoothness_prior(p=4)

## Results
- Basis size: 490

## Files
- PACE model: fair_mlpace.yace
- Pair table: fair_mlpace_pairpot.table

## LAMMPS Usage
```lammps
pair_style      hybrid/overlay pace table spline 5500
pair_coeff      * * pace fair_mlpace.yace Ti Al
pair_coeff      1 1 table fair_mlpace_pairpot.table Ti_Ti
pair_coeff      1 2 table fair_mlpace_pairpot.table Al_Ti
pair_coeff      2 2 table fair_mlpace_pairpot.table Al_Al
```
