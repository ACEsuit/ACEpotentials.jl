# Committed benchmark artefacts

The files a published number is quoted from, kept in the repository rather than only in the
untracked `bench_parity/` working directory.

**Why they are here.** Task 5 recorded that every artefact backing the README's rows was
untracked, so the published figures were reproducible only for as long as one scratch
directory survived on one host; Task 7 then found two published tables with no artefact at
all, and committing the script behind one of them revealed that the table was wrong by ~17 %.
A number whose evidence is not in the repository is a number nobody can check.

**Why `.txt` and not `.log`.** The repository's root `.gitignore` carries `export/**/*.log`.
These are copies of `bench_parity/*.log`, renamed rather than un-ignored, because changing a
root-level ignore rule to commit a benchmark transcript is a larger change than the transcript
is worth.

| file | what it is | produced by |
|---|---|---|
| `rows_task8.txt` | the close-out table's rows, one line per block, in execution order | `export/bench/run_task8_table.sh` |
| `task8_table.txt` | the session transcript for those rows (block order, loadavg, plugin) | the same script |
| `bytecmp_task8.txt` | generated source vs `b826c831`, four models, `EXPORT_BUILD_ID` | `export/test/bytecmp_generator.jl` |
| `mpi4_task8.txt` | the 4-rank `%varavg` sanity check on the 2000-atom TiAl box | `export/bench/mpi_sanity.sh` |
| `suite_task8.txt` | **HISTORICAL** — the full test suite before the `:hermite_spline` removal (32 562 pass, 10 groups). Kept as the reconciliation baseline for what that removal took. | `ACE_REQUIRE_GROUPS=all runtests.jl` |
| `suite_close_out.txt` | the full test suite **as it stands** (32 485 pass, 9 groups) — the evidence for the finding's headline figure | `ACE_REQUIRE_GROUPS=all runtests.jl` |
| `omp_task8.txt` | `OMP_NUM_THREADS=4` vs serial, bitwise, both models | `julia -t 4 test_workspace.jl` |
| `mpi_psT/*_psT.txt` | the `ps -T` thread census of each 4-rank run — the evidence that a rank is not single-threaded | `export/bench/mpi_sanity.sh` |
| `mpi_psT/*_screen.txt` | the LAMMPS screen logs the §4 per-rank figures are read from | the same |
| `gates/*.gated.txt` | the gate manifest of each shipped library — every exactness figure in the finding's §2 | `verify_bench_models.jl` + `gate_bench_libs.jl` |

Quote a row with `export/bench/summarise_rows.py <file> [tags] --series both`, never by reading
a number off a line: the published statistic is the pooled median over included blocks, and the
tool applies the protocol's exclusions rather than leaving them to the reader.
