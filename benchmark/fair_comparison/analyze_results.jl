# Analyze benchmark results and generate comparison report
# Parses LAMMPS log files and creates a comprehensive comparison table

using Dates

# Load unified parameters for the report
include("params.jl")
using .FairParams

const RESULTS_DIR = joinpath(@__DIR__, "results")
const METHODS = ["oldace", "mlpace", "etace_spline", "etace_poly"]
const METHOD_NAMES = Dict(
    "oldace" => "Old ACE (juliac)",
    "mlpace" => "ML-PACE (native)",
    "etace_spline" => "ETACE Spline",
    "etace_poly" => "ETACE Polynomial"
)
const PROCS = [1, 2, 4, 8]

"""
Parse a LAMMPS log file to extract benchmark metrics
"""
function parse_lammps_log(filepath::String)
    if !isfile(filepath)
        return nothing
    end

    content = read(filepath, String)

    metrics = Dict{String, Any}()

    # Extract loop time (total simulation time for 100 steps)
    m = match(r"Loop time of ([\d.]+) on (\d+) procs", content)
    if m !== nothing
        metrics["loop_time"] = parse(Float64, m.captures[1])
        metrics["nprocs"] = parse(Int, m.captures[2])
    end

    # Extract pair time (time spent in potential evaluation)
    # Look for the timing breakdown table
    m = match(r"Pair\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)", content)
    if m !== nothing
        metrics["pair_time_avg"] = parse(Float64, m.captures[2])
    end

    # Extract performance in ns/day
    m = match(r"Performance:\s+([\d.]+)\s+ns/day", content)
    if m !== nothing
        metrics["ns_per_day"] = parse(Float64, m.captures[1])
    end

    # Extract timesteps per second
    m = match(r"([\d.]+)\s+timesteps/s", content)
    if m !== nothing
        metrics["timesteps_per_sec"] = parse(Float64, m.captures[1])
    end

    # Extract final potential energy
    m = match(r"Final potential energy:\s+([-\d.]+)\s+eV", content)
    if m !== nothing
        metrics["final_pe"] = parse(Float64, m.captures[1])
    end

    return metrics
end

"""
Generate the comparison report
"""
function generate_report()
    println("="^70)
    println("Fair Performance Comparison Results")
    println("="^70)
    println()
    println("Generated: $(Dates.now())")
    println()

    # Print model parameters
    println("## Model Parameters (ALL METHODS)")
    println("-"^40)
    println("- Elements:      $(join(ELEMENTS, ", "))")
    println("- Order:         $ORDER")
    println("- Total degree:  $TOTALDEGREE")
    println("- Max L:         $MAXL")
    println("- Cutoff:        $RCUT Å")
    println("- Training:      Every $(TRAIN_STRIDE)th configuration")
    println("- Solver:        QR(lambda=$SOLVER_LAMBDA)")
    println("- Prior:         algebraic_smoothness_prior(p=$PRIOR_P)")
    println()

    # Parse all results
    results = Dict()
    for method in METHODS
        results[method] = Dict()
        for np in PROCS
            logfile = joinpath(RESULTS_DIR, "fair_$(method)_np$(np).log")
            metrics = parse_lammps_log(logfile)
            results[method][np] = metrics
        end
    end

    # Generate performance table
    println("## Performance Results (100 MD steps, 2000 atoms B2 TiAl)")
    println()

    # Header
    header = "| Method                | "
    for np in PROCS
        header *= "np=$np     | "
    end
    header *= "Scaling (1→8) |"
    println(header)

    # Separator
    sep = "|" * "-"^23 * "|"
    for _ in PROCS
        sep *= "-"^11 * "|"
    end
    sep *= "-"^15 * "|"
    println(sep)

    # Data rows
    for method in METHODS
        row = "| $(rpad(METHOD_NAMES[method], 21)) | "

        times = Float64[]
        for np in PROCS
            metrics = results[method][np]
            if metrics !== nothing && haskey(metrics, "loop_time")
                t = metrics["loop_time"]
                push!(times, t)
                row *= "$(lpad(round(t, digits=2), 6))s   | "
            else
                push!(times, NaN)
                row *= "   N/A    | "
            end
        end

        # Calculate scaling
        if !isnan(times[1]) && !isnan(times[end]) && times[end] > 0
            scaling = times[1] / times[end]
            row *= "$(lpad(round(scaling, digits=1), 11))x |"
        else
            row *= "         N/A |"
        end

        println(row)
    end
    println()

    # Pair time comparison (single-threaded)
    println("## Pair Evaluation Time (np=1)")
    println()
    println("| Method                | Pair time (s) | % of total |")
    println("|" * "-"^23 * "|" * "-"^15 * "|" * "-"^12 * "|")

    for method in METHODS
        metrics = results[method][1]
        if metrics !== nothing && haskey(metrics, "pair_time_avg") && haskey(metrics, "loop_time")
            pair_t = metrics["pair_time_avg"]
            total_t = metrics["loop_time"]
            pct = round(100 * pair_t / total_t, digits=1)
            println("| $(rpad(METHOD_NAMES[method], 21)) | $(lpad(round(pair_t, digits=2), 13)) | $(lpad(pct, 9))% |")
        else
            println("| $(rpad(METHOD_NAMES[method], 21)) |           N/A |        N/A |")
        end
    end
    println()

    # Calculate speedups relative to baseline (Old ACE at np=1)
    println("## Relative Performance (vs Old ACE at np=1)")
    println()

    baseline = results["oldace"][1]
    if baseline !== nothing && haskey(baseline, "loop_time")
        baseline_time = baseline["loop_time"]
        println("Baseline: Old ACE (np=1) = $(round(baseline_time, digits=2))s")
        println()
        println("| Method                | np=1 speedup | np=8 speedup |")
        println("|" * "-"^23 * "|" * "-"^14 * "|" * "-"^14 * "|")

        for method in METHODS
            row = "| $(rpad(METHOD_NAMES[method], 21)) | "

            # np=1 speedup
            m1 = results[method][1]
            if m1 !== nothing && haskey(m1, "loop_time")
                speedup1 = baseline_time / m1["loop_time"]
                row *= "$(lpad(round(speedup1, digits=2), 10))x  | "
            else
                row *= "         N/A | "
            end

            # np=8 speedup
            m8 = results[method][8]
            if m8 !== nothing && haskey(m8, "loop_time")
                speedup8 = baseline_time / m8["loop_time"]
                row *= "$(lpad(round(speedup8, digits=2), 10))x  |"
            else
                row *= "         N/A |"
            end

            println(row)
        end
        println()
    end

    # Energy comparison
    println("## Final Potential Energy (eV)")
    println()
    println("(Checking numerical consistency across methods)")
    println()

    energies = Dict()
    for method in METHODS
        metrics = results[method][1]
        if metrics !== nothing && haskey(metrics, "final_pe")
            energies[method] = metrics["final_pe"]
            println("- $(METHOD_NAMES[method]): $(round(energies[method], digits=3)) eV")
        else
            println("- $(METHOD_NAMES[method]): N/A")
        end
    end
    println()

    # Check energy differences if we have reference
    if haskey(energies, "oldace") && length(energies) > 1
        ref_E = energies["oldace"]
        println("Energy differences (vs Old ACE):")
        for (method, E) in energies
            if method != "oldace"
                diff = E - ref_E
                println("  - $(METHOD_NAMES[method]): $(round(diff, digits=3)) eV ($(round(diff * 1000, digits=1)) meV)")
            end
        end
        println()
    end

    println("="^70)
    println("Report complete")
    println("="^70)
end

# Write report to file as well
function save_report()
    report_file = joinpath(@__DIR__, "FAIR_BENCHMARK_RESULTS.md")
    open(report_file, "w") do io
        redirect_stdout(io) do
            generate_report()
        end
    end
    println("Report saved to: $report_file")
end

# Main
if abspath(PROGRAM_FILE) == @__FILE__
    generate_report()
    save_report()
end
