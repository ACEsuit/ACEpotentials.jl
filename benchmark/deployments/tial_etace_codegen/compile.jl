# Compile ETACE models using juliac CLI (not JuliaC package API)
# The CLI approach produces proper ELF shared libraries

mkpath("lib")

# Path to juliac.jl
juliac_path = joinpath(dirname(Sys.BINDIR), "share", "julia", "juliac", "juliac.jl")
julia_path = joinpath(Sys.BINDIR, "julia")
project_dir = @__DIR__

println("Using juliac from: $juliac_path")
println()

# Compile spline-based model
println("Compiling Hermite spline ETACE model...")
spline_lib = joinpath(project_dir, "lib", "libace_etace_spline.so")
spline_model = joinpath(project_dir, "tial_etace_model.jl")

run(`$julia_path --project=$project_dir $juliac_path
     --output-lib $spline_lib
     --experimental --trim=safe --compile-ccallable
     $spline_model`)
println("✓ Spline library compiled!")

# Compile polynomial model
println("\nCompiling polynomial ETACE model...")
poly_lib = joinpath(project_dir, "lib", "libace_etace_poly.so")
poly_model = joinpath(project_dir, "tial_etace_model_poly.jl")

run(`$julia_path --project=$project_dir $juliac_path
     --output-lib $poly_lib
     --experimental --trim=safe --compile-ccallable
     $poly_model`)
println("✓ Polynomial library compiled!")

println("\nLibrary files:")
for f in readdir("lib")
    if endswith(f, ".so")
        sz = filesize(joinpath("lib", f))
        println("  $f: $(round(sz/1024/1024, digits=1)) MB")
        # Check file type
        run(`file $(joinpath("lib", f))`)
    end
end
