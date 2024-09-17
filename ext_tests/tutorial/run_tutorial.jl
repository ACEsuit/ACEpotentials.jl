# Run this test using 
#   julia --project=. run_tutorial.jl 
# if the command fails, then clean the folder using 
#   rm ACEpotentials-Tutorial.jl ACEpotentials-Tutorial.ipynb Project.toml Manifest.toml Si_dataset.xyz Si_tiny_tutorial.json

julia_cmd = Base.julia_cmd()
appath = abspath(joinpath(@__DIR__(), "..", ".."))
setuptutorial = """
   begin 
      using Pkg; 
      Pkg.develop(; path = \"$appath\"); 
      using ACEpotentials; 
      ACEpotentials.copy_tutorial();
   end
"""

run(`$julia_cmd --project=. -e $setuptutorial`)

if !isfile("ACEpotentials-Tutorial.ipynb")
   error("Tutorial notebook not installed.")
end

tutorial_file = joinpath(appath, "examples", "Tutorial", "ACEpotentials-Tutorial.jl")
cp(tutorial_file, joinpath(pwd(), "ACEpotentials-Tutorial.jl"); force=true)

run(`$julia_cmd --project=. ACEpotentials-Tutorial.jl`)

@info("Cleaning up")
run(`rm ACEpotentials-Tutorial.jl ACEpotentials-Tutorial.ipynb Project.toml Manifest.toml Si_dataset.xyz Si_tiny_tutorial.json`)
