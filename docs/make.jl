using ACEpotentials
using Documenter, Literate 

DocMeta.setdocmeta!(ACEpotentials, :DocTestSetup, :(using ACEpotentials); recursive=true)


# ~~~~~~~~~~ Generate the tutorial files  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_tutorial_out = joinpath(@__DIR__(), "src", "literate_tutorials")
_tutorial_src = joinpath(@__DIR__(), "..", "tutorials")

# Literate.markdown(_tutorial_src * "/first_example_basis.jl", 
#                   _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/first_example_model.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/TiAl_model.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/TiAl_basis.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/ACEpotentials_TiAl.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/descriptor.jl",
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/committee.jl",
                  _tutorial_out; documenter = true)

# ???? cf Jump.jl docs, they do also this: 
# postprocess = _link_example,
# # Turn off the footer. We manually add a modified one.
# credit = false,

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


makedocs(;
    modules=[ACEpotentials],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEpotentials.jl/blob/{commit}{path}#{line}",
    sitename="ACEpotentials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEpotentials.jl",
        assets=String[],
    ),
    strict=true,
    pages=[
        "Home" => "index.md",
        "Getting Started" => Any[
            "gettingstarted/installation.md",
            "gettingstarted/pkg.md",
            # "gettingstarted/readinglist.md",
            "gettingstarted/aceintro.md",
            "gettingstarted/parallel-fitting.md",
        ],
        "Tutorials" => Any[
            "tutorials/index.md",
            # "tutorials/first_example_json.md",
            # "literate_tutorials/first_example_basis.md",
            "literate_tutorials/first_example_model.md",
            "literate_tutorials/TiAl_model.md",
            "literate_tutorials/TiAl_basis.md",
            # "literate_tutorials/ACEpotentials_TiAl.md",
            "literate_tutorials/descriptor.md",
            "literate_tutorials/committee.md",
            "tutorials/lammps.md",
            "tutorials/python_ase.md",
            "tutorials/molly.md"
        ],
        # "Using ACE potentials" => Any[
        #     "Using_ACE/python_ase.md",
        #     "Using_ACE/openmm.md",
        # ],
        "Command line" => Any[
            "tutorials/command_line_old.md",
        ],
        "ACEpotentials Internals" => Any[
            "ACEpotentials/acepotentials_overview.md",
            "ACEpotentials/fit.md",    
            "ACEpotentials/helpers.md",
            "ACEpotentials/data.md",
            "ACEpotentials/basis.md",   
            "ACEpotentials/solver.md",
            "ACEpotentials/all_exported.md",
        ],
        # "ACE" => Any[
        #     # "ACE/datatypes.md",
        #     # "ACE/create_ACE.md",
        # ],
        "ACEfit Internals" => Any[
            "ACEfit/Fitting.md",
            # "ACEfit/File IO.md",
            # "ACEfit/Atomic Configurations in Julia.md",
            "ACEfit/Solvers.md",
            # "ACEfit/Manipulating potentials.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACEpotentials.jl",
    devbranch="main",
    push_preview=true,
)
