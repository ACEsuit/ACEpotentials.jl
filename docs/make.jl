# Instructions for building docs locally: 
#   - activate and resolve/up docs Project
#   - ] dev ..  to link to the *current* version of ACEpotentials
#   - julia --project=. make.jl  or   julia --project=docs docs/make.jl
#

using ACEpotentials
using Documenter, Literate 

DocMeta.setdocmeta!(ACEpotentials, :DocTestSetup, :(using ACEpotentials); recursive=true)


# ~~~~~~~~~~ Generate the tutorial files  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_tutorial_out = joinpath(@__DIR__(), "src", "literate_tutorials")
_tutorial_src = joinpath(@__DIR__(), "src", "tutorials")


Literate.markdown(_tutorial_src * "/basic_julia_workflow.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/smoothness_priors.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/dataset_analysis.jl", 
                  _tutorial_out; documenter = true)

# Literate.markdown(_tutorial_src * "/descriptor.jl",
#                   _tutorial_out; documenter = true)


# Literate.markdown(_tutorial_src * "/first_example_model.jl", 
#                   _tutorial_out; documenter = true)

# Literate.markdown(_tutorial_src * "/TiAl_basis.jl", 
#                   _tutorial_out; documenter = true)

# bring back once we fix the JSON interface 
# Literate.markdown(_tutorial_src * "/ACEpotentials_TiAl.jl", 
#                   _tutorial_out; documenter = true)

# Literate.markdown(_tutorial_src * "/committee.jl",
#                   _tutorial_out; documenter = true)

# Literate.markdown(_tutorial_src * "/experimental.jl",
#                   _tutorial_out; documenter = true)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


makedocs(;
    modules=[ACEpotentials],
    # authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEpotentials.jl/blob/{commit}{path}#{line}",
    sitename="ACEpotentials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEpotentials.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => Any[
            "gettingstarted/installation.md",
            "gettingstarted/saving-and-loading.md", 
            ], 
        "Tutorials" => Any[
                "tutorials/index.md",
                "literate_tutorials/basic_julia_workflow.md",
                "literate_tutorials/smoothness_priors.md",
                "literate_tutorials/dataset_analysis.md",
                "tutorials/scripting.md", 
                # "tutorials/lammps.md",
                # "tutorials/python_ase.md",
                # "tutorials/molly.md",
                # "literate_tutorials/descriptor.md",
                # "literate_tutorials/committee.md",
                # "tutorials/AtomsBase_interface.md",
                # "literate_tutorials/experimental.md",
            ],
        "Additional Topics" => Any[
            "gettingstarted/parallel-fitting.md",
            "gettingstarted/aceintro.md",
            "gettingstarted/pkg.md",
        ],
        "Reference" => "all_exported.md",
            ] 
    )


deploydocs(;
    repo="github.com/ACEsuit/ACEpotentials.jl",
    devbranch="main",
    push_preview=true,
)
