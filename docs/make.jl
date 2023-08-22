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


Literate.markdown(_tutorial_src * "/first_example_model.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/TiAl_model.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/TiAl_basis.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/smoothness_priors.jl", 
                  _tutorial_out; documenter = true)

Literate.markdown(_tutorial_src * "/dataset_analysis.jl", 
                  _tutorial_out; documenter = true)

# bring back once we fix the JSON interface 
# Literate.markdown(_tutorial_src * "/ACEpotentials_TiAl.jl", 
#                   _tutorial_out; documenter = true)

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
            "literate_tutorials/first_example_model.md",
            "literate_tutorials/TiAl_model.md",
            "literate_tutorials/TiAl_basis.md",
            "literate_tutorials/smoothness_priors.md", 
            "literate_tutorials/dataset_analysis.md", 
            "literate_tutorials/committee.md",
            "literate_tutorials/descriptor.md",
            "tutorials/molly.md",
            "tutorials/lammps.md",
            "tutorials/python_ase.md",
            "tutorials/Solvers.md",
        ],
        "ACEpotentials Internals" => Any[
            "ACEpotentials/all_exported.md",
        ],
        "Outdated" => Any[
            "outdated/acepotentials_overview.md",
            "outdated/fit.md",    
            "outdated/helpers.md",
            "outdated/data.md",
            "outdated/basis.md",   
            "outdated/solver.md",
            "outdated/command_line_old.md",
            "outdated/first_example_json.md",
            # "literate_tutorials/ACEpotentials_TiAl.md",
            # "ACEfit/File IO.md",
            # "ACEfit/Atomic Configurations in Julia.md",
            # "ACEfit/Manipulating potentials.md",
            "outdated/Fitting.md",
        ],
      ],
    )

        # "Using ACE potentials" => Any[
        #     "Using_ACE/python_ase.md",
        #     "Using_ACE/openmm.md",
        # ],
        # "ACE" => Any[
        #     # "ACE/datatypes.md",
        #     # "ACE/create_ACE.md",
        # ],


deploydocs(;
    repo="github.com/ACEsuit/ACEpotentials.jl",
    devbranch="main",
    push_preview=true,
)
