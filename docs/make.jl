using ACE1pack
using Documenter, Literate 

DocMeta.setdocmeta!(ACE1pack, :DocTestSetup, :(using ACE1pack); recursive=true)


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

Literate.markdown(_tutorial_src * "/ACE1pack_TiAl.jl", 
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
    modules=[ACE1pack],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACE1pack.jl/blob/{commit}{path}#{line}",
    sitename="ACE1pack.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACE1pack.jl",
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
            # "literate_tutorials/ACE1pack_TiAl.md",
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
        "ACE1pack Internals" => Any[
            "ACE1pack/ace1pack_overview.md",
            "ACE1pack/fit.md",    
            "ACE1pack/helpers.md",
            "ACE1pack/data.md",
            "ACE1pack/basis.md",   
            "ACE1pack/solver.md",
            "ACE1pack/all_exported.md",
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
    repo="github.com/ACEsuit/ACE1pack.jl",
    devbranch="main",
    push_preview=true,
)
