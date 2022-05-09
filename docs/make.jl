using ACE1pack
using Documenter

DocMeta.setdocmeta!(ACE1pack, :DocTestSetup, :(using ACE1pack); recursive=true)

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
    pages=[
        "Home" => "index.md",
        "Getting Started" => Any[
            "gettingstarted/installation.md",
            "gettingstarted/developing.md",
            "gettingstarted/pkg.md",
        ],
        "ACE" => Any[
            "ACE/aceintro.md",
            "ACE/datatypes.md",
            "ACE/create_ACE.md",
        ],
        "IPFitting" => Any[
            "IPFitting/IPFitting.md",
            "IPFitting/File IO.md",
            "IPFitting/Atomic Configurations in Julia.md",
            "IPFitting/Solvers.md",
            "IPFitting/Manipulating potentials.md",
        ],
        "Using ACE potentials" => Any[
            "Using_ACE/python_ase.md",
            "Using_ACE/lammps.md",
            "Using_ACE/openmm.md",
        ],
        "Tutorials" => Any[
            "tutorials/developing.md",
            "tutorials/createsavefit.md",
            "tutorials/lammps.md",
        ],

    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACE1pack.jl",
    devbranch="main",
)
