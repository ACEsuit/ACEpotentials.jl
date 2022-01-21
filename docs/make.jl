using ACE1pack
using Documenter

DocMeta.setdocmeta!(ACE1pack, :DocTestSetup, :(using ACE1pack); recursive=true)

makedocs(;
    modules=[ACE1pack],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    repo="https://github.com/cortner/ACE1pack.jl/blob/{commit}{path}#{line}",
    sitename="ACE1pack.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cortner.github.io/ACE1pack.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cortner/ACE1pack.jl",
    devbranch="main",
)
