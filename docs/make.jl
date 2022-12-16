using OpenADMIXTURE
using Documenter

makedocs(;
    modules=[OpenADMIXTURE],
    authors="Seyoon Ko <kos@ucla.edu> and contributors",
    repo="https://github.com/OpenMendel/OpenADMIXTURE.jl/blob/{commit}{path}#L{line}",
    sitename="SKFR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://OpenMendel.github.io/OpenADMIXTURE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/OpenMendel/OpenADMIXTURE.jl",
    devbranch = "main"
)
