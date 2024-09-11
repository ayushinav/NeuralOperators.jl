using Documenter, NeuralOperators

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml"; force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml"; force = true)

ENV["GKSwstype"] = "100"
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("pages.jl")

makedocs(; sitename = "NeuralOperators.jl",
    clean = true,
    doctest = false,
    linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    modules = [NeuralOperators],)
    # format = Documenter.HTML(; assets = ["assets/favicon.ico"],
    #     canonical = "https://docs.sciml.ai/NeuralOperators.jl/stable/"),
    # pages = pages)

deploydocs(; repo = "github.com/SciML/NeuralOperators.jl.git", push_preview = true)