module OpenADMIXTURE
using Random, LoopVectorization
import LinearAlgebra: svd, norm, diag, mul!, dot
using SnpArrays
using Base.Threads
using SKFR
export AdmixData
using Requires, Adapt
using Polyester
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using .CUDA
        CUDA.allowscalar(true)
        include("cuda/structs.jl")
        include("cuda/kernels.jl")
        include("cuda/runners.jl")
        include("cuda/transfer.jl")
    end
end
include("structs.jl")
include("projections.jl")
include("qp.jl")
include("quasi_newton.jl")
include("loops.jl")
include("algorithms_inner.jl")
include("algorithms_outer.jl")
include("driver.jl")
end
