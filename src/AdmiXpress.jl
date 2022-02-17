module AdmiXpress
using Random, LoopVectorization
import LinearAlgebra: svd, norm, diag, mul!
import SnpArrays: SnpLinAlg
export AdmixData
include("structs.jl")
include("projections.jl")
include("qp.jl")
include("quasi_newton.jl")
include("loops.jl")
include("algorithms_inner.jl")
include("algorithms_outer.jl")
using Requires, Adapt
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using .CUDA
        include("cuda/structs.jl")
        include("cuda/kernels.jl")
        include("cuda/runners.jl")
    end
end
end
