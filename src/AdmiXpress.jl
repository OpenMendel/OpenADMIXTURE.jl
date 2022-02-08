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
end
